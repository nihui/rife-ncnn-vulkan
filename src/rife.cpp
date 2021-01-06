// rife implemented with ncnn library

#include "rife.h"

#include <algorithm>
#include <vector>
#include "benchmark.h"

#include "rife_preproc.comp.hex.h"
#include "rife_postproc.comp.hex.h"
#include "rife_preproc_tta.comp.hex.h"
#include "rife_postproc_tta.comp.hex.h"
#include "rife_flow_tta_avg.comp.hex.h"

#include "rife_ops.h"

DEFINE_LAYER_CREATOR(Warp)

RIFE::RIFE(int gpuid, bool _tta_mode, bool _uhd_mode, int _num_threads)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    rife_preproc = 0;
    rife_postproc = 0;
    rife_flow_tta_avg = 0;
    rife_uhd_downscale_image = 0;
    rife_uhd_upscale_flow = 0;
    rife_uhd_double_flow = 0;
    tta_mode = _tta_mode;
    uhd_mode = _uhd_mode;
    num_threads = _num_threads;
}

RIFE::~RIFE()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete rife_preproc;
        delete rife_postproc;
        delete rife_flow_tta_avg;
    }

    if (uhd_mode)
    {
        rife_uhd_downscale_image->destroy_pipeline(flownet.opt);
        delete rife_uhd_downscale_image;

        rife_uhd_upscale_flow->destroy_pipeline(flownet.opt);
        delete rife_uhd_upscale_flow;

        rife_uhd_double_flow->destroy_pipeline(flownet.opt);
        delete rife_uhd_double_flow;
    }
}

#if _WIN32
static void load_param_model(ncnn::Net& net, const std::wstring& modeldir, const wchar_t* name)
{
    wchar_t parampath[256];
    wchar_t modelpath[256];
    swprintf(parampath, 256, L"%s/%s.param", modeldir.c_str(), name);
    swprintf(modelpath, 256, L"%s/%s.bin", modeldir.c_str(), name);

    {
        FILE* fp = _wfopen(parampath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath);
        }

        net.load_param(fp);

        fclose(fp);
    }
    {
        FILE* fp = _wfopen(modelpath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath);
        }

        net.load_model(fp);

        fclose(fp);
    }
}
#else
static void load_param_model(ncnn::Net& net, const std::string& modeldir, const char* name)
{
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/%s.param", modeldir.c_str(), name);
    sprintf(modelpath, "%s/%s.bin", modeldir.c_str(), name);

    net.load_param(parampath);
    net.load_model(modelpath);
}
#endif

#if _WIN32
int RIFE::load(const std::wstring& modeldir)
#else
int RIFE::load(const std::string& modeldir)
#endif
{
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = vkdev ? true : false;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = true;

    flownet.opt = opt;
    contextnet.opt = opt;
    fusionnet.opt = opt;

    flownet.set_vulkan_device(vkdev);
    contextnet.set_vulkan_device(vkdev);
    fusionnet.set_vulkan_device(vkdev);

    flownet.register_custom_layer("rife.Warp", Warp_layer_creator);
    contextnet.register_custom_layer("rife.Warp", Warp_layer_creator);
    fusionnet.register_custom_layer("rife.Warp", Warp_layer_creator);

#if _WIN32
    load_param_model(flownet, modeldir, L"flownet");
    load_param_model(contextnet, modeldir, L"contextnet");
    load_param_model(fusionnet, modeldir, L"fusionnet");
#else
    load_param_model(flownet, modeldir, "flownet");
    load_param_model(contextnet, modeldir, "contextnet");
    load_param_model(fusionnet, modeldir, "fusionnet");
#endif

    // initialize preprocess and postprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_preproc_tta_comp_data, sizeof(rife_preproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_preproc_comp_data, sizeof(rife_preproc_comp_data), opt, spirv);
                }
            }

            rife_preproc = new ncnn::Pipeline(vkdev);
            rife_preproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_postproc_tta_comp_data, sizeof(rife_postproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_postproc_comp_data, sizeof(rife_postproc_comp_data), opt, spirv);
                }
            }

            rife_postproc = new ncnn::Pipeline(vkdev);
            rife_postproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    if (vkdev && tta_mode)
    {
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(rife_flow_tta_avg_comp_data, sizeof(rife_flow_tta_avg_comp_data), opt, spirv);
            }
        }

        std::vector<ncnn::vk_specialization_type> specializations(0);

        rife_flow_tta_avg = new ncnn::Pipeline(vkdev);
        rife_flow_tta_avg->set_optimal_local_size_xyz(8, 8, 1);
        rife_flow_tta_avg->create(spirv.data(), spirv.size() * 4, specializations);
    }

    if (uhd_mode)
    {
        {
            rife_uhd_downscale_image = ncnn::create_layer("Interp");
            rife_uhd_downscale_image->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, 0.5f);
            pd.set(2, 0.5f);
            rife_uhd_downscale_image->load_param(pd);

            rife_uhd_downscale_image->create_pipeline(opt);
        }
        {
            rife_uhd_upscale_flow = ncnn::create_layer("Interp");
            rife_uhd_upscale_flow->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, 2.f);
            pd.set(2, 2.f);
            rife_uhd_upscale_flow->load_param(pd);

            rife_uhd_upscale_flow->create_pipeline(opt);
        }
        {
            rife_uhd_double_flow = ncnn::create_layer("BinaryOp");
            rife_uhd_double_flow->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// mul
            pd.set(1, 1);// with_scalar
            pd.set(2, 2.f);// b
            rife_uhd_double_flow->load_param(pd);

            rife_uhd_double_flow->create_pipeline(opt);
        }
    }

    return 0;
}

int RIFE::process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    if (!vkdev)
    {
        // cpu only
        return process_cpu(in0image, in1image, timestep, outimage);
    }

    if (timestep == 0.f)
    {
        outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        outimage = in1image;
        return 0;
    }

    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = flownet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    if (opt.use_fp16_storage && opt.use_int8_storage)
    {
        in0 = ncnn::Mat(w, h, (unsigned char*)pixel0data, (size_t)channels, 1);
        in1 = ncnn::Mat(w, h, (unsigned char*)pixel1data, (size_t)channels, 1);
    }
    else
    {
#if _WIN32
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
#else
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB, w, h);
#endif
    }

    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    {
        cmd.record_clone(in0, in0_gpu, opt);
        cmd.record_clone(in1, in1_gpu, opt);
    }

    ncnn::VkMat out_gpu;

    if (tta_mode)
    {
        // preproc
        ncnn::VkMat in0_gpu_padded[8];
        ncnn::VkMat in1_gpu_padded[8];
        {
            in0_gpu_padded[0].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[1].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[2].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[3].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[4].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[5].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[6].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[7].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded[0];
            bindings[2] = in0_gpu_padded[1];
            bindings[3] = in0_gpu_padded[2];
            bindings[4] = in0_gpu_padded[3];
            bindings[5] = in0_gpu_padded[4];
            bindings[6] = in0_gpu_padded[5];
            bindings[7] = in0_gpu_padded[6];
            bindings[8] = in0_gpu_padded[7];

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded[0].w;
            constants[4].i = in0_gpu_padded[0].h;
            constants[5].i = in0_gpu_padded[0].cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded[0]);
        }
        {
            in1_gpu_padded[0].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[1].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[2].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[3].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[4].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[5].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[6].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[7].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded[0];
            bindings[2] = in1_gpu_padded[1];
            bindings[3] = in1_gpu_padded[2];
            bindings[4] = in1_gpu_padded[3];
            bindings[5] = in1_gpu_padded[4];
            bindings[6] = in1_gpu_padded[5];
            bindings[7] = in1_gpu_padded[6];
            bindings[8] = in1_gpu_padded[7];

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded[0].w;
            constants[4].i = in1_gpu_padded[0].h;
            constants[5].i = in1_gpu_padded[0].cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded[0]);
        }

        ncnn::VkMat flow[8];
        for (int ti = 0; ti < 8; ti++)
        {
            // flownet
            {
                ncnn::Extractor ex = flownet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                if (uhd_mode)
                {
                    ncnn::VkMat in0_gpu_padded_downscaled;
                    ncnn::VkMat in1_gpu_padded_downscaled;
                    rife_uhd_downscale_image->forward(in0_gpu_padded[ti], in0_gpu_padded_downscaled, cmd, opt);
                    rife_uhd_downscale_image->forward(in1_gpu_padded[ti], in1_gpu_padded_downscaled, cmd, opt);

                    ex.input("input0", in0_gpu_padded_downscaled);
                    ex.input("input1", in1_gpu_padded_downscaled);

                    ncnn::VkMat flow_downscaled;
                    ex.extract("flow", flow_downscaled, cmd);

                    ncnn::VkMat flow_half;
                    rife_uhd_upscale_flow->forward(flow_downscaled, flow_half, cmd, opt);

                    rife_uhd_double_flow->forward(flow_half, flow[ti], cmd, opt);
                }
                else
                {
                    ex.input("input0", in0_gpu_padded[ti]);
                    ex.input("input1", in1_gpu_padded[ti]);
                    ex.extract("flow", flow[ti], cmd);
                }
            }
        }

        // avg flow
        {
            std::vector<ncnn::VkMat> bindings(8);
            bindings[0] = flow[0];
            bindings[1] = flow[1];
            bindings[2] = flow[2];
            bindings[3] = flow[3];
            bindings[4] = flow[4];
            bindings[5] = flow[5];
            bindings[6] = flow[6];
            bindings[7] = flow[7];

            std::vector<ncnn::vk_constant_type> constants(3);
            constants[0].i = flow[0].w;
            constants[1].i = flow[0].h;
            constants[2].i = flow[0].cstep;

            ncnn::VkMat dispatcher;
            dispatcher.w = flow[0].w;
            dispatcher.h = flow[0].h;
            dispatcher.c = 1;
            cmd.record_pipeline(rife_flow_tta_avg, bindings, constants, dispatcher);
        }

        ncnn::VkMat out_gpu_padded[8];
        for (int ti = 0; ti < 8; ti++)
        {
            // contextnet
            ncnn::VkMat ctx0[4];
            ncnn::VkMat ctx1[4];
            {
                ncnn::Extractor ex = contextnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input.1", in0_gpu_padded[ti]);
                ex.input("flow.0", flow[ti]);
                ex.extract("f1", ctx0[0], cmd);
                ex.extract("f2", ctx0[1], cmd);
                ex.extract("f3", ctx0[2], cmd);
                ex.extract("f4", ctx0[3], cmd);
            }
            {
                ncnn::Extractor ex = contextnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input.1", in1_gpu_padded[ti]);
                ex.input("flow.1", flow[ti]);
                ex.extract("f1", ctx1[0], cmd);
                ex.extract("f2", ctx1[1], cmd);
                ex.extract("f3", ctx1[2], cmd);
                ex.extract("f4", ctx1[3], cmd);
            }

            // fusionnet
            {
                ncnn::Extractor ex = fusionnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("img0", in0_gpu_padded[ti]);
                ex.input("img1", in1_gpu_padded[ti]);
                ex.input("flow", flow[ti]);
                ex.input("3", ctx0[0]);
                ex.input("4", ctx0[1]);
                ex.input("5", ctx0[2]);
                ex.input("6", ctx0[3]);
                ex.input("7", ctx1[0]);
                ex.input("8", ctx1[1]);
                ex.input("9", ctx1[2]);
                ex.input("10", ctx1[3]);

                // save some memory
                if (ti == 0)
                {
                    in0_gpu.release();
                    in1_gpu.release();
                }
                else
                {
                    in0_gpu_padded[ti - 1].release();
                    in1_gpu_padded[ti - 1].release();
                    flow[ti - 1].release();
                }
                ctx0[0].release();
                ctx0[1].release();
                ctx0[2].release();
                ctx0[3].release();
                ctx1[0].release();
                ctx1[1].release();
                ctx1[2].release();
                ctx1[3].release();

                ex.extract("output", out_gpu_padded[ti], cmd);
            }
        }

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w, h, (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w, h, channels, (size_t)4u, 1, blob_vkallocator);
        }

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = out_gpu_padded[0];
            bindings[1] = out_gpu_padded[1];
            bindings[2] = out_gpu_padded[2];
            bindings[3] = out_gpu_padded[3];
            bindings[4] = out_gpu_padded[4];
            bindings[5] = out_gpu_padded[5];
            bindings[6] = out_gpu_padded[6];
            bindings[7] = out_gpu_padded[7];
            bindings[8] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded[0].w;
            constants[1].i = out_gpu_padded[0].h;
            constants[2].i = out_gpu_padded[0].cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
        }
    }
    else
    {
        // preproc
        ncnn::VkMat in0_gpu_padded;
        ncnn::VkMat in1_gpu_padded;
        {
            in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded.w;
            constants[4].i = in0_gpu_padded.h;
            constants[5].i = in0_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded);
        }
        {
            in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded.w;
            constants[4].i = in1_gpu_padded.h;
            constants[5].i = in1_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded);
        }

        // flownet
        ncnn::VkMat flow;
        {
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            if (uhd_mode)
            {
                ncnn::VkMat in0_gpu_padded_downscaled;
                ncnn::VkMat in1_gpu_padded_downscaled;
                rife_uhd_downscale_image->forward(in0_gpu_padded, in0_gpu_padded_downscaled, cmd, opt);
                rife_uhd_downscale_image->forward(in1_gpu_padded, in1_gpu_padded_downscaled, cmd, opt);

                ex.input("input0", in0_gpu_padded_downscaled);
                ex.input("input1", in1_gpu_padded_downscaled);

                ncnn::VkMat flow_downscaled;
                ex.extract("flow", flow_downscaled, cmd);

                ncnn::VkMat flow_half;
                rife_uhd_upscale_flow->forward(flow_downscaled, flow_half, cmd, opt);

                rife_uhd_double_flow->forward(flow_half, flow, cmd, opt);
            }
            else
            {
                ex.input("input0", in0_gpu_padded);
                ex.input("input1", in1_gpu_padded);
                ex.extract("flow", flow, cmd);
            }
        }

        // contextnet
        ncnn::VkMat ctx0[4];
        ncnn::VkMat ctx1[4];
        {
            ncnn::Extractor ex = contextnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input.1", in0_gpu_padded);
            ex.input("flow.0", flow);
            ex.extract("f1", ctx0[0], cmd);
            ex.extract("f2", ctx0[1], cmd);
            ex.extract("f3", ctx0[2], cmd);
            ex.extract("f4", ctx0[3], cmd);
        }
        {
            ncnn::Extractor ex = contextnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input.1", in1_gpu_padded);
            ex.input("flow.1", flow);
            ex.extract("f1", ctx1[0], cmd);
            ex.extract("f2", ctx1[1], cmd);
            ex.extract("f3", ctx1[2], cmd);
            ex.extract("f4", ctx1[3], cmd);
        }

        // fusionnet
        ncnn::VkMat out_gpu_padded;
        {
            ncnn::Extractor ex = fusionnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("img0", in0_gpu_padded);
            ex.input("img1", in1_gpu_padded);
            ex.input("flow", flow);
            ex.input("3", ctx0[0]);
            ex.input("4", ctx0[1]);
            ex.input("5", ctx0[2]);
            ex.input("6", ctx0[3]);
            ex.input("7", ctx1[0]);
            ex.input("8", ctx1[1]);
            ex.input("9", ctx1[2]);
            ex.input("10", ctx1[3]);

            // save some memory
            in0_gpu.release();
            in1_gpu.release();
            flow.release();
            ctx0[0].release();
            ctx0[1].release();
            ctx0[2].release();
            ctx0[3].release();
            ctx1[0].release();
            ctx1[1].release();
            ctx1[2].release();
            ctx1[3].release();

            ex.extract("output", out_gpu_padded, cmd);
        }

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w, h, (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w, h, channels, (size_t)4u, 1, blob_vkallocator);
        }

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = out_gpu_padded;
            bindings[1] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded.w;
            constants[1].i = out_gpu_padded.h;
            constants[2].i = out_gpu_padded.cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
        }
    }

    // download
    {
        ncnn::Mat out;

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char*)outimage.data, (size_t)channels, 1);
        }

        cmd.record_clone(out_gpu, out, opt);

        cmd.submit_and_wait();

        if (!(opt.use_fp16_storage && opt.use_int8_storage))
        {
#if _WIN32
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
#else
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RIFE::process_cpu(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    if (timestep == 0.f)
    {
        outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        outimage = in1image;
        return 0;
    }

    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::Option opt = flownet.opt;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    ncnn::Mat in0;
    ncnn::Mat in1;
    {
#if _WIN32
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
#else
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB, w, h);
#endif
    }

    ncnn::Mat out;

    if (tta_mode)
    {
        // preproc and border padding
        ncnn::Mat in0_padded[8];
        ncnn::Mat in1_padded[8];
        {
            in0_padded[0].create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in0_padded[0].channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in0.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f);
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            in1_padded[0].create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in1_padded[0].channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in1.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f);
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }

        // the other 7 directions
        {
            in0_padded[1].create(w_padded, h_padded, 3);
            in0_padded[2].create(w_padded, h_padded, 3);
            in0_padded[3].create(w_padded, h_padded, 3);
            in0_padded[4].create(h_padded, w_padded, 3);
            in0_padded[5].create(h_padded, w_padded, 3);
            in0_padded[6].create(h_padded, w_padded, 3);
            in0_padded[7].create(h_padded, w_padded, 3);

            for (int q = 0; q < 3; q++)
            {
                const ncnn::Mat in0_padded_0 = in0_padded[0].channel(q);
                ncnn::Mat in0_padded_1 = in0_padded[1].channel(q);
                ncnn::Mat in0_padded_2 = in0_padded[2].channel(q);
                ncnn::Mat in0_padded_3 = in0_padded[3].channel(q);
                ncnn::Mat in0_padded_4 = in0_padded[4].channel(q);
                ncnn::Mat in0_padded_5 = in0_padded[5].channel(q);
                ncnn::Mat in0_padded_6 = in0_padded[6].channel(q);
                ncnn::Mat in0_padded_7 = in0_padded[7].channel(q);

                for (int i = 0; i < h_padded; i++)
                {
                    const float* outptr0 = in0_padded_0.row(i);
                    float* outptr1 = in0_padded_1.row(i) + w_padded - 1;
                    float* outptr2 = in0_padded_2.row(h_padded - 1 - i) + w_padded - 1;
                    float* outptr3 = in0_padded_3.row(h_padded - 1 - i);

                    for (int j = 0; j < w_padded; j++)
                    {
                        float* outptr4 = in0_padded_4.row(j) + i;
                        float* outptr5 = in0_padded_5.row(j) + h_padded - 1 - i;
                        float* outptr6 = in0_padded_6.row(w_padded - 1 - j) + h_padded - 1 - i;
                        float* outptr7 = in0_padded_7.row(w_padded - 1 - j) + i;

                        float v = *outptr0++;

                        *outptr1-- = v;
                        *outptr2-- = v;
                        *outptr3++ = v;
                        *outptr4 = v;
                        *outptr5 = v;
                        *outptr6 = v;
                        *outptr7 = v;
                    }
                }
            }
        }
        {
            in1_padded[1].create(w_padded, h_padded, 3);
            in1_padded[2].create(w_padded, h_padded, 3);
            in1_padded[3].create(w_padded, h_padded, 3);
            in1_padded[4].create(h_padded, w_padded, 3);
            in1_padded[5].create(h_padded, w_padded, 3);
            in1_padded[6].create(h_padded, w_padded, 3);
            in1_padded[7].create(h_padded, w_padded, 3);

            for (int q = 0; q < 3; q++)
            {
                const ncnn::Mat in1_padded_0 = in1_padded[0].channel(q);
                ncnn::Mat in1_padded_1 = in1_padded[1].channel(q);
                ncnn::Mat in1_padded_2 = in1_padded[2].channel(q);
                ncnn::Mat in1_padded_3 = in1_padded[3].channel(q);
                ncnn::Mat in1_padded_4 = in1_padded[4].channel(q);
                ncnn::Mat in1_padded_5 = in1_padded[5].channel(q);
                ncnn::Mat in1_padded_6 = in1_padded[6].channel(q);
                ncnn::Mat in1_padded_7 = in1_padded[7].channel(q);

                for (int i = 0; i < h_padded; i++)
                {
                    const float* outptr0 = in1_padded_0.row(i);
                    float* outptr1 = in1_padded_1.row(i) + w_padded - 1;
                    float* outptr2 = in1_padded_2.row(h_padded - 1 - i) + w_padded - 1;
                    float* outptr3 = in1_padded_3.row(h_padded - 1 - i);

                    for (int j = 0; j < w_padded; j++)
                    {
                        float* outptr4 = in1_padded_4.row(j) + i;
                        float* outptr5 = in1_padded_5.row(j) + h_padded - 1 - i;
                        float* outptr6 = in1_padded_6.row(w_padded - 1 - j) + h_padded - 1 - i;
                        float* outptr7 = in1_padded_7.row(w_padded - 1 - j) + i;

                        float v = *outptr0++;

                        *outptr1-- = v;
                        *outptr2-- = v;
                        *outptr3++ = v;
                        *outptr4 = v;
                        *outptr5 = v;
                        *outptr6 = v;
                        *outptr7 = v;
                    }
                }
            }
        }

        ncnn::Mat flow[8];
        for (int ti = 0; ti < 8; ti++)
        {
            // flownet
            {
                ncnn::Extractor ex = flownet.create_extractor();

                if (uhd_mode)
                {
                    ncnn::Mat in0_padded_downscaled;
                    ncnn::Mat in1_padded_downscaled;
                    rife_uhd_downscale_image->forward(in0_padded[ti], in0_padded_downscaled, opt);
                    rife_uhd_downscale_image->forward(in1_padded[ti], in1_padded_downscaled, opt);

                    ex.input("input0", in0_padded_downscaled);
                    ex.input("input1", in1_padded_downscaled);

                    ncnn::Mat flow_downscaled;
                    ex.extract("flow", flow_downscaled);

                    ncnn::Mat flow_half;
                    rife_uhd_upscale_flow->forward(flow_downscaled, flow_half, opt);

                    rife_uhd_double_flow->forward(flow_half, flow[ti], opt);
                }
                else
                {
                    ex.input("input0", in0_padded[ti]);
                    ex.input("input1", in1_padded[ti]);
                    ex.extract("flow", flow[ti]);
                }
            }
        }

        // avg flow
        {
            ncnn::Mat flow_x0 = flow[0].channel(0);
            ncnn::Mat flow_x1 = flow[1].channel(0);
            ncnn::Mat flow_x2 = flow[2].channel(0);
            ncnn::Mat flow_x3 = flow[3].channel(0);
            ncnn::Mat flow_x4 = flow[4].channel(0);
            ncnn::Mat flow_x5 = flow[5].channel(0);
            ncnn::Mat flow_x6 = flow[6].channel(0);
            ncnn::Mat flow_x7 = flow[7].channel(0);

            ncnn::Mat flow_y0 = flow[0].channel(1);
            ncnn::Mat flow_y1 = flow[1].channel(1);
            ncnn::Mat flow_y2 = flow[2].channel(1);
            ncnn::Mat flow_y3 = flow[3].channel(1);
            ncnn::Mat flow_y4 = flow[4].channel(1);
            ncnn::Mat flow_y5 = flow[5].channel(1);
            ncnn::Mat flow_y6 = flow[6].channel(1);
            ncnn::Mat flow_y7 = flow[7].channel(1);

            for (int i = 0; i < flow_x0.h; i++)
            {
                float* x0 = flow_x0.row(i);
                float* x1 = flow_x1.row(i) + flow_x0.w - 1;
                float* x2 = flow_x2.row(flow_x0.h - 1 - i) + flow_x0.w - 1;
                float* x3 = flow_x3.row(flow_x0.h - 1 - i);

                float* y0 = flow_y0.row(i);
                float* y1 = flow_y1.row(i) + flow_x0.w - 1;
                float* y2 = flow_y2.row(flow_x0.h - 1 - i) + flow_x0.w - 1;
                float* y3 = flow_y3.row(flow_x0.h - 1 - i);

                for (int j = 0; j < flow_x0.w; j++)
                {
                    float* x4 = flow_x4.row(j) + i;
                    float* x5 = flow_x5.row(j) + flow_x0.h - 1 - i;
                    float* x6 = flow_x6.row(flow_x0.w - 1 - j) + flow_x0.h - 1 - i;
                    float* x7 = flow_x7.row(flow_x0.w - 1 - j) + i;

                    float* y4 = flow_y4.row(j) + i;
                    float* y5 = flow_y5.row(j) + flow_x0.h - 1 - i;
                    float* y6 = flow_y6.row(flow_x0.w - 1 - j) + flow_x0.h - 1 - i;
                    float* y7 = flow_y7.row(flow_x0.w - 1 - j) + i;

                    float x = (*x0 + -*x1 + -*x2 + *x3 + *y4 + *y5 + -*y6 + -*y7) * 0.125f;
                    float y = (*y0 + *y1 + -*y2 + -*y3 + *x4 + -*x5 + -*x6 + *x7) * 0.125f;

                    *x0++ = x;
                    *x1-- = -x;
                    *x2-- = -x;
                    *x3++ = x;
                    *x4 = y;
                    *x5 = -y;
                    *x6 = -y;
                    *x7 = y;

                    *y0++ = y;
                    *y1-- = y;
                    *y2-- = -y;
                    *y3++ = -y;
                    *y4 = x;
                    *y5 = x;
                    *y6 = -x;
                    *y7 = -x;
                }
            }
        }

        ncnn::Mat out_padded[8];
        for (int ti = 0; ti < 8; ti++)
        {
            // contextnet
            ncnn::Mat ctx0[4];
            ncnn::Mat ctx1[4];
            {
                ncnn::Extractor ex = contextnet.create_extractor();

                ex.input("input.1", in0_padded[ti]);
                ex.input("flow.0", flow[ti]);
                ex.extract("f1", ctx0[0]);
                ex.extract("f2", ctx0[1]);
                ex.extract("f3", ctx0[2]);
                ex.extract("f4", ctx0[3]);
            }
            {
                ncnn::Extractor ex = contextnet.create_extractor();

                ex.input("input.1", in1_padded[ti]);
                ex.input("flow.1", flow[ti]);
                ex.extract("f1", ctx1[0]);
                ex.extract("f2", ctx1[1]);
                ex.extract("f3", ctx1[2]);
                ex.extract("f4", ctx1[3]);
            }

            // fusionnet
            {
                ncnn::Extractor ex = fusionnet.create_extractor();

                ex.input("img0", in0_padded[ti]);
                ex.input("img1", in1_padded[ti]);
                ex.input("flow", flow[ti]);
                ex.input("3", ctx0[0]);
                ex.input("4", ctx0[1]);
                ex.input("5", ctx0[2]);
                ex.input("6", ctx0[3]);
                ex.input("7", ctx1[0]);
                ex.input("8", ctx1[1]);
                ex.input("9", ctx1[2]);
                ex.input("10", ctx1[3]);

                ex.extract("output", out_padded[ti]);
            }
        }

        // cut padding and postproc
        {
            out.create(w, h, 3);
            for (int q = 0; q < 3; q++)
            {
                const ncnn::Mat out_padded_0 = out_padded[0].channel(q);
                const ncnn::Mat out_padded_1 = out_padded[1].channel(q);
                const ncnn::Mat out_padded_2 = out_padded[2].channel(q);
                const ncnn::Mat out_padded_3 = out_padded[3].channel(q);
                const ncnn::Mat out_padded_4 = out_padded[4].channel(q);
                const ncnn::Mat out_padded_5 = out_padded[5].channel(q);
                const ncnn::Mat out_padded_6 = out_padded[6].channel(q);
                const ncnn::Mat out_padded_7 = out_padded[7].channel(q);
                float* outptr = out.channel(q);

                for (int i = 0; i < h; i++)
                {
                    const float* ptr0 = out_padded_0.row(i);
                    const float* ptr1 = out_padded_1.row(i) + w_padded - 1;
                    const float* ptr2 = out_padded_2.row(h_padded - 1 - i) + w_padded - 1;
                    const float* ptr3 = out_padded_3.row(h_padded - 1 - i);

                    for (int j = 0; j < w; j++)
                    {
                        const float* ptr4 = out_padded_4.row(j) + i;
                        const float* ptr5 = out_padded_5.row(j) + h_padded - 1 - i;
                        const float* ptr6 = out_padded_6.row(w_padded - 1 - j) + h_padded - 1 - i;
                        const float* ptr7 = out_padded_7.row(w_padded - 1 - j) + i;

                        float v = (*ptr0++ + *ptr1-- + *ptr2-- + *ptr3++ + *ptr4 + *ptr5 + *ptr6 + *ptr7) / 8;

                        *outptr++ = v * 255.f + 0.5f;
                    }
                }
            }
        }
    }
    else
    {
        // preproc and border padding
        ncnn::Mat in0_padded;
        ncnn::Mat in1_padded;
        {
            in0_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in0_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in0.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f);
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            in1_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in1_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in1.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f);
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }

        // flownet
        ncnn::Mat flow;
        {
            ncnn::Extractor ex = flownet.create_extractor();

            if (uhd_mode)
            {
                ncnn::Mat in0_padded_downscaled;
                ncnn::Mat in1_padded_downscaled;
                rife_uhd_downscale_image->forward(in0_padded, in0_padded_downscaled, opt);
                rife_uhd_downscale_image->forward(in1_padded, in1_padded_downscaled, opt);

                ex.input("input0", in0_padded_downscaled);
                ex.input("input1", in1_padded_downscaled);

                ncnn::Mat flow_downscaled;
                ex.extract("flow", flow_downscaled);

                ncnn::Mat flow_half;
                rife_uhd_upscale_flow->forward(flow_downscaled, flow_half, opt);

                rife_uhd_double_flow->forward(flow_half, flow, opt);
            }
            else
            {
                ex.input("input0", in0_padded);
                ex.input("input1", in1_padded);
                ex.extract("flow", flow);
            }
        }

        // contextnet
        ncnn::Mat ctx0[4];
        ncnn::Mat ctx1[4];
        {
            ncnn::Extractor ex = contextnet.create_extractor();

            ex.input("input.1", in0_padded);
            ex.input("flow.0", flow);
            ex.extract("f1", ctx0[0]);
            ex.extract("f2", ctx0[1]);
            ex.extract("f3", ctx0[2]);
            ex.extract("f4", ctx0[3]);
        }
        {
            ncnn::Extractor ex = contextnet.create_extractor();

            ex.input("input.1", in1_padded);
            ex.input("flow.1", flow);
            ex.extract("f1", ctx1[0]);
            ex.extract("f2", ctx1[1]);
            ex.extract("f3", ctx1[2]);
            ex.extract("f4", ctx1[3]);
        }

        // fusionnet
        ncnn::Mat out_padded;
        {
            ncnn::Extractor ex = fusionnet.create_extractor();

            ex.input("img0", in0_padded);
            ex.input("img1", in1_padded);
            ex.input("flow", flow);
            ex.input("3", ctx0[0]);
            ex.input("4", ctx0[1]);
            ex.input("5", ctx0[2]);
            ex.input("6", ctx0[3]);
            ex.input("7", ctx1[0]);
            ex.input("8", ctx1[1]);
            ex.input("9", ctx1[2]);
            ex.input("10", ctx1[3]);

            ex.extract("output", out_padded);
        }

        // cut padding and postproc
        {
            out.create(w, h, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = out.channel(q);

                for (int i = 0; i < h; i++)
                {
                    const float* ptr = out_padded.channel(q).row(i);

                    for (int j = 0; j < w; j++)
                    {
                        *outptr++ = *ptr++ * 255.f + 0.5f;
                    }
                }
            }
        }
    }

    // download
    {
#if _WIN32
        out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
#else
        out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif
    }

    return 0;
}
