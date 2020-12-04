// rife implemented with ncnn library

#include "rife.h"

#include <algorithm>
#include <vector>
#include "benchmark.h"

#include "rife_preproc.comp.hex.h"
#include "rife_postproc.comp.hex.h"

#include "rife_ops.h"

DEFINE_LAYER_CREATOR(Warp)

RIFE::RIFE(int gpuid)
{
    vkdev = ncnn::get_gpu_device(gpuid);
    rife_preproc = 0;
    rife_postproc = 0;
}

RIFE::~RIFE()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete rife_preproc;
        delete rife_postproc;
    }
}

int RIFE::load()
{
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
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

    flownet.load_param("flownet.param");
    flownet.load_model("flownet.bin");

    contextnet.load_param("contextnet.param");
    contextnet.load_model("contextnet.bin");

    fusionnet.load_param("fusionnet.param");
    fusionnet.load_model("fusionnet.bin");

    // initialize preprocess and postprocess pipeline
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
                    compile_spirv_module(rife_postproc_comp_data, sizeof(rife_postproc_comp_data), opt, spirv);
                }
            }

            rife_postproc = new ncnn::Pipeline(vkdev);
            rife_postproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

int RIFE::process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
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
    if (opt.use_fp16_storage && opt.use_int8_storage)
    {
        out_gpu.create(w, h, (size_t)channels, 1, blob_vkallocator);
    }
    else
    {
        out_gpu.create(w, h, channels, (size_t)4u, 1, blob_vkallocator);
    }

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

        ex.input("input0", in0_gpu_padded);
        ex.input("input1", in1_gpu_padded);
        ex.extract("flow", flow, cmd);
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
