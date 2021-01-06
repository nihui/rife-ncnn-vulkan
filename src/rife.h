// rife implemented with ncnn library

#ifndef RIFE_H
#define RIFE_H

#include <string>

// ncnn
#include "net.h"

class RIFE
{
public:
    RIFE(int gpuid, bool tta_mode = false, bool uhd_mode = false, int num_threads = 1);
    ~RIFE();

#if _WIN32
    int load(const std::wstring& modeldir);
#else
    int load(const std::string& modeldir);
#endif

    int process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

    int process_cpu(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net flownet;
    ncnn::Net contextnet;
    ncnn::Net fusionnet;
    ncnn::Pipeline* rife_preproc;
    ncnn::Pipeline* rife_postproc;
    ncnn::Pipeline* rife_flow_tta_avg;
    ncnn::Layer* rife_uhd_downscale_image;
    ncnn::Layer* rife_uhd_upscale_flow;
    ncnn::Layer* rife_uhd_double_flow;
    bool tta_mode;
    bool uhd_mode;
    int num_threads;
};

#endif // RIFE_H
