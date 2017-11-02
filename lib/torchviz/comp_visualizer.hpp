#ifndef SIMCTRL_VISUALIZER_HPP
#define SIMCTRL_VISUALIZER_HPP

// Boost thread/mutex
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>

// Eigen
#include <Eigen/Dense>

// OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

// DART
#include "tracker.h"

//////////////////////
/// \brief The PyCompData class - Class that encapsulates all data we want to pass from Python to the pangolin visualizer
///
class PyCompData
{
public:
    // Members
    int imgHeight, imgWidth;
    float imgScale;
    float fx, fy, cx, cy; // Camera parameters
    boost::mutex dataMutex;

    // Init done
    bool init_done;
    int showposepts;
    int nSE3;

    // Input, target & current pts for display
    float *inp_cloud, *gt_cloud, *se3pose_cloud, *se3posepts_cloud, *se3_cloud, *flow_cloud;
    float *gt_mask, *se3pose_mask, *se3_mask, *se3posepts_mask;

    // Empty Constructor
    PyCompData();

    // Better constructor
    PyCompData(const int &_imgHeight, const int &_imgWidth, const float &_imgScale,
               const float &_fx, const float &_fy, const float &_cx, const float &_cy,
               int _showposepts):
            imgHeight(_imgHeight), imgWidth(_imgWidth), imgScale(_imgScale),
            fx(_fx), fy(_fy), cx(_cx), cy(_cy), showposepts(_showposepts)
    {
        // Init
        init_done = false;
        nSE3 = 8;

        // Input / Target / Current clouds
        inp_cloud     = new float[3 * imgHeight * imgWidth];
        gt_cloud      = new float[3 * imgHeight * imgWidth];
        se3pose_cloud = new float[3 * imgHeight * imgWidth];
        se3posepts_cloud = new float[3 * imgHeight * imgWidth];
        se3_cloud     = new float[3 * imgHeight * imgWidth];
        flow_cloud    = new float[3 * imgHeight * imgWidth];
        memset(inp_cloud,     0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(gt_cloud,      0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(se3pose_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(se3posepts_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(se3_cloud,     0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(flow_cloud,    0, 3 * imgHeight * imgWidth * sizeof(float));

        gt_mask      = new float[nSE3 * imgHeight * imgWidth];
        se3pose_mask = new float[nSE3 * imgHeight * imgWidth];
        se3posepts_mask = new float[nSE3 * imgHeight * imgWidth];
        se3_mask     = new float[nSE3 * imgHeight * imgWidth];
        memset(gt_mask,      0, nSE3 * imgHeight * imgWidth * sizeof(float));
        memset(se3pose_mask, 0, nSE3 * imgHeight * imgWidth * sizeof(float));
        memset(se3posepts_mask, 0, nSE3 * imgHeight * imgWidth * sizeof(float));
        memset(se3_mask,     0, nSE3 * imgHeight * imgWidth * sizeof(float));
    }

    // Copy data
    void copyData(boost::shared_ptr<PyCompData> data)
    {
        memcpy(inp_cloud, data->inp_cloud, 3 * imgHeight * imgWidth * sizeof(float));
        memcpy(gt_cloud, data->gt_cloud, 3 * imgHeight * imgWidth * sizeof(float));
        memcpy(se3pose_cloud, data->se3pose_cloud, 3 * imgHeight * imgWidth * sizeof(float));
        memcpy(se3posepts_cloud, data->se3posepts_cloud, 3 * imgHeight * imgWidth * sizeof(float));
        memcpy(se3_cloud, data->se3_cloud, 3 * imgHeight * imgWidth * sizeof(float));
        memcpy(flow_cloud, data->flow_cloud, 3 * imgHeight * imgWidth * sizeof(float));
        memcpy(gt_mask, data->gt_mask, nSE3 * imgHeight * imgWidth * sizeof(float));
        memcpy(se3pose_mask, data->se3pose_mask, nSE3 * imgHeight * imgWidth * sizeof(float));
        memcpy(se3posepts_mask, data->se3posepts_mask, nSE3 * imgHeight * imgWidth * sizeof(float));
        memcpy(se3_mask, data->se3_mask, nSE3 * imgHeight * imgWidth * sizeof(float));
    }

    // Free members alone
    void freeMemberMemory()
    {
        free(inp_cloud);
        free(gt_cloud);
        free(se3pose_cloud);
        free(se3_cloud);
        free(se3posepts_cloud);
        free(flow_cloud);
        free(gt_mask);
        free(se3pose_mask);
        free(se3posepts_mask);
        free(se3_mask);
    }

    // Destructor
    ~PyCompData()
    {
        // Free member memory
        freeMemberMemory();
    }
};

//////////////// == Static global variables == /////////////////////

class CompViz
{
    public:
        CompViz(int imgHeight, int imgWidth, float imgScale,
                float fx, float fy, float cx, float cy,
                int showposepts);
        ~CompViz();

        void update(const float *inp_cloud,
                    const float *gt_cloud,
                    const float *se3pose_cloud,
                    const float *se3posepts_cloud,
                    const float *se3_cloud,
                    const float *flow_cloud,
                    const float *gt_mask,
                    const float *se3pose_mask,
                    const float *se3posepts_mask,
                    const float *se3_mask,
                    int save_frame);

        void start_saving_frames(const std::string framesavedir);

        void stop_saving_frames();

    private:
        boost::shared_ptr<PyCompData> data;
        boost::shared_ptr<boost::thread> pangolin_gui_thread;
};

#endif // REALCTRL_VISUALIZER_HPP
