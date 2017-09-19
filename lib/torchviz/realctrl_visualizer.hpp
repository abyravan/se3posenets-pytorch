#ifndef REALCTRL_VISUALIZER_HPP
#define REALCTRL_VISUALIZER_HPP

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
/// \brief The PyRealData class - Class that encapsulates all data we want to pass from Python to the pangolin visualizer
///
class PyRealData
{
public:
    // Members
    int imgHeight, imgWidth;
    float imgScale;
    float fx, fy, cx, cy; // Camera parameters
    boost::mutex dataMutex;

    // Render data
    float *rendered_img;
    float *rendered_img_sub;
    float *rendered_vertmap;
    float *rendered_vertmap_sub;
    float *render_jts;
    bool render;
    boost::mutex renderMutex;

    // GT rendering data
    Eigen::Matrix4f modelView;

    // Init done
    bool init_done;

    // Input, target & current pts for display
    float *init_cloud, *curr_cloud, *final_cloud;
    float *initnorm_cloud, *currnorm_cloud, *finalnorm_cloud;
    float *init_masks, *curr_masks, *final_masks;
    float *init_poses, *curr_poses, *final_poses;
    float *init_jts, *curr_jts, *final_jts;
    float *curr_labels;
    int nSE3;
    std::string savedir;

    // Errors
    std::vector<float> pose_errors;
    std::vector<std::vector<float> > deg_errors;

    // Empty Constructor
    PyRealData();

    // Better constructor
    PyRealData(const int &_imgHeight, const int &_imgWidth, const float &_imgScale, const int _nSE3,
            const float &_fx, const float &_fy, const float &_cx, const float &_cy, const std::string _savedir):
            imgHeight(_imgHeight), imgWidth(_imgWidth), imgScale(_imgScale), nSE3(_nSE3),
            fx(_fx), fy(_fy), cx(_cx), cy(_cy), savedir(_savedir)
    {
        // Render data
        render_jts   = new float[7];
        render = false;
        init_done = false;
        rendered_img = new float[640 * 480];
        rendered_img_sub = new float[imgWidth * imgHeight];
        rendered_vertmap = new float[640 * 480 * 4];
        rendered_vertmap_sub = new float[imgWidth * imgHeight * 4];

        // Input / Target / Current clouds
        init_cloud   = new float[3 * imgHeight * imgWidth];
        curr_cloud   = new float[3 * imgHeight * imgWidth];
        final_cloud  = new float[3 * imgHeight * imgWidth];
        memset(init_cloud,  0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(curr_cloud,  0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(final_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));

        // Input / Target / Current clouds - Normals
        initnorm_cloud   = new float[3 * imgHeight * imgWidth];
        currnorm_cloud   = new float[3 * imgHeight * imgWidth];
        finalnorm_cloud  = new float[3 * imgHeight * imgWidth];
        memset(initnorm_cloud,  0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(currnorm_cloud,  0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(finalnorm_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));

        // Init masks
        init_masks   = new float[nSE3 * imgHeight * imgWidth];
        curr_masks   = new float[nSE3 * imgHeight * imgWidth];
        final_masks  = new float[nSE3 * imgHeight * imgWidth];
        curr_labels  = new float[imgHeight * imgWidth];
        memset(init_masks,  0, nSE3 * imgHeight * imgWidth * sizeof(float));
        memset(curr_masks,  0, nSE3 * imgHeight * imgWidth * sizeof(float));
        memset(final_masks, 0, nSE3 * imgHeight * imgWidth * sizeof(float));
        memset(curr_labels,  0, imgHeight * imgWidth * sizeof(float));

        // Init poses
        init_poses   = new float [nSE3 * 12];
        curr_poses   = new float [nSE3 * 12];
        final_poses  = new float [nSE3 * 12];
        memset(init_poses,  0, nSE3 * 12 * sizeof(float));
        memset(curr_poses,  0, nSE3 * 12 * sizeof(float));
        memset(final_poses, 0, nSE3 * 12 * sizeof(float));

        // Input / Target / Current jts
        init_jts  = new float[7];
        curr_jts  = new float[7];
        final_jts = new float[7];
        memset(init_jts,  0, 7 * sizeof(float));
        memset(curr_jts,  0, 7 * sizeof(float));
        memset(final_jts, 0, 7 * sizeof(float));
    }

    // Copy data
    void copyData(boost::shared_ptr<PyRealData> data)
    {
        memcpy(init_cloud, data->init_cloud, 3 * imgHeight * imgWidth * sizeof(float));
        memcpy(curr_cloud, data->curr_cloud, 3 * imgHeight * imgWidth * sizeof(float));
        memcpy(final_cloud, data->final_cloud, 3 * imgHeight * imgWidth * sizeof(float));

        memcpy(init_masks, data->init_masks, nSE3 * imgHeight * imgWidth * sizeof(float));
        memcpy(curr_masks, data->curr_masks, nSE3 * imgHeight * imgWidth * sizeof(float));
        memcpy(final_masks, data->final_masks, nSE3 * imgHeight * imgWidth * sizeof(float));

        memcpy(init_poses, data->init_poses, nSE3 * 12 * sizeof(float));
        memcpy(curr_poses, data->curr_poses, nSE3 * 12 * sizeof(float));
        memcpy(final_poses, data->final_poses, nSE3 * 12 * sizeof(float));

        memcpy(init_jts, data->init_jts, 7 * sizeof(float));
        memcpy(curr_jts, data->curr_jts, 7 * sizeof(float));
        memcpy(final_jts, data->final_jts, 7 * sizeof(float));

        pose_errors = data->pose_errors;
        deg_errors  = data->deg_errors;
    }

    // Free members alone
    void freeMemberMemory()
    {
        free(render_jts);
        free(rendered_img);
        free(rendered_img_sub);
        free(rendered_vertmap);
        free(rendered_vertmap_sub);

        free(init_cloud);
        free(curr_cloud);
        free(final_cloud);

        free(init_jts);
        free(curr_jts);
        free(final_jts);

        free(initnorm_cloud);
        free(currnorm_cloud);
        free(finalnorm_cloud);

        free(init_masks);
        free(curr_masks);
        free(final_masks);
        free(curr_labels);

        free(init_poses);
        free(curr_poses);
        free(final_poses);
    }

    // Destructor
    ~PyRealData()
    {
        // Free member memory
        freeMemberMemory();
    }
};

//////////////// == Static global variables == /////////////////////

class RealCtrlViz
{
    public:
        RealCtrlViz(int imgHeight, int imgWidth, float imgScale, int nSE3,
                    float fx, float fy, float cx, float cy,
                    const std::string savedir);
        ~RealCtrlViz();

        // Compatibility to sim
        void render_arm(const float *config,
                        float *rendered_ptcloud,
                        float *rendered_labels);

        // Real data
        void update_real_curr(const float *curr_angles, const float *curr_ptcloud,
                              const float *curr_poses, const float *curr_masks,
                              const float curr_pose_error, const float *curr_deg_error);

        void update_real_init(const float *start_angles, const float *start_ptcloud,
                              const float *start_poses, const float *start_masks,
                              const float *goal_angles, const float *goal_ptcloud,
                              const float *goal_poses, const float *goal_masks,
                              const float start_pose_error, const float *start_deg_error);

        void reset();

        // Save frames
        void start_saving_frames(const std::string framesavedir);

        void stop_saving_frames();

    private:
        boost::shared_ptr<PyRealData> data;
        boost::shared_ptr<boost::thread> pangolin_gui_thread;
};

#endif // REALCTRL_VISUALIZER_HPP
