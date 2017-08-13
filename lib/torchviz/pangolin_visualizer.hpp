#ifndef PANGOLIN_VISUALIZER_HPP
#define PANGOLIN_VISUALIZER_HPP

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
/// \brief The LuaData class - Class that encapsulates all data we want to pass from LUA to the pangolin visualizer
///
class LuaData
{
public:
    // Members
    float *inputPts;
    float *outputPts_gt;   // Sequence of GT output points
    float *outputPts_pred; // Sequence of predicted points
    float *jtAngles_gt;
    float *jtAngles_pred;
    int imgHeight, imgWidth, seqLength;
    float imgScale;
    float dt;
    float fx, fy, cx, cy; // Camera parameters
    boost::mutex dataMutex;

    // Render data
    cv::Mat depth_img_f, depth_img_f_sub;
    cv::Mat vertmap_f, vertmap_f_sub;
    float *rendered_img;
    float *rendered_img_sub;
    float *rendered_vertmap;
    float *rendered_vertmap_sub;
    float *render_jts;
    bool render;
    boost::mutex renderMutex;

    // GT rendering data
    float **gtda_res;
    float *gtwarped_out;
    float *gtda_ids;
    std::vector<dart::SE3> mesh_transforms;
    Eigen::Matrix4f modelView;

    // Data association points (only for one step)
    float *warpedcurrinput_cloud;
    float *current_da_ids;

    // Local pts
    float *local_1;
    float *local_2;
    float *local_matches;

    // Opencv mats for masks
    cv::Mat gt_mask, step_mask;

    // Init done
    bool oldgrippermodel;
    bool init_done;

    // Input, target & current pts for display
    float *init_cloud, *final_cloud, *finalobs_cloud, *currinput_cloud;
    float *currpred_cloud, *currpredgrad_cloud;
    float *currpreduw_cloud, *currpredgraduw_cloud;
    float *init_jts, *final_jts, *current_jts;
    float *currmask_img;
    float *init_poses, *tar_poses, *curr_poses;
    int nSE3;
    std::string savedir;

    // NOrmal clouds
    float *initnorm_cloud, *finalnorm_cloud, *finalobsnorm_cloud, *currinputnorm_cloud;
    float *currprednorm_cloud, *currpreduwnorm_cloud;

    // Empty Constructor
    LuaData();

    // Better constructor
    LuaData(const int &_seqLength, const int &_imgHeight, const int &_imgWidth, const float &_imgScale, const int _nSE3,
            const float &_fx, const float &_fy, const float &_cx, const float &_cy, const float &_dt,
            const bool _oldgrippermodel, const std::string _savedir):
            seqLength(_seqLength), imgHeight(_imgHeight), imgWidth(_imgWidth), imgScale(_imgScale),
            fx(_fx), fy(_fy), cx(_cx), cy(_cy), dt(_dt), oldgrippermodel(_oldgrippermodel), savedir(_savedir)
    {
        // Initialize memory for the pointers (done only once)
        inputPts        = new float[3 * imgHeight * imgWidth];
        outputPts_gt    = new float[seqLength * 3 * imgHeight * imgWidth];
        outputPts_pred  = new float[seqLength * 3 * imgHeight * imgWidth];
        jtAngles_gt     = new float[(seqLength+1) * 7]; memset(jtAngles_gt, 0, seqLength * 7);
        jtAngles_pred   = new float[(seqLength+1) * 7]; memset(jtAngles_gt, 0, seqLength * 7);

        // Render data
        render_jts   = new float[7];
        render = false;
        init_done = false;
        rendered_img = new float[640 * 480];
        rendered_img_sub = new float[imgWidth * imgHeight];
        rendered_vertmap = new float[640 * 480 * 4];
        rendered_vertmap_sub = new float[imgWidth * imgHeight * 4];

        // Depth img & Vertmap
        depth_img_f = cv::Mat(480, 640, CV_32FC1, cv::Scalar(0));
        vertmap_f   = cv::Mat(480, 640, CV_32FC4, cv::Scalar(0,0,0,0));
        depth_img_f_sub = cv::Mat(imgHeight, imgWidth, CV_32FC1, cv::Scalar(0));
        vertmap_f_sub   = cv::Mat(imgHeight, imgWidth, CV_32FC4, cv::Scalar(0,0,0,0));

        // GT render data
        gtwarped_out = new float[3 * imgHeight * imgWidth];
        gtda_ids     = new float[imgHeight * imgWidth];
        memset(gtwarped_out, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(gtda_ids, 0, imgHeight * imgWidth * sizeof(float));

        // Create float array
        gtda_res = (float **)(malloc(sizeof(float *) * 2)); // Returning 2 things

        // Input / Target / Current clouds
        init_cloud   = new float[3 * imgHeight * imgWidth];
        final_cloud  = new float[3 * imgHeight * imgWidth];
        finalobs_cloud  = new float[3 * imgHeight * imgWidth];
        currinput_cloud = new float[3 * imgHeight * imgWidth];
        memset(init_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(final_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(finalobs_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(currinput_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));

        // Initialize memory for point arrays
        warpedcurrinput_cloud = new float[3 * imgHeight * imgWidth];
        current_da_ids      = new float[imgHeight * imgWidth];
        memset(current_da_ids, 0, imgHeight * imgWidth * sizeof(float));
        memset(warpedcurrinput_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));

        // Predicted / Unwarped
        currpred_cloud       = new float[3 * imgHeight * imgWidth];
        currpredgrad_cloud   = new float[3 * imgHeight * imgWidth];
        currpreduw_cloud     = new float[3 * imgHeight * imgWidth];
        currpredgraduw_cloud = new float[3 * imgHeight * imgWidth];
        memset(currpred_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(currpredgrad_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(currpreduw_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(currpredgraduw_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));

        // Input / Target / Current clouds - Normals
        initnorm_cloud       = new float[3 * imgHeight * imgWidth];
        finalnorm_cloud      = new float[3 * imgHeight * imgWidth];
        finalobsnorm_cloud   = new float[3 * imgHeight * imgWidth];
        currinputnorm_cloud  = new float[3 * imgHeight * imgWidth];
        currprednorm_cloud   = new float[3 * imgHeight * imgWidth];
        currpreduwnorm_cloud = new float[3 * imgHeight * imgWidth];
        memset(initnorm_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(finalnorm_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(finalobsnorm_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(currinputnorm_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(currprednorm_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));
        memset(currpreduwnorm_cloud, 0, 3 * imgHeight * imgWidth * sizeof(float));

        // Init predicted masks & poses
        nSE3 = _nSE3;
        currmask_img = new float [imgHeight * imgWidth];
        init_poses   = new float [nSE3 * 12];
        tar_poses    = new float [nSE3 * 12];
        curr_poses   = new float [nSE3 * 12];
        memset(currmask_img, 0, imgHeight * imgWidth * sizeof(float));
        memset(init_poses, 0, nSE3 * 12 * sizeof(float));
        memset(tar_poses, 0, nSE3 * 12 * sizeof(float));
        memset(curr_poses, 0, nSE3 * 12 * sizeof(float));

        // Init memory
        local_1       = new float[4 * imgHeight * imgWidth];
        local_2       = new float[4 * imgHeight * imgWidth];
        local_matches = new float[4 * imgHeight * imgWidth];
        memset(local_1, 0, 4 * imgHeight * imgWidth * sizeof(float));
        memset(local_2, 0, 4 * imgHeight * imgWidth * sizeof(float));
        memset(local_matches, 0, 4 * imgHeight * imgWidth * sizeof(float));

        // Masks
        gt_mask = cv::Mat(imgHeight, imgWidth, CV_8UC1, cv::Scalar(255));
        step_mask = cv::Mat(imgHeight, imgWidth, CV_8UC1, cv::Scalar(255));

        // Input / Target / Current jts
        init_jts   = new float[7];
        current_jts = new float[7];
        final_jts  = new float[7];
        memset(init_jts, 0, 7 * sizeof(float));
        memset(current_jts, 0, 7 * sizeof(float));
        memset(final_jts, 0, 7 * sizeof(float));

        // Init, target poses (GT and predicted by net). Current pose prediction by net.
    }

    // Free members alone
    void freeMemberMemory()
    {
        free(inputPts);
        free(outputPts_gt);
        free(outputPts_pred);
        free(jtAngles_gt);
        free(jtAngles_pred);

        free(render_jts);
        free(rendered_img);
        free(rendered_img_sub);
        free(rendered_vertmap);
        free(rendered_vertmap_sub);

        free(gtwarped_out);
        free(gtda_ids);
        free(gtda_res);

        free(current_da_ids);
        free(warpedcurrinput_cloud);

        free(local_1);
        free(local_2);
        free(local_matches);

        free(init_cloud);
        free(currinput_cloud);
        free(final_cloud);
        free(finalobs_cloud);
        free(currpred_cloud);
        free(currpredgrad_cloud);
        free(currpreduw_cloud);
        free(currpredgraduw_cloud);
        free(init_jts);
        free(current_jts);
        free(final_jts);

        free(initnorm_cloud);
        free(currinputnorm_cloud);
        free(finalnorm_cloud);
        free(finalobsnorm_cloud);
        free(currprednorm_cloud);
        free(currpreduwnorm_cloud);

        free(currmask_img);
        free(init_poses);
        free(tar_poses);
        free(curr_poses);
    }

    // Destructor
    ~LuaData()
    {
        // Free member memory
        freeMemberMemory();
    }
};

//////////////// == Static global variables == /////////////////////

class PangolinViz
{
    public:
        PangolinViz(int seqLength, int imgHeight, int imgWidth, float imgScale, int nSE3,
                    float fx, float fy, float cx, float cy, float dt, int oldgrippermodel,
                    const std::string savedir);
        ~PangolinViz();

        void initialize_problem(const float *start_jts,
                                const float *goal_jts,
                                float *start_pts,
                                float *da_goal_pts);

        void update_viz(const float *inputpts,
                        const float *outputpts_gt,
                        const float *outputpts_pred,
                        const float *jtangles_gt,
                        const float *jtangles_pred);

        void render_arm(const float *config,
                        float *rendered_ptcloud);

        void update_da(const float *init_pts,
                       const float *current_pts,
                       const float *current_da_ids,
                       const float * warpedcurrent_pts,
                       const float *final_pts);

        void compute_gt_da(const float *input_jts,
                           const float *target_jts,
                           const int winsize,
                           const float thresh,
                           const float *final_pts,
                           float *gtwarped_out,
                           float *gtda_ids);

        void update_pred_pts(const float *net_preds,
                             const float *net_grads);

        void update_pred_pts_unwarped(const float *net_preds,
                                      const float *net_grads);

        void initialize_poses(const float *init_poses,
                              const float *tar_poses);

        void update_masklabels_and_poses(const float *curr_masks,
                                         const float *curr_poses);

        void start_saving_frames(const std::string framesavedir);

        void stop_saving_frames();

    private:
        boost::shared_ptr<LuaData> data;
        boost::shared_ptr<boost::thread> pangolin_gui_thread;
};


#endif // PANGOLIN_VISUALIZER_HPP
