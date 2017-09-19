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
};
