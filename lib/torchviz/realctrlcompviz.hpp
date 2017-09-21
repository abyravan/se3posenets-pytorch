class RealCtrlViz
{
    public:
        RealCtrlViz(int imgHeight, int imgWidth, float imgScale, int nSE3,
                    float fx, float fy, float cx, float cy,
                    const std::string savedir, const int use_simulator);
        ~RealCtrlViz();

        // Compatibility to sim
        void render_arm(const float *config,
                        float *rendered_ptcloud,
                        float *rendered_labels);
        void compute_gt_da(const float *input_jts,
                           const float *target_jts,
                           const int winsize,
                           const float thresh,
                           float *gtwarped_out,
                           float *gtda_ids);
        void initialize_problem(const float *start_jts, const float *goal_jts,
                                float *start_pts, float *da_goal_pts);

        // Real data
        void update_real_curr(const float *curr_angles, const float *curr_ptcloud,
                              const float *curr_poses, const float *curr_masks, const unsigned char *curr_rgb,
                              const float curr_pose_error, const float *curr_pose_errors_indiv,
                              const float *curr_deg_error,
                              const float *curr_angles_bp, const float *curr_ptcloud_bp,
                              const float *curr_poses_bp, const float *curr_masks_bp, const unsigned char *curr_rgb_bp,
                              const float curr_pose_error_bp, const float *curr_pose_errors_indiv_bp,
                              const float *curr_deg_error_bp,
                              int save_frame);

        void update_real_init(const float *start_angles, const float *start_ptcloud,
                              const float *start_poses, const float *start_masks, const unsigned char *start_rgb,
                              const float *goal_angles, const float *goal_ptcloud,
                              const float *goal_poses, const float *goal_masks, const unsigned char *goal_rgb,
                              const float start_pose_error, const float *start_pose_errors_indiv,
                              const float *start_deg_error);

        void reset();

        // Save frames
        void start_saving_frames(const std::string framesavedir);

        void stop_saving_frames();
};
