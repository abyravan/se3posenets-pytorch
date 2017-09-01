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
                        float *rendered_ptcloud,
                        float *rendered_labels);

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
};

