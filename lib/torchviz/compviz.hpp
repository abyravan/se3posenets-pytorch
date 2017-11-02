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
};
