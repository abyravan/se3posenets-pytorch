class PangolinPoseViz
{
    public:
        PangolinPoseViz();
        ~PangolinPoseViz();

        void update_viz(const float *gtpose, const float *predpose, const float *predmask,
                        float *config, float *ptcloud);
};
