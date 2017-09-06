class PangolinPoseViz
{
    public:
        PangolinPoseViz();
        ~PangolinPoseViz();

        void update_viz(const float *gtpose, const float *predpose,
                        float *config, float *ptcloud);
};
