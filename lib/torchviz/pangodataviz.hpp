class PangolinDataViz
{
    public:
        PangolinDataViz(std::string data_path, int step_len, int seq_len, int nSE3, int ht, int wd,
                        int nstate, int nctrl, int ntracker, float fx, float fy, float cx, float cy,
                        const float *initconfig);
        ~PangolinDataViz();

        void update_viz( const float *ptclouds,
                         const float *fwdflows,
                         const float *bwdflows,
                         const unsigned char *fwdvis,
                         const unsigned char *bwdvis,
                         const unsigned char *labels,
                         const unsigned char *rgbs,
                         const unsigned char *masks,
                         const float *poses,
                         const float *camposes,
                         const float *modelview,
                         const float *actconfigs,
                         const float *actvels,
                         const float *comconfigs,
                         const float *comvels,
                         const float *trackerconfigs,
                         const float *controls,
                         float *id);
};
