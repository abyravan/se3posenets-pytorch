#ifndef DATA_VISUALIZER_HPP
#define DATA_VISUALIZER_HPP

// Boost thread/mutex
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>

//////////////////////
/// \brief The LuaData class - Class that encapsulates all data we want to pass from LUA to the pangolin visualizer
///
class LuaData
{
public:

    boost::mutex dataMutex;

    // Passed in data
    float *ptclouds;
    float *fwdflows;
    float *bwdflows;
    unsigned char *fwdvis;
    unsigned char *bwdvis;
    unsigned char *labels;
    unsigned char *rgbs;
    unsigned char *masks;
    float *poses;
    float *camposes;
    float *modelview;
    float *actconfigs, *actvels;
    float *comconfigs, *comvels;
    float *trackerconfigs;
    float *controls;
    float *initconfig;
    float *actdiffvels, *comdiffvels, *dartdiffvels;

    // Params
    int nSE3;
    int seq;
    int step;
    int se3Dim;
    int ht, wd;
    std::string data_path;
    float dt;
    int nstate;
    int nctrl;
    int ntracker;
    int id;

    // Cam params
    float fx, fy, cx, cy;

    // Setup num images & state/ctrl labels
    int nimages;
    std::vector<std::string> statelabels;
    std::vector<std::string> ctrllabels;
    std::vector<std::string> trackerlabels;

    // Init
    bool init_done;
    bool new_seq, done_seq;

    // Better constructor
    LuaData(std::string data, int nimages, int step_len, int seq_len, int nSE3, int ht, int wd,
            int nstate, int nctrl, int ntracker, float fx, float fy, float cx, float cy):
        data_path(data), step(step_len), seq(seq_len), nSE3(nSE3), ht(ht), wd(wd), nstate(nstate),
        nctrl(nctrl), ntracker(ntracker), fx(fx), cx(cx), fy(fy), cy(cy), nimages(nimages)
    {
        // Initialize memory for the pointers (done only once)
        ptclouds  = new float[(seq+1) * 3 * ht * wd];
        fwdflows  = new float[seq * 3 * ht * wd];
        bwdflows  = new float[seq * 3 * ht * wd];
        fwdvis    = new unsigned char[seq * 1 * ht * wd];
        bwdvis    = new unsigned char[seq * 1 * ht * wd];
        labels    = new unsigned char[(seq+1) * 1 * ht * wd];
        rgbs      = new unsigned char[(seq+1) * 3 * ht * wd];
        masks     = new unsigned char[(seq+1) * nSE3 * ht * wd];
        memset(ptclouds, 0, (seq+1) * 3 * ht * wd * sizeof(float));
        memset(fwdflows, 0, seq * 3 * ht * wd * sizeof(float));
        memset(bwdflows, 0, seq * 3 * ht * wd * sizeof(float));
        memset(fwdvis, 0, seq * 1 * ht * wd * sizeof(unsigned char));
        memset(bwdvis, 0, seq * 1 * ht * wd * sizeof(unsigned char));
        memset(labels, 0, (seq+1) * 1 * ht * wd * sizeof(unsigned char));
        memset(rgbs,   0, (seq+1) * 3 * ht * wd * sizeof(unsigned char));
        memset(masks,  0, (seq+1) * nSE3 * ht * wd * sizeof(unsigned char));

        // Poses
        se3Dim = 12;
        poses     = new float[(seq+1) * nSE3 * se3Dim];
        camposes  = new float[(seq+1) * nSE3 * se3Dim];
        modelview = new float[16];
        memset(poses, 0, (seq+1) * nSE3 * se3Dim * sizeof(float));
        memset(camposes, 0, (seq+1) * nSE3 * se3Dim * sizeof(float));
        memset(modelview, 0, 16 * sizeof(float));

        // Configs
        actconfigs = new float[(seq+1) * nstate];
        actvels    = new float[(seq+1) * nstate];
        comconfigs = new float[(seq+1) * nctrl];
        comvels    = new float[(seq+1) * nctrl];
        trackerconfigs = new float[(seq+1) * ntracker];
        controls   = new float[seq * nctrl];
        memset(actconfigs, 0, (seq+1) * nstate * sizeof(float));
        memset(actvels,    0, (seq+1) * nstate * sizeof(float));
        memset(comconfigs, 0, (seq+1) * nctrl * sizeof(float));
        memset(comvels,    0, (seq+1) * nctrl * sizeof(float));
        memset(trackerconfigs, 0, (seq+1) * ntracker * sizeof(float));
        memset(controls,   0, seq * nctrl * sizeof(float));

        actdiffvels    = new float[(seq) * nctrl];
        comdiffvels    = new float[(seq) * nctrl];
        dartdiffvels   = new float[(seq) * nctrl];
        memset(actdiffvels,    0, (seq) * nctrl * sizeof(float));
        memset(comdiffvels,    0, (seq) * nctrl * sizeof(float));
        memset(dartdiffvels,    0, (seq) * nctrl * sizeof(float));

        // Init config
        this->initconfig = new float[nstate];
        memset(this->initconfig,    0, nstate * sizeof(float));
        //memcpy(this->initconfig, initconfig, nstate * sizeof(float));

        // Other params
        dt = step * (1.0/30);
        id = 0;

        // Read num images & state/ctrl labels
        // TODO: Make this variable
        statelabels = std::vector<std::string>({"head_nod", "head_pan", "left_e0", "left_e1", "left_s0", "left_s1", "left_w0", "left_w1", "left_w2",
                       "right_e0", "right_e1", "right_s0", "right_s1", "right_w0", "right_w1", "right_w2",
                       "torso_t0", "r_gripper_l_finger_joint", "l_gripper_l_finger_joint"});
        ctrllabels  = std::vector<std::string>({"right_s0", "right_s1", "right_e0", "right_e1", "right_w0",
                                                "right_w1", "right_w2"});
        trackerlabels = std::vector<std::string>({"left_s0","left_s1","left_e0","left_e1","left_w0","left_w1","left_w2",
                                                  "l_gripper_r_finger_joint","l_gripper_l_finger_joint",
                                                  "right_s0","right_s1","right_e0","right_e1","right_w0","right_w1","right_w2",
                                                  "r_gripper_r_finger_joint","r_gripper_l_finger_joint" });

        // Booleans
        init_done = false;
        new_seq  = false;
        done_seq = true;
    }

//    // Destructor
//    ~LuaData()
//    {
//        // Free member memory
//        freeMemberMemory();
//    }
};

//////////////// == Static global variables == /////////////////////

class PangolinDataViz
{
    public:
        PangolinDataViz(std::string data_path, int nimages, int step_len, int seq_len, int nSE3, int ht, int wd,
                        int nstate, int nctrl, int ntracker, float fx, float fy, float cx, float cy);
        ~PangolinDataViz();

        void update_viz(const float *ptclouds,
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
                        const float *actdiffvels,
                        const float *comdiffvels,
                        const float *dartdiffvels,
                        const float *controls,
                        float *id);

    private:
        boost::shared_ptr<LuaData> data;
        boost::shared_ptr<boost::thread> pangolin_gui_thread;
};

#endif // DATA_VISUALIZER_HPP
