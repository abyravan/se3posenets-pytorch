#ifndef POSE_VISUALIZER_HPP
#define POSE_VISUALIZER_HPP

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
    float *pose;
    float *predpose;
    float *predmask;
    float *config;
    float *rendered_img, *ptcloud;
    int nSE3;
    int se3Dim;
    bool init_done;

    // Better constructor
    LuaData()
    {
        // Get params
        nSE3 = 8;
        se3Dim = 12;

        // Poses
        int npose = nSE3 * se3Dim;
        pose     = new float[npose];
        predpose = new float[npose];
        predmask = new float[nSE3*240*320];
        memset(pose, 0, npose * sizeof(float));
        memset(predpose, 0, npose * sizeof(float));
        memset(predmask, 0, nSE3*240*320 * sizeof(float));

        // Configs
        int nconfig = 7;
        config       = new float[nconfig];
        rendered_img = new float[1*480*640];
        ptcloud      = new float[3*240*320];
        memset(config, 0, nconfig * sizeof(float));
        memset(rendered_img, 0, 1*480*640*sizeof(float));
        memset(ptcloud, 0, 3*240*320*sizeof(float));

        // Booleans
        init_done = false;
    }

    // Free members alone
    void freeMemberMemory()
    {
        // Pose
        free(pose);
        free(predpose);
        free(config);
        free(ptcloud);
    }

    // Destructor
    ~LuaData()
    {
        // Free member memory
        freeMemberMemory();
    }
};

//////////////// == Static global variables == /////////////////////

class PangolinPoseViz
{
    public:
        PangolinPoseViz();
        ~PangolinPoseViz();

        void update_viz(const float *gtpose, const float *predpose, const float *predmask,
                        float *config, float *ptcloud);

    private:
        boost::shared_ptr<LuaData> data;
        boost::shared_ptr<boost::thread> pangolin_gui_thread;
};

#endif // POSE_VISUALIZER_HPP
