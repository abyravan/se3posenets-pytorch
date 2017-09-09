// C++
#include <fstream>
#include <iostream>
#include <string>

// Boost thread/mutex
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Pangolin
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/utils/timer.h>

// Eigen
#include <Eigen/Dense>

// CUDA
#include <cuda_runtime.h>
#include <vector_types.h>

#include "geometry/plane_fitting.h"
#include "img_proc/img_ops.h"
#include "optimization/priors.h"
#include "tracker.h"
#include "util/dart_io.h"
#include "util/gl_dart.h"
#include "util/image_io.h"
#include "util/ostream_operators.h"
#include "util/string_format.h"
#include "visualization/color_ramps.h"
#include "visualization/data_association_viz.h"
#include "visualization/gradient_viz.h"
#include "optimization/kernels/intersection.h"

// OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

// New rendering stuff
#include "render/renderer.h"

// Pose visualizer
#include "data_visualizer.hpp"

using namespace std;

//////////////// == Static global variables == /////////////////////

/// -----------------------------------------
// Read/Write eigen matrices from/to disk
namespace Eigen{
template<class Matrix>
bool write_binary(const char* filename, const Matrix& matrix){
    std::ofstream out(filename,ios::out | ios::binary | ios::trunc);
    if (!out.is_open()) return false;
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
    return true;
}
template<class Matrix>
bool read_binary(const char* filename, Matrix& matrix){
    std::ifstream in(filename,ios::in | std::ios::binary);
    if (!in.is_open()) return false;
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
    return true;
}
} // Eigen::

//////////////// == Pangolin global vars == /////////////////////
static bool terminate_pangolin = false;
const static int panelWidth = 180;

void drawAxis(const float lineWidth)
{
    // draw axis
    glLineWidth(lineWidth);
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(1.0, 0, 0);
    glEnd();
    glColor3f(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0, 1, 0);
    glEnd();
    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0, 1);
    glEnd();
}

//////////////// == COMPUTE NORMALS == /////////////////
void compute_normals(const float *ptcloud, float *normcloud, int ncols, int nrows)
{
    // Compute normals
    int npts = ncols * nrows;
    for(int c = 1; c < ncols - 1; ++c)
    {
         for(int r = 1; r < nrows - 1; ++r)
         {
              // Get the points (top, left, center)
              int tid = ((r-1) * ncols + c);
            cv::Vec3f top 	 = cv::Vec3f(ptcloud[tid+0*npts], ptcloud[tid+1*npts], ptcloud[tid+2*npts]);
            int lid = (r * ncols + c-1);
            cv::Vec3f left 	 = cv::Vec3f(ptcloud[lid+0*npts], ptcloud[lid+1*npts], ptcloud[lid+2*npts]);
            int cid = (r * ncols + c);
            cv::Vec3f center = cv::Vec3f(ptcloud[cid+0*npts], ptcloud[cid+1*npts], ptcloud[cid+2*npts]);

            // Compute cross product and normalize
            cv::Vec3f d = (left-center).cross(top-center);
            cv::Vec3f n = cv::normalize(d);
              normcloud[cid + 0*npts] = n(0);
              normcloud[cid + 1*npts] = n(1);
              normcloud[cid + 2*npts] = n(2);
         }
    }
}

void drawFrame(const dart::SE3 se3, const float colorScale, const float frameLength, const float lineWidth)
{
    // Get the 3D points (x,y,z axis at frameLength distance from the center)
    float4 c = se3 * make_float4(0,0,0,1); // Center
    float4 x = se3 * make_float4(frameLength,0,0,1); // add translation to center (x-axis)
    float4 y = se3 * make_float4(0,frameLength,0,1); // add translation to center (y-axis)
    float4 z = se3 * make_float4(0,0,frameLength,1); // add translation to center (z-axis)

    // Line width
    glLineWidth(lineWidth);

    // Draw x-axis
    glColor3f(colorScale, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(c.x, c.y, c.z);
    glVertex3f(x.x, x.y, x.z);
    glEnd();

    // Draw y-axis
    glColor3f(0.0, colorScale, 0.0);
    glBegin(GL_LINES);
    glVertex3f(c.x, c.y, c.z);
    glVertex3f(y.x, y.y, y.z);
    glEnd();

    // Draw z-axis
    glColor3f(0.0, 0.0, colorScale);
    glBegin(GL_LINES);
    glVertex3f(c.x, c.y, c.z);
    glVertex3f(z.x, z.y, z.z);
    glEnd();
}

// Create SE3 from 12 values
dart::SE3 createSE3FromRt(float *rt)
{
    float4 r0 = make_float4(rt[0], rt[1], rt[2], rt[3]); // x,y,z,w (last is translation)
    float4 r1 = make_float4(rt[4], rt[5], rt[6], rt[7]); // x,y,z,w (last is translation)
    float4 r2 = make_float4(rt[8], rt[9], rt[10], rt[11]); // x,y,z,w (last is translation)
    return dart::SE3(r0, r1, r2);
}

// ------------------------------------------
///
/// \brief renderPangolinFrame - Render a pangolin frame
/// \param tracker  - Dart tracker instance
/// \param camState - OpenGL render state
///
void renderPangolinFrame(dart::Tracker &tracker, const pangolin::OpenGlRenderState &camState,
                         const pangolin::View &camDisp, const std::vector<char> &modelAlphas)
{
    //glShadeModel (GL_SMOOTH);
    float4 lightPosition = make_float4(normalize(make_float3(-0.4405,-0.5357,-0.619)),0);
    GLfloat light_ambient[] = { 0.3, 0.3, 0.3, 1.0 };
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_POSITION, (float*)&lightPosition);

    camDisp.ActivateScissorAndClear(camState);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHTING);

    camDisp.ActivateAndScissor(camState);

    glPushMatrix();
    //drawAxis(1);

    glColor4ub(0xff,0xff,0xff,0xff);
    glEnable(GL_COLOR_MATERIAL);
    for (int m=0; m<tracker.getNumModels(); ++m) {
        // Update pose
        tracker.updatePose(m);

        // Enable Alpha blending for rendering
        glEnable(GL_BLEND);
        tracker.getModel(m).render(modelAlphas[m]);
        tracker.getModel(m).renderSkeleton(modelAlphas[m]);
        glDisable(GL_BLEND);
    }

    /////////////////
    glPopMatrix();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHT0);
    glDisable(GL_NORMALIZE);
    glDisable(GL_LIGHTING);
    glColor4ub(255,255,255,255);

    /// Finish frame
    pangolin::FinishFrame();
}

//////////////// == PANGOLIN GUI THREAD == /////////////////////
/// \brief run_pangolin - Function that runs the pangolin GUI till terminate is called
/// \param data - Class instance containing all data copied from LUA
///
pangolin::Var<bool> *sendSeqToNet;
pangolin::Var<bool> *sliderControlled;
Eigen::Matrix4f modelView;
void run_pangolin(const boost::shared_ptr<LuaData> data)
{

    /// ===== Set up a DART tracker with the baxter model

    // Setup OpenGL/CUDA/Pangolin stuff - Has to happen before DART tracker initialization
    //cudaGLSetGLDevice(0);
    //cudaDeviceReset();
    const float totalwidth = 2*640 + panelWidth;
    const float totalheight = 480*3 + panelWidth;
    pangolin::CreateWindowAndBind("GD_Baxter: Results",totalwidth,totalheight);
    printf("Initialized Pangolin GUI  \n");

    /// ===== Pangolin initialization
    /// Pangolin mirrors the display, so we need to use TopLeft direndered_imgsplay for it. Our rendering needs BottomLeft

    // -=-=-=- pangolin window setup -=-=-=-
    pangolin::CreatePanel("gui").SetBounds(0.0,1.0,0,pangolin::Attach::Pix(panelWidth));
    pangolin::CreatePanel("data").SetBounds(0.9,1.0,pangolin::Attach::Pix(panelWidth),1.0);

    // Use default params as for rendering - what we used for the actual dataset rendering (rather than subsampled version)
    int glWidth = data->wd;
    int glHeight = data->ht;
    float glFLx = data->fx;// not sure what to do about these dimensions
    float glFLy = data->fy;// not sure what to do about these dimensions
    float glPPx = data->cx;
    float glPPy = data->cy;
    pangolin::OpenGlMatrixSpec glK_pangolin = pangolin::ProjectionMatrixRDF_TopLeft(glWidth,glHeight,glFLx,glFLy,glPPx,glPPy,0.01,1000);

    // Create a display renderer
    pangolin::OpenGlRenderState camState(glK_pangolin);
    pangolin::OpenGlRenderState camStatePose(glK_pangolin);
    pangolin::View & pcDisp = pangolin::Display("pc").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & flDisp   = pangolin::Display("fl").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & rDisp = pangolin::Display("rgb").SetAspect(glWidth*1.0f/(glHeight*1.0f));
    pangolin::View & allDisp = pangolin::Display("multi")
            .SetBounds(0.0, 0.35, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pcDisp)
            .AddDisplay(flDisp)
            .AddDisplay(rDisp);

    // Cam Disp is separate from others
    pangolin::View & camDisp = pangolin::Display("cam")
            .SetBounds(0.35, 0.9, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetAspect(glWidth*1.0f/(glHeight*1.0f))
            .SetHandler(new pangolin::Handler3D(camState));

    /// ===== Pangolin options

    // == GUI options
    pangolin::Var<float> frameLength("gui.frameLength",0.05,0.01,0.2);
    pangolin::Var<float> lineWidth("gui.lineWidth",3,1,10);
    pangolin::Var<bool> colorByLabel("gui.colorByLabel",false,true);
    //static pangolin::Var<int> ptsColorMap("gui.ptsColorMap",2,-1,11);

    // Pose display
    //pangolin::Var<bool> showPose("gui.showPose",true,true);
    //pangolin::Var<bool> showPredPose("gui.showPredPose",true,true);
    pangolin::Var<bool> showFDataCurrPtCloud("gui.showFlDataCurr",true,true);
    pangolin::Var<bool> showFDataNextPtCloud("gui.showFlDataNext",true,true);
    pangolin::Var<bool> showFFlowNextPtCloud("gui.showFlFlowNext",true,true);
    pangolin::Var<bool> showFTrackerCurrPtCloud("gui.showFlTrCurr",true,true);
    pangolin::Var<bool> showFTrackerNextPtCloud("gui.showFlTrNext",true,true);
    pangolin::Var<bool> showFlowAssoc("gui.showFlowAssoc",true,true);
    pangolin::Var<bool> showOnlyVisible("gui.showOnlyVisible",true,true);
    static pangolin::Var<int> arrowDensity("gui.arrowDensity",4,1,10);

    // Control display
    pangolin::Var<bool> showVelComparison("gui.showVelComparison",true,true);
    pangolin::Var<bool> showActVel("gui.showActVel",true,true);
    pangolin::Var<bool> showComVel("gui.showComVel",true,true);
    pangolin::Var<bool> showActDiffVel("gui.showActDiffVel",true,true);
    pangolin::Var<bool> showComDiffVel("gui.showComDiffVel",true,true);
    pangolin::Var<bool> showDartDiffVel("gui.showDartDiffVel",true,true);
    pangolin::Var<bool> useDartJtAngles("gui.useDartJtAngles",false,true);
    pangolin::Var<bool> showOnlyVels("gui.showOnlyVels",false,true);

    // == Dataset loading options
    sendSeqToNet = new pangolin::Var<bool>("data.sendSeqToNet",false,true);
    sliderControlled = new pangolin::Var<bool>("data.sliderControlled",false,true);
    pangolin::Var<int> dataID("data.id",0,0,data->nimages-1);
    pangolin::Var<int> idInSeq("data.idinseq",1,1,data->seq+1);

    /// ====== Setup model view matrix from disk
    // Model view matrix
    modelView = Eigen::Matrix4f::Identity();
    Eigen::read_binary("/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/cameramodelview.dat", modelView);

    // Set cam state
    camState.SetModelViewMatrix(modelView);
    camDisp.SetHandler(new pangolin::Handler3D(camState));

    // Create a renderer
    pangolin::OpenGlMatrixSpec glK = pangolin::ProjectionMatrixRDF_BottomLeft(data->wd,data->ht,data->fx,data->fy,data->cx,data->cy,0.01,1000);
    l2s::Renderer<TYPELIST2(l2s::IntToType<l2s::RenderVertMapWMeshID>, l2s::IntToType<l2s::RenderDepth>)> renderer(data->wd, data->ht, glK);
    renderer.setModelViewMatrix(modelView);

    /// ===== Setup DART

    // Setup tracker
    dart::Tracker tracker;

    // Baxter model path
    const std::string modelFile = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter_rosmesh.xml";

    // Load baxter model 1 (display init pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_1 = 0;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID_1).getReducedArticulatedDimensions() << endl;

    // Get the baxter pose and create a map between "frame name" and "pose dimension"
    dart::Pose &baxter_pose(tracker.getPose(baxterID_1));
    std::vector<std::string> model_joint_names;
    std::map<std::string, int> joint_name_to_pose_dim;
    for(int i = 0; i < baxter_pose.getReducedArticulatedDimensions(); i++)
    {
        model_joint_names.push_back(baxter_pose.getReducedName(i));
        joint_name_to_pose_dim[baxter_pose.getReducedName(i)] = i;
    }

    // Load baxter model 1 (display init pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_2 = 1;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID_1).getReducedArticulatedDimensions() << endl;

    // Load baxter model 1 (display init pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_3 = 2;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID_1).getReducedArticulatedDimensions() << endl;

    // Load baxter model 1 (display init pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_4 = 3;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID_1).getReducedArticulatedDimensions() << endl;

    // Load baxter model 1 (display init pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_5 = 4;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID_1).getReducedArticulatedDimensions() << endl;

    // Load baxter model 1 (display init pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_6 = 5;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID_1).getReducedArticulatedDimensions() << endl;

    // Load baxter model 1 (display init pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_7 = 6;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID_1).getReducedArticulatedDimensions() << endl;


    // Initialize to init config
    for(int k = 0; k < data->statelabels.size(); k++)
    {
        // Set baxter GT
        if (joint_name_to_pose_dim.find(data->statelabels[k]) != joint_name_to_pose_dim.end())
        {
            int pose_dim = joint_name_to_pose_dim[data->statelabels[k]];
            tracker.getPose(baxterID_1).getReducedArticulation()[pose_dim] = data->initconfig[k]; // q_1
            tracker.getPose(baxterID_2).getReducedArticulation()[pose_dim] = data->initconfig[k]; // q_1
        }
    }

    // Assuming that the joints are fixed - change later
    std::vector<std::string> valid_joint_names = {"right_s0", "right_s1", "right_e0", "right_e1", "right_w0",
                                                  "right_w1", "right_w2"};
    std::vector<char> modelAlphas = {(char)128, (char)64, (char)64, (char)64, (char)64, (char)64, (char)64}; // GT model alpha = 0.5, pred = 1, render = 0

    /// == Pre-process to compute the mesh vertices and indices for all the robot parts
    std::vector<std::vector<float3> > meshVertices, transformedMeshVertices;
    std::vector<std::vector<float4> > meshVerticesWMeshID;
    std::vector<pangolin::GlBuffer> meshIndexBuffers;
    std::vector<std::vector<pangolin::GlBuffer *> > meshVertexAttributeBuffers;
    std::vector<int> meshFrameids, meshModelids;

    // Get the model
    int m = baxterID_1;
    for (int s = 0; s < tracker.getModel(m).getNumSdfs(); ++s)
    {
        // Get the frame number for the SDF and it's transform w.r.t robot base
        int f = tracker.getModel(m).getSdfFrameNumber(s);
        const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);

        // Iterate over all the geometries for the model and get the mesh attributes for the data
        for(int g = 0; g < tracker.getModel(m).getFrameNumGeoms(f); ++g)
        {
            // Get the mesh index
            int gid = tracker.getModel(m).getFrameGeoms(f)[g];
            int mid = tracker.getModel(m).getMeshNumber(gid);
            if(mid == -1) continue; // Has no mesh

            // Get scales & transforms
            const float3 geomScale = tracker.getModel(m).getGeometryScale(gid);
            const dart::SE3 geomTransform = tracker.getModel(m).getGeometryTransform(gid);

            // Get the mesh
            const dart::Mesh mesh = tracker.getModel(m).getMesh(mid);
            meshFrameids.push_back(f); // Index of the frame for that particular mesh
            meshModelids.push_back(m); // ID of the model for that particular mesh

            // Get their vertices and transform them using the given frame to model transform
            meshVertices.push_back(std::vector<float3>(mesh.nVertices));
            transformedMeshVertices.push_back(std::vector<float3>(mesh.nVertices));
            meshVerticesWMeshID.push_back(std::vector<float4>(mesh.nVertices));
            for(int i = 0; i < mesh.nVertices; ++i)
            {
                // The mesh might have a rotation / translation / scale to get it to the current frame's reference.
                float3 rotV   = geomTransform * mesh.vertices[i]; // Rotate / translate to get to frame reference
                float3 vertex = make_float3(geomScale.x * rotV.x,
                                            geomScale.y * rotV.y,
                                            geomScale.z * rotV.z); // Scale the vertex

                // Get mesh vertex and transform it
                meshVertices.back()[i] = vertex;
                transformedMeshVertices.back()[i] = tfm * vertex;

                // Update the canonical vertex with the mesh ID
                meshVerticesWMeshID.back()[i] = make_float4(vertex.x, vertex.y,
                                                            vertex.z, meshVertices.size()); // Add +1 to mesh IDs (BG is zero)
            }

            // For each mesh, initialize memory for the transformed vertex buffers & the (canonical/fixed )mesh vertices with mesh ids
            std::vector<pangolin::GlBuffer *> attributes;
            attributes.push_back(new pangolin::GlBuffer(pangolin::GlBufferType::GlArrayBuffer, mesh.nVertices, GL_FLOAT, 3)); // deformed mesh vertices
            attributes.push_back(new pangolin::GlBuffer(pangolin::GlBufferType::GlArrayBuffer, mesh.nVertices, GL_FLOAT, 4)); // canonical mesh vertices with mesh id
            attributes[1]->Upload(meshVerticesWMeshID.back().data(), mesh.nVertices*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
            meshVertexAttributeBuffers.push_back(attributes);

            // For each mesh, save the faces - one time thing only
            meshIndexBuffers.resize(meshIndexBuffers.size()+1);
            meshIndexBuffers.back().Reinitialise(pangolin::GlBufferType::GlElementArrayBuffer, mesh.nFaces*3, GL_INT, 3, GL_DYNAMIC_DRAW);
            meshIndexBuffers.back().Upload(mesh.faces, mesh.nFaces*sizeof(int3));
        }
    }

    /// === KEEP a local copy of the data
    // Enable blending once
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    data->init_done = true;

    // Init stuff
    float rendered_ptcloud[3*data->ht*data->wd], rendered_ptcloud_com[3*data->ht*data->wd];
    float rendered_img[data->ht*data->wd], rendered_img_com[data->ht*data->wd];
    float rendered_normals[3*data->ht*data->wd], rendered_normals_com[3*data->ht*data->wd];
    float data_normals[3*data->ht*data->wd], data_normals_fl[3*data->ht*data->wd];

    // Flow stuff
    float rendered_ptcloud_fl[3*data->ht*data->wd], rendered_ptcloud_tr[3*data->ht*data->wd];
    float rendered_img_fl[data->ht*data->wd], rendered_img_tr[data->ht*data->wd];
    float rendered_normals_fl[3*data->ht*data->wd], rendered_normals_tr[3*data->ht*data->wd];

    // RGB texture
    pangolin::GlTexture texRGB(glWidth,glHeight,GL_RGB8);

    // Run till asked to terminate
    while (!pangolin::ShouldQuit() && !terminate_pangolin)
    {
        // General stuff
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        if (pangolin::HasResized())
        {
            pangolin::DisplayBase().ActivateScissorAndClear();
        }

        /// === Update all the rendering based on the options
        boost::mutex::scoped_lock data_lock(data->dataMutex);

        // Update seq start & length based on slider if it is slider controlled, else do the opposite
        if(*sliderControlled)
            data->id  = dataID;
        else
            dataID  = data->id;

        /// === Send data to network
        if (*sendSeqToNet && data->done_seq)
        {
            data->id  = dataID;

            /// === Update flags
            data->new_seq  = true;
            data->done_seq = false;
            *sliderControlled = false;

            // Set cam state
            camState.SetModelViewMatrix(modelView);
            camDisp.SetHandler(new pangolin::Handler3D(camState));
            pcDisp.SetHandler(new pangolin::Handler3D(camState));
        }

        /// === Use data
        int b = idInSeq-1;

        /// ****************** Render the point cloud based on actual joint angles

        // Set the actual config
        float *actconfig = &data->actconfigs[b * data->nstate];
        for(int k = 0; k < data->statelabels.size(); k++)
        {
            // Set baxter GT
            if (joint_name_to_pose_dim.find(data->statelabels[k]) != joint_name_to_pose_dim.end())
            {
                int pose_dim = joint_name_to_pose_dim[data->statelabels[k]];
                tracker.getPose(baxterID_1).getReducedArticulation()[pose_dim] = actconfig[k]; // q_1
            }
        }

        // Update the pose so that the FK is computed properly
        tracker.updatePose(baxterID_1);

        /// == Render a depth image / pt cloud

        // == Update mesh vertices based on new pose
        for (int k = 0; k < meshVertices.size(); k++)
        {
            // Get the SE3 transform for the frame
            int m = meshModelids[k];
            int f = meshFrameids[k];
            const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);

            // Transform the canonical vertices based on the new transform
            for(int j = 0; j < meshVertices[k].size(); ++j)
            {
                transformedMeshVertices[k][j] = tfm * meshVertices[k][j];
            }

            // Upload to pangolin vertex buffer
            meshVertexAttributeBuffers[k][0]->Upload(transformedMeshVertices[k].data(), transformedMeshVertices[k].size()*sizeof(float3));
            meshVertexAttributeBuffers[k][1]->Upload(meshVerticesWMeshID[k].data(), meshVerticesWMeshID[k].size()*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
        }

        // == Render a depth image
        glEnable(GL_DEPTH_TEST);
        renderer.setModelViewMatrix(modelView);
        renderer.renderMeshes<l2s::RenderDepth>(meshVertexAttributeBuffers, meshIndexBuffers);

        // Get the depth data as a float
        renderer.texture<l2s::RenderDepth>().Download(rendered_img);
        glDisable(GL_DEPTH_TEST);

        // == Compute pt cloud
        int npts = data->ht * data->wd;
        for(int r = 0; r < data->ht; r++)
        {
            for(int c = 0; c < data->wd; c++)
            {
                // Get X & Y value
                int id  = r * data->wd + c;
                float x = (c - glPPx) / glFLx;
                float y = (r - glPPy) / glFLy;

                // Compute input point value & set in PCL
                float zi = (float) rendered_img[id];
                rendered_ptcloud[id + 0*npts] = x*zi;
                rendered_ptcloud[id + 1*npts] = y*zi;
                rendered_ptcloud[id + 2*npts] = zi;
            }
        }

        /// ****************** Render the point cloud based on commanded joint angles

        // Set the commanded config using ctrl labels
        float *comconfig = &data->comconfigs[b * data->nctrl];
        for(int k = 0; k < data->ctrllabels.size(); k++)
        {
            // Set baxter GT
            if (joint_name_to_pose_dim.find(data->ctrllabels[k]) != joint_name_to_pose_dim.end())
            {
                int pose_dim = joint_name_to_pose_dim[data->ctrllabels[k]];
                tracker.getPose(baxterID_2).getReducedArticulation()[pose_dim] = comconfig[k]; // q_1
            }
        }

        // Update the pose so that the FK is computed properly
        tracker.updatePose(baxterID_2);

        /// == Render a depth image / pt cloud

        // == Update mesh vertices based on new pose
        for (int k = 0; k < meshVertices.size(); k++)
        {
            // Get the SE3 transform for the frame
            int m = baxterID_2; // Use model 2
            int f = meshFrameids[k];
            const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);

            // Transform the canonical vertices based on the new transform
            for(int j = 0; j < meshVertices[k].size(); ++j)
            {
                transformedMeshVertices[k][j] = tfm * meshVertices[k][j];
            }

            // Upload to pangolin vertex buffer
            meshVertexAttributeBuffers[k][0]->Upload(transformedMeshVertices[k].data(), transformedMeshVertices[k].size()*sizeof(float3));
            meshVertexAttributeBuffers[k][1]->Upload(meshVerticesWMeshID[k].data(), meshVerticesWMeshID[k].size()*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
        }

        // == Render a depth image
        glEnable(GL_DEPTH_TEST);
        renderer.setModelViewMatrix(modelView);
        renderer.renderMeshes<l2s::RenderDepth>(meshVertexAttributeBuffers, meshIndexBuffers);

        // Get the depth data as a float
        renderer.texture<l2s::RenderDepth>().Download(rendered_img_com);
        glDisable(GL_DEPTH_TEST);

        // == Compute pt cloud
        for(int r = 0; r < data->ht; r++)
        {
            for(int c = 0; c < data->wd; c++)
            {
                // Get X & Y value
                int id  = r * data->wd + c;
                float x = (c - glPPx) / glFLx;
                float y = (r - glPPy) / glFLy;

                // Compute input point value & set in PCL
                float zi = (float) rendered_img_com[id];
                rendered_ptcloud_com[id + 0*npts] = x*zi;
                rendered_ptcloud_com[id + 1*npts] = y*zi;
                rendered_ptcloud_com[id + 2*npts] = zi;
            }
        }

        /// ****************** Render the point cloud at current step

        // Set the commanded config using ctrl labels
        float *trackerconfig = &data->trackerconfigs[0 * data->ntracker];
        for(int k = 0; k < data->trackerlabels.size(); k++)
        {
            // Set baxter GT
            if (joint_name_to_pose_dim.find(data->trackerlabels[k]) != joint_name_to_pose_dim.end())
            {
                int pose_dim = joint_name_to_pose_dim[data->trackerlabels[k]];
                tracker.getPose(baxterID_1).getReducedArticulation()[pose_dim] = trackerconfig[k]; // q_1
            }
        }

        // Update the pose so that the FK is computed properly
        tracker.updatePose(baxterID_1);

        /// == Render a depth image / pt cloud
        // == Update mesh vertices based on new pose
        for (int k = 0; k < meshVertices.size(); k++)
        {
            // Get the SE3 transform for the frame
            int m = baxterID_1; // Use model 2
            int f = meshFrameids[k];
            const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);

            // Transform the canonical vertices based on the new transform
            for(int j = 0; j < meshVertices[k].size(); ++j)
            {
                transformedMeshVertices[k][j] = tfm * meshVertices[k][j];
            }

            // Upload to pangolin vertex buffer
            meshVertexAttributeBuffers[k][0]->Upload(transformedMeshVertices[k].data(), transformedMeshVertices[k].size()*sizeof(float3));
            meshVertexAttributeBuffers[k][1]->Upload(meshVerticesWMeshID[k].data(), meshVerticesWMeshID[k].size()*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
        }

        // == Render a depth image
        glEnable(GL_DEPTH_TEST);
        renderer.setModelViewMatrix(modelView);
        renderer.renderMeshes<l2s::RenderDepth>(meshVertexAttributeBuffers, meshIndexBuffers);

        // Get the depth data as a float
        renderer.texture<l2s::RenderDepth>().Download(rendered_img_tr);
        glDisable(GL_DEPTH_TEST);

        // == Compute pt cloud
        for(int r = 0; r < data->ht; r++)
        {
            for(int c = 0; c < data->wd; c++)
            {
                // Get X & Y value
                int id  = r * data->wd + c;
                float x = (c - glPPx) / glFLx;
                float y = (r - glPPy) / glFLy;

                // Compute input point value & set in PCL
                float zi = (float) rendered_img_tr[id];
                rendered_ptcloud_tr[id + 0*npts] = x*zi;
                rendered_ptcloud_tr[id + 1*npts] = y*zi;
                rendered_ptcloud_tr[id + 2*npts] = zi;
            }
        }

        /// ****************** Render the point cloud at flow step away

        // Set the commanded config using ctrl labels
        int next = min(b + data->step, data->seq);
        float *flowconfig = &data->trackerconfigs[b * data->ntracker];
        float *nextflowconfig = &data->trackerconfigs[next * data->ntracker];
        for(int k = 0; k < data->trackerlabels.size(); k++)
        {
            // Set baxter GT
            if (joint_name_to_pose_dim.find(data->trackerlabels[k]) != joint_name_to_pose_dim.end())
            {
                int pose_dim = joint_name_to_pose_dim[data->trackerlabels[k]];
                tracker.getPose(baxterID_2).getReducedArticulation()[pose_dim] = flowconfig[k]; // q_1
                if (useDartJtAngles)
                    tracker.getPose(baxterID_7).getReducedArticulation()[pose_dim] = nextflowconfig[k]; // q_1
            }
        }

        // Update the pose so that the FK is computed properly
        tracker.updatePose(baxterID_2);

        /// == Render a depth image / pt cloud

        // == Update mesh vertices based on new pose
        for (int k = 0; k < meshVertices.size(); k++)
        {
            // Get the SE3 transform for the frame
            int m = baxterID_2; // Use model 2
            int f = meshFrameids[k];
            const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);

            // Transform the canonical vertices based on the new transform
            for(int j = 0; j < meshVertices[k].size(); ++j)
            {
                transformedMeshVertices[k][j] = tfm * meshVertices[k][j];
            }

            // Upload to pangolin vertex buffer
            meshVertexAttributeBuffers[k][0]->Upload(transformedMeshVertices[k].data(), transformedMeshVertices[k].size()*sizeof(float3));
            meshVertexAttributeBuffers[k][1]->Upload(meshVerticesWMeshID[k].data(), meshVerticesWMeshID[k].size()*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
        }

        // == Render a depth image
        glEnable(GL_DEPTH_TEST);
        renderer.setModelViewMatrix(modelView);
        renderer.renderMeshes<l2s::RenderDepth>(meshVertexAttributeBuffers, meshIndexBuffers);

        // Get the depth data as a float
        renderer.texture<l2s::RenderDepth>().Download(rendered_img_fl);
        glDisable(GL_DEPTH_TEST);

        // == Compute pt cloud
        for(int r = 0; r < data->ht; r++)
        {
            for(int c = 0; c < data->wd; c++)
            {
                // Get X & Y value
                int id  = r * data->wd + c;
                float x = (c - glPPx) / glFLx;
                float y = (r - glPPy) / glFLy;

                // Compute input point value & set in PCL
                float zi = (float) rendered_img_fl[id];
                rendered_ptcloud_fl[id + 0*npts] = x*zi;
                rendered_ptcloud_fl[id + 1*npts] = y*zi;
                rendered_ptcloud_fl[id + 2*npts] = zi;
            }
        }

        /// Show current from ROS jt angles & DART

        // Set the actual config
        float *nextconfig = &data->actconfigs[next * data->nstate];
        for(int k = 0; k < data->statelabels.size(); k++)
        {
            // Set baxter GT
            if (joint_name_to_pose_dim.find(data->statelabels[k]) != joint_name_to_pose_dim.end())
            {
                int pose_dim = joint_name_to_pose_dim[data->statelabels[k]];
                tracker.getPose(baxterID_1).getReducedArticulation()[pose_dim] = actconfig[k]; // q_1
                tracker.getPose(baxterID_3).getReducedArticulation()[pose_dim] = actconfig[k]; // q_1
                tracker.getPose(baxterID_4).getReducedArticulation()[pose_dim] = actconfig[k]; // q_1
                tracker.getPose(baxterID_5).getReducedArticulation()[pose_dim] = actconfig[k]; // q_1
                tracker.getPose(baxterID_6).getReducedArticulation()[pose_dim] = actconfig[k]; // q_1
                if (!useDartJtAngles)
                    tracker.getPose(baxterID_7).getReducedArticulation()[pose_dim] = nextconfig[k]; // q_2
            }
        }

        /// ****************** Vel comparison
        if (showVelComparison)
        {
            float baseconfig[data->nctrl];
            float nextconfig[data->nctrl];
            for(int k = 0; k < data->ctrllabels.size(); k++)
            {
                if (joint_name_to_pose_dim.find(data->ctrllabels[k]) != joint_name_to_pose_dim.end())
                {
                    int pose_dim = joint_name_to_pose_dim[data->ctrllabels[k]];
                    if (useDartJtAngles)
                    {
                        baseconfig[k] = tracker.getPose(baxterID_2).getReducedArticulation()[pose_dim];
                    }
                    else
                    {
                        baseconfig[k] = tracker.getPose(baxterID_1).getReducedArticulation()[pose_dim];
                    }
                    nextconfig[k] = tracker.getPose(baxterID_7).getReducedArticulation()[pose_dim];
                }
            }

            float actvel[data->nctrl], comvel[data->nctrl], actdiffvel[data->nctrl], comdiffvel[data->nctrl], dartdiffvel[data->nctrl];
            float diffmax = -HUGE_VALF, diffmin = HUGE_VALF;
            for(int k = 0; k < data->nctrl; k++)
            {
                actvel[k] = showActVel ? (&data->actvels[b * data->nctrl])[k] : 0;
                comvel[k] = showComVel ? (&data->comvels[b * data->nctrl])[k] : 0;
                actdiffvel[k] = showActDiffVel ? (&data->actdiffvels[b * data->nctrl])[k] : 0;
                comdiffvel[k] = showComDiffVel ? (&data->comdiffvels[b * data->nctrl])[k] : 0;
                dartdiffvel[k] = showDartDiffVel ? (&data->dartdiffvels[b * data->nctrl])[k] : 0;
                float diff = abs(dartdiffvel[k] - actdiffvel[k]);
                if (diff > diffmax) diffmax = diff;
                if (diff < diffmin) diffmin = diff;
            }
            cout << diffmax << " " << diffmin << endl;

            // Set the actual config
            for(int k = 0; k < data->ctrllabels.size(); k++)
            {
                // Set baxter GT
                if (joint_name_to_pose_dim.find(data->ctrllabels[k]) != joint_name_to_pose_dim.end())
                {
                    int pose_dim = joint_name_to_pose_dim[data->ctrllabels[k]];
                    tracker.getPose(baxterID_1).getReducedArticulation()[pose_dim] = baseconfig[k]; // q_1
                    tracker.getPose(baxterID_2).getReducedArticulation()[pose_dim] = baseconfig[k] + data->dt * actvel[k]; // q_1
                    tracker.getPose(baxterID_3).getReducedArticulation()[pose_dim] = baseconfig[k] + data->dt * comvel[k]; // q_1
                    tracker.getPose(baxterID_4).getReducedArticulation()[pose_dim] = baseconfig[k] + data->dt * actdiffvel[k]; // q_1
                    tracker.getPose(baxterID_5).getReducedArticulation()[pose_dim] = baseconfig[k] + data->dt * comdiffvel[k]; // q_1
                    tracker.getPose(baxterID_6).getReducedArticulation()[pose_dim] = baseconfig[k] + data->dt * dartdiffvel[k]; // q_1
                    tracker.getPose(baxterID_7).getReducedArticulation()[pose_dim] = nextconfig[k]; // q_2
                }
            }
        }

        if (showOnlyVels)
        {
            modelAlphas[0] = 0;
            if (!showActVel) modelAlphas[1] = 0;
            if (!showComVel) modelAlphas[2] = 0;
            if (!showActDiffVel) modelAlphas[3] = 0;
            if (!showComDiffVel) modelAlphas[4] = 0;
            if (!showDartDiffVel) modelAlphas[5] = 0;
            modelAlphas[6] = 0;
        }
        else
        {
            modelAlphas = std::vector<char>({(char)128, (char)64, (char)64, (char)64, (char)64, (char)64, (char)0});
        }

        /// ****************** Render RGB image

        // Display RGB image
        glColor3f(1,1,1);
        rDisp.ActivateScissorAndClear();
        unsigned char *rgbimg = &data->rgbs[b * data->wd * data->ht * 3];
        texRGB.Upload(rgbimg,GL_RGB,GL_UNSIGNED_BYTE);
        texRGB.RenderToViewportFlipY();

        /// ================ Render the poses

        // Update modelview

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        pcDisp.ActivateScissorAndClear(camState);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor4ub(0xff,0xff,0xff,0xff);

        // Apply inverse of modelview to get pts in model frame
        Eigen::Matrix4f modelViewInv = modelView.inverse();
        glMultMatrixf(modelViewInv.data());

        // Draw all the poses
        float *pose = &data->poses[b * data->nSE3 * data->se3Dim];
        for(int i = 0; i < data->nSE3; i++)
        {
            // Create the different SE3s
            dart::SE3 p1 = createSE3FromRt(&pose[i*data->se3Dim]); // Gt pose 1
            drawFrame(p1, 1.0, frameLength, lineWidth);
        }

        // === Draw 3D points (both rendered & from data)

        // Compute normals
        compute_normals(&data->ptclouds[b*3*npts], data_normals, data->wd, data->ht);
        compute_normals(rendered_ptcloud, rendered_normals, data->wd, data->ht);
        compute_normals(rendered_ptcloud_com, rendered_normals_com, data->wd, data->ht);

        // Get label color
        unsigned char *label = &data->labels[b*npts];

        // Get max val
        std::vector<float3> colors = {make_float3(0,0,0),
                                      make_float3(1,1,1),
                                      make_float3(1,0,0),
                                      make_float3(0,1,0),
                                      make_float3(0,0,1),
                                      make_float3(1,1,0),
                                      make_float3(1,0,1),
                                      make_float3(0,1,1),
                                      make_float3(0.5,1,0.5),
                                      make_float3(0.5,0.5,1),
                                      make_float3(1,0.5,0.5),
                                      make_float3(0.5,0.25,0.75),
                                      make_float3(0.25,0.5,0.75),
                                      make_float3(0.75,0.25,0.5),
                                      make_float3(0.5,0.5,0.5),
                                      make_float3(0.5,0,0),
                                      make_float3(0,0.5,0),
                                      make_float3(0,0,0.5),
                                      make_float3(1,0,1),
                                      make_float3(1,1,1)
                                     };

        glPointSize(3.0);
        glBegin(GL_POINTS);
        float *data_ptcloud = &data->ptclouds[b*3*npts];
        for (int r = 0; r < data->ht; r++) // copy over from the flow matrix
        {
           for (int c = 0; c < data->wd; c++)
           {
               // Get pt index
               int id = r * data->wd + c;

               ///// == From data
               if (colorByLabel)
               {
                   float3 color = colors[label[id]];
                   glColor3f(color.x,color.y,color.z);
               }
               else
               {
                   glColor3f(0,1,0);
               }

               // Render pts and normals
               glNormal3f(data_normals[id + 0*npts],
                          data_normals[id + 1*npts],
                          data_normals[id + 2*npts]);
               glVertex3f(data_ptcloud[id + 0*npts],
                          data_ptcloud[id + 1*npts],
                          data_ptcloud[id + 2*npts]);

               ///// == Rendered
               // Fixed color
               glColor3f(0,0,1);

               // Render pts and normals
               glNormal3f(rendered_normals[id + 0*npts],
                          rendered_normals[id + 1*npts],
                          rendered_normals[id + 2*npts]);
               glVertex3f(rendered_ptcloud[id + 0*npts],
                          rendered_ptcloud[id + 1*npts],
                          rendered_ptcloud[id + 2*npts]);

               ///// == Rendered
               // Fixed color
               glColor3f(1,0,0);

               // Render pts and normals
               glNormal3f(rendered_normals_com[id + 0*npts],
                          rendered_normals_com[id + 1*npts],
                          rendered_normals_com[id + 2*npts]);
               glVertex3f(rendered_ptcloud_com[id + 0*npts],
                          rendered_ptcloud_com[id + 1*npts],
                          rendered_ptcloud_com[id + 2*npts]);
           }
        }
        glEnd();

        // Finish
        glPopMatrix(); // Remove inverse transform

        // Disable flags
        glColor4ub(255,255,255,255);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        glDisable(GL_LIGHTING);

        /// === Flow

        // Update modelview

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        flDisp.ActivateScissorAndClear(camState);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor4ub(0xff,0xff,0xff,0xff);

        // Apply inverse of modelview to get pts in model frame
        glMultMatrixf(modelViewInv.data());

        // Compute normals
        //compute_normals(rendered_ptcloud_tr, rendered_normals_tr, data->wd, data->ht);
        //compute_normals(rendered_ptcloud_fl, rendered_normals_fl, data->wd, data->ht);
        //compute_normals(&data->ptclouds[(b+f)*3*npts], data_normals_fl, data->wd, data->ht);

        // Draw
        glBegin(GL_LINES);
        int aden = arrowDensity;
        float *initdata = &data->ptclouds[0 * npts * 3];
        float *nextdata = &data->ptclouds[b * npts * 3];
        float *flow = &data->fwdflows[b * npts * 3];
        unsigned char *vis = &data->fwdvis[b * npts];
        for (int r = 0; r < data->ht; r+=aden) // copy over from the flow matrix
        {
           for (int c = 0; c < data->wd; c+=aden)
           {
               // Get pt index & init point
               int id = r * data->wd + c;
               float xi = initdata[id + 0*npts];
               float yi = initdata[id + 1*npts];
               float zi = initdata[id + 2*npts];

               if (showOnlyVisible && !vis[id]) continue;

               // Init -> Curr Input associations (Yellow - 110)
               if (showFlowAssoc)
               {
                   // Get points from the current input cloud (warped to match the input)
                   float xi_a = xi + flow[id + 0*npts];
                   float yi_a = yi + flow[id + 1*npts];
                   float zi_a = zi + flow[id + 2*npts];
                   glColor3ub(255,255,255); // Y
                   glVertex3f(xi, yi, zi);
                   glVertex3f(xi_a, yi_a, zi_a);
               }
           }
        }
        glEnd();

        glPointSize(3.0);
        glBegin(GL_POINTS);
        for (int r = 0; r < data->ht; r++) // copy over from the flow matrix
        {
           for (int c = 0; c < data->wd; c++)
           {
               // Get pt index
               int id = r * data->wd + c;

               if (showFDataCurrPtCloud)
               {
                   ///// == From data
                   glColor3f(0,1,0);

                   // Render pts and normals
                   /*
                   glNormal3f(data_normals[id + 0*npts],
                              data_normals[id + 1*npts],
                              data_normals[id + 2*npts]);*/
                   glVertex3f(initdata[id + 0*npts],
                              initdata[id + 1*npts],
                              initdata[id + 2*npts]);
               }

               if (showFTrackerCurrPtCloud)
               {
                   ///// == Rendered
                   // Fixed color
                   glColor3f(0,0,1);

                   // Render pts and normals
                   /*
                   glNormal3f(rendered_normals_tr[id + 0*npts],
                              rendered_normals_tr[id + 1*npts],
                              rendered_normals_tr[id + 2*npts]);*/
                   glVertex3f(rendered_ptcloud_tr[id + 0*npts],
                              rendered_ptcloud_tr[id + 1*npts],
                              rendered_ptcloud_tr[id + 2*npts]);
               }

               if (showFDataNextPtCloud)
               {
                   ///// == Rendered
                   // Fixed color
                   glColor3f(1,1,0);

                   // Render pts and normals
                   /*
                   glNormal3f(data_normals_fl[id + 0*npts],
                              data_normals_fl[id + 1*npts],
                              data_normals_fl[id + 2*npts]);*/
                   glVertex3f(nextdata[id + 0*npts],
                              nextdata[id + 1*npts],
                              nextdata[id + 2*npts]);
               }

               if (showFFlowNextPtCloud)
               {
                   ///// == Rendered
                   // Fixed color
                   glColor3f(1,0,1);

                   // Render pts and normals
                   /*
                   glNormal3f(data_normals_fl[id + 0*npts],
                              data_normals_fl[id + 1*npts],
                              data_normals_fl[id + 2*npts]);*/
                   glVertex3f(data_ptcloud[id + 0*npts] + flow[id + 0*npts],
                              data_ptcloud[id + 1*npts] + flow[id + 1*npts],
                              data_ptcloud[id + 2*npts] + flow[id + 2*npts]);
               }

               if (showFTrackerNextPtCloud)
               {
                   ///// == Rendered
                   // Fixed color
                   glColor3f(1,0,0);

                   // Render pts and normals
                   /*
                   glNormal3f(rendered_normals_fl[id + 0*npts],
                              rendered_normals_fl[id + 1*npts],
                              rendered_normals_fl[id + 2*npts]);*/
                   glVertex3f(rendered_ptcloud_fl[id + 0*npts],
                              rendered_ptcloud_fl[id + 1*npts],
                              rendered_ptcloud_fl[id + 2*npts]);
               }
           }
        }
        glEnd();

        // Finish
        glPopMatrix(); // Remove inverse transform

        // Disable flags
        glColor4ub(255,255,255,255);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        glDisable(GL_LIGHTING);

        // Unlock lock on data
        data_lock.unlock();

        /// == Render
        renderPangolinFrame(tracker, camState, camDisp, modelAlphas);
        usleep(100);
     }
}

//////////////// == FFI functions for calling from LUA == /////////////////////
/// \brief initialize_viz - Initialize pangolin based visualizer
///
PangolinDataViz::PangolinDataViz(std::string data_path, int nimages, int step_len, int seq_len, int nSE3, int ht, int wd,
                                 int nstate, int nctrl, int ntracker, float fx, float fy, float cx, float cy)
{
    printf("==> [PANGOLIN_VIZ] Initializing data for visualizer \n");
    data = boost::shared_ptr<LuaData>(new LuaData(data_path, nimages, step_len, seq_len, nSE3, ht, wd,
                                                  nstate, nctrl, ntracker, fx, fy, cx, cy));

    /// ===== PANGOLIN viewer
    printf("==> [PANGOLIN_VIZ] Starting pangolin in a separate thread \n");
    pangolin_gui_thread.reset(new boost::thread(run_pangolin, data));

    while(!data->init_done) { usleep(100); }
    printf("==> [PANGOLIN_VIZ] Finished initializing pangolin visualizer \n");
    return;
}

//////////////////////
///
/// \brief terminate_viz - Kill the visualizer
///
PangolinDataViz::~PangolinDataViz()
{
    // Terminate visualizer (data will be deleted automatically as it is a shared ptr)
    terminate_pangolin = true;
    pangolin_gui_thread->join(); // Wait for thread to join
    printf("==> [PANGOLIN_VIZ] Terminated pangolin visualizer \n");
}

////////////////////////////////////////////////////////////////////////////////////////
/// == VIZ STUFF FOR RNNs

///
/// \brief update_viz - Get new data from the lua code
///
void PangolinDataViz::update_viz(const float *ptclouds,
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
                                 float *id)
{
    /// === In case there was a new control that was not done, we render the net predictions
    /// === We then reset the state & advance the config
    if (data->new_seq)
    {
        // === Copy data to pangolin
        boost::mutex::scoped_lock data_mutex(data->dataMutex);

        memcpy(data->ptclouds, ptclouds, (data->seq+1) * 3 * data->ht * data->wd * sizeof(float));
        memcpy(data->fwdflows, fwdflows, data->seq * 3 * data->ht * data->wd * sizeof(float));
        memcpy(data->bwdflows, bwdflows, data->seq * 3 * data->ht * data->wd * sizeof(float));
        memcpy(data->fwdvis, fwdvis, data->seq * 1 * data->ht * data->wd * sizeof(unsigned char));
        memcpy(data->bwdvis, bwdvis, data->seq * 1 * data->ht * data->wd * sizeof(unsigned char));
        memcpy(data->labels, labels, (data->seq+1) * 1 * data->ht * data->wd * sizeof(unsigned char));
        memcpy(data->rgbs,   rgbs,   (data->seq+1) * 3 * data->ht * data->wd * sizeof(unsigned char));
        memcpy(data->masks,  masks,  (data->seq+1) * data->nSE3 * data->ht * data->wd * sizeof(unsigned char));

        memcpy(data->poses,     poses,    (data->seq+1) * data->nSE3 * data->se3Dim * sizeof(float));
        memcpy(data->camposes,  camposes, (data->seq+1) * data->nSE3 * data->se3Dim * sizeof(float));
        memcpy(data->modelview, modelview, 16 * sizeof(float));

        memcpy(data->actconfigs, actconfigs, (data->seq+1) * data->nstate * sizeof(float));
        memcpy(data->actvels,    actvels,    (data->seq+1) * data->nstate * sizeof(float));
        memcpy(data->comconfigs, comconfigs, (data->seq+1) * data->nctrl * sizeof(float));
        memcpy(data->comvels,    comvels,    (data->seq+1) * data->nctrl * sizeof(float));
        memcpy(data->trackerconfigs, trackerconfigs, (data->seq+1) * data->ntracker * sizeof(float));
        memcpy(data->controls,   controls,   data->seq * data->nctrl * sizeof(float));
        memcpy(data->actdiffvels,    actdiffvels,    (data->seq) * data->nctrl * sizeof(float));
        memcpy(data->comdiffvels,    comdiffvels,    (data->seq) * data->nctrl * sizeof(float));
        memcpy(data->dartdiffvels,    dartdiffvels,    (data->seq) * data->nctrl * sizeof(float));


        /// === Update modelview
        Eigen::Map<Eigen::Matrix4f> modelview_d(data->modelview);
        modelView = modelview_d.transpose();

        usleep(10000); // Sleep for a bit so that the rendering has a chance to update the poses to avoid race condition

        // === Update flags
        data->new_seq  = false;
        data->done_seq = true;
        *sendSeqToNet  = false; // We can wait for another control now
        *sliderControlled = true; // Enable slider control again

        // Unlock mutex
        data_mutex.unlock();
    }

    /// === Now wait till we are given a new control
    while(!data->new_seq)
    {
        usleep(1000);
    }

    // Copy over data to the existing vars
    // Update in the mutex
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Copy data ID
    id[0] = data->id;

    // Finished updating
    update_lock.unlock();
}
