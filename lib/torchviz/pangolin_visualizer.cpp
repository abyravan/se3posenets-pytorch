// C++
#include <fstream>
#include <iostream>
#include <string>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Pangolin
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/utils/timer.h>

// CUDA
#include <cuda_runtime.h>
#include <vector_types.h>

// New rendering stuff
#include "render/renderer.h"

// Viz class for cython
#include "pangolin_visualizer.hpp"

using namespace std;

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
const static int panelWidth = 180;
static bool terminate_pangolin = false;

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

///////// == Global parameters
// Find scaling of point within the color cube specified below
float3 cube_min = make_float3(-1.0,-1.0,0.0); // -1m in x and y, 0m in z
float3 cube_max = make_float3(1,1,4);   // +1m in x and y, +5m in z

// For height based coloring
float height_min = -1;
float height_max = 2;

///////// == Find co-ordinate of a point within a 3D cube
float3 get_color(const float3 &pt, const float3 &min, const float3 &max)
{
    float3 color;
    float xc = clamp(pt.x, min.x, max.x); color.x = (xc - min.x) / (max.x - min.x);
    float yc = clamp(pt.y, min.y, max.y); color.y = (yc - min.y) / (max.y - min.y);
    float zc = clamp(pt.z, min.z, max.z); color.z = (zc - min.z) / (max.z - min.z);
    return color;
}

///////// == Draw a 3D frame
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

///////// == Create SE3 from 12 values
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

    assert(modelAlphas.size() == tracker.getNumModels());
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

// Colors
// Init              - red (100)
// Net input         - blue (001)
// Net prediction    - (101)
// Net prediction uw - (0.5,0,0.5)
// Final cloud       - green (010)
// Input assoc       - (110)
// Pred assoc        - (011)
// Net grad          - (111)
// Net grad uw       - (0.5,0.5,0.5)

//////////////// == PANGOLIN GUI THREAD == /////////////////////
/// \brief run_pangolin - Function that runs the pangolin GUI till terminate is called
/// \param data - Class instance containing all data copied from LUA
///

int glWidth = 640;
int glHeight = 480;

// Saving files to disk
std::string save_frames_dir;
bool save_frames = false;
int framectr = 0;
bool updated_masks = false;

void run_pangolin(const boost::shared_ptr<LuaData> data)
{
    /// ===== Set up a DART tracker with the baxter model

    // Setup OpenGL/CUDA/Pangolin stuff - Has to happen before DART tracker initialization
    //cudaGLSetGLDevice(0);
    //cudaDeviceReset();
    const float totalwidth = glWidth*3 + panelWidth;
    const float totalheight = glHeight*2;
    pangolin::CreateWindowAndBind("GD_Baxter: Results",totalwidth,totalheight);
    printf("Initialized Pangolin GUI  \n");

    //glewInit();
    //glutInit(&argc, argv);

    /// ===== Pangolin initialization
    /// Pangolin mirrors the display, so we need to use TopLeft direndered_imgsplay for it. Our rendering needs BottomLeft

    // Use default params as for rendering - what we used for the actual dataset rendering (rather than subsampled version)
    float glFLx = 589.3664541825391;// not sure what to do about these dimensions
    float glFLy = 589.3664541825391;// not sure what to do about these dimensions
    float glPPx = 320.5;
    float glPPy = 240.5;
    pangolin::OpenGlMatrixSpec glK_pangolin = pangolin::ProjectionMatrixRDF_TopLeft(glWidth,glHeight,glFLx,glFLy,glPPx,glPPy,0.01,1000);

    /*
    // -=-=-=- pangolin window setup -=-=-=-
    pangolin::CreatePanel("gui").SetBounds(0.5,1,0,pangolin::Attach::Pix(panelWidth));
    pangolin::CreatePanel("se3gui").SetBounds(0.0,0.5,0,pangolin::Attach::Pix(panelWidth));

    // Create a display renderer
    pangolin::OpenGlRenderState camState(glK_pangolin);
    pangolin::OpenGlRenderState camStatePose(glK_pangolin);
    pangolin::View & camDisp = pangolin::Display("cam").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & pcDisp = pangolin::Display("pointcloud").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & maskDisp = pangolin::Display("mask").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & poseDisp = pangolin::Display("pose").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camStatePose));
    pangolin::View & allDisp = pangolin::Display("multi")
            .SetBounds(0.0, 1, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(camDisp)
            .AddDisplay(maskDisp)
            .AddDisplay(pcDisp)
            .AddDisplay(poseDisp);
    */

    // -=-=-=- pangolin window setup -=-=-=-
    pangolin::CreatePanel("gui").SetBounds(0.5,1,0,pangolin::Attach::Pix(panelWidth));
    pangolin::CreatePanel("se3gui").SetBounds(0.0,0.5,0,pangolin::Attach::Pix(panelWidth));

    // Create a display renderer
    pangolin::OpenGlRenderState camState(glK_pangolin);
    pangolin::OpenGlRenderState camStatePose(glK_pangolin);
    pangolin::View & pcDisp = pangolin::Display("pointcloud").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & maskDisp = pangolin::Display("mask").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & poseDisp = pangolin::Display("pose").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camStatePose));
    pangolin::View & allDisp = pangolin::Display("multi")
            .SetBounds(0.0, 0.5, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pcDisp)
            .AddDisplay(poseDisp)
            .AddDisplay(maskDisp);

    // Cam Disp is separate from others
    pangolin::View & camDisp = pangolin::Display("cam")
            .SetBounds(0.5, 1.0, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetAspect(glWidth*1.0f/(glHeight*1.0f))
            .SetHandler(new pangolin::Handler3D(camState));

    /// ===== Pangolin options

    // == GUI options
    pangolin::Var<bool> showInit("gui.showInit",false,true);
    pangolin::Var<bool> showNetInput("gui.showNetInput",true,true);
    pangolin::Var<bool> showNetPred("gui.showNetPred",false,true);
    pangolin::Var<bool> showNetPredUnWarped("gui.showNetPredUnWarped",false,true);
    pangolin::Var<bool> showFinal("gui.showFinal",false,true);
    pangolin::Var<bool> showFinalObs("gui.showFinalObs",true,true);
    pangolin::Var<bool> showNetGrad("gui.showNetGrad",false,true);
    pangolin::Var<bool> showNetGradUnWarped("gui.showNetGradUnWarped",false,true);
    pangolin::Var<bool> showInputAssoc("gui.showInputAssoc",false,true);
    pangolin::Var<bool> showPredAssoc("gui.showPredAssoc",false,true);

    // Grad options
    pangolin::Var<bool> normalizeGrad("gui.normalizeGrad",false,true);
    pangolin::Var<bool> showBGAssoc("gui.showBGAssoc",false,true);
    static pangolin::Var<float> scaleNormalizedGrad("gui.scaleNormGrad",0.05,0,1);
    static pangolin::Var<float> scaleUnNormalizedGrad("gui.scaleUnNormGrad",50,0,250);
    static pangolin::Var<int> arrowDensity("gui.arrowDensity",4,1,10);

    // Options for color map
    static pangolin::Var<int> maskColorMap("se3gui.maskColorMap",2,0,11);
    pangolin::Var<bool> useNormalsForMaskColor("se3gui.useMaskNormals",true,true);
    pangolin::Var<bool> showInitPoses("se3gui.showInitPoses",false,true);
    pangolin::Var<bool> showTarPoses("se3gui.showTarPoses",true,true);
    pangolin::Var<bool> showCurrPoses("se3gui.showCurrPoses",true,true);
    pangolin::Var<float> frameLength("se3gui.frameLength",0.05,0.01,0.2);
    pangolin::Var<float> lineWidth("se3gui.lineWidth",3,1,10);
    pangolin::Var<bool> showAllFrames("se3gui.showAllFrames",false,true);
    pangolin::Var<bool> **frameStatus = new pangolin::Var<bool>*[data->nSE3];
    for(int i = 0; i < data->nSE3; i++)
        frameStatus[i] = new pangolin::Var<bool>(dart::stringFormat("se3gui.showFrame%d",i),true,true);
    pangolin::Var<bool> printPoseModelView("se3gui.printPoseMV",false,true,false);

    // For current model, do not display BG frames 3,4,6,7
    //*frameStatus[3] = false; *frameStatus[4] = false;
    //*frameStatus[6] = false; *frameStatus[7] = false;

    /// ====== Setup model view matrix from disk
    // Model view matrix
    Eigen::Matrix4f modelView = Eigen::Matrix4f::Identity();
    Eigen::read_binary("/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/cameramodelview.dat", modelView);

    // Set cam state
    camState.SetModelViewMatrix(modelView);
    camDisp.SetHandler(new pangolin::Handler3D(camState));
    pcDisp.SetHandler(new pangolin::Handler3D(camState));
    maskDisp.SetHandler(new pangolin::Handler3D(camState));

    // Set cam state for pose
    Eigen::Matrix4f modelViewPose;
    modelViewPose << -0.0309999, 0.999487, -0.00799519,	0.18, //0.117511,
                      0.24277,	-0.000230283, -0.970084, 0.3, //0.193581,
                     -0.969588,	-0.0320134,	-0.242639,	2, //0.29869,
                      0,	0,	0,	1;
    camStatePose.SetModelViewMatrix(modelViewPose);
    poseDisp.SetHandler(new pangolin::Handler3D(camStatePose));

    // Create a renderer
    pangolin::OpenGlMatrixSpec glK = pangolin::ProjectionMatrixRDF_BottomLeft(glWidth,glHeight,glFLx,glFLy,glPPx,glPPy,0.01,1000);
    l2s::Renderer<TYPELIST2(l2s::IntToType<l2s::RenderVertMapWMeshID>, l2s::IntToType<l2s::RenderDepth>)> renderer(glWidth, glHeight, glK);
    renderer.setModelViewMatrix(modelView);

    // Save it
    data->modelView = modelView;
    Eigen::Matrix4f modelViewInv = modelView.inverse();
    Eigen::Matrix4f modelViewPoseInv = modelViewPose.inverse();

    /// ===== Setup DART

    // Setup tracker
    dart::Tracker tracker;

    // Baxter model path
    const std::string modelFile = (data->oldgrippermodel) ?
              "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter_rosmesh_old.xml" :
              "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter_rosmesh.xml";

    // Load baxter model 1 (display init pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_in = 0;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID_in).getReducedArticulatedDimensions() << endl;

    // Get the baxter pose and create a map between "frame name" and "pose dimension"
    dart::Pose &baxter_pose(tracker.getPose(baxterID_in));
    std::vector<std::string> model_joint_names;
    std::map<std::string, int> joint_name_to_pose_dim;
    for(int i = 0; i < baxter_pose.getReducedArticulatedDimensions(); i++)
    {
        model_joint_names.push_back(baxter_pose.getReducedName(i));
        joint_name_to_pose_dim[baxter_pose.getReducedName(i)] = i;
    }

    // Load baxter model 2 (display target pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_tar = 1;

    // Load baxter model 3 (display curr pose)
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_curr = 2;

    // Load baxter model 3 ==> FOR RENDERING
    tracker.addModel(modelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID_render = 3;

    // Assuming that the joints are fixed - change later
    std::vector<std::string> valid_joint_names = {"right_s0", "right_s1", "right_e0", "right_e1", "right_w0",
                                                  "right_w1", "right_w2"};
    std::vector<char> modelAlphas = {(char)255, (char)128, (char)255, (char)0}; // GT model alpha = 0.5, pred = 1, render = 0

    /// == Pre-process to compute the mesh vertices and indices for all the robot parts
    std::vector<std::vector<float3> > meshVertices, transformedMeshVertices;
    std::vector<std::vector<float4> > meshVerticesWMeshID;
    std::vector<uchar3> meshColors;
    std::vector<pangolin::GlBuffer> meshIndexBuffers;
    std::vector<std::vector<pangolin::GlBuffer *> > meshVertexAttributeBuffers;
    std::vector<int> meshFrameids, meshModelids;

    // Get the model
    int m = baxterID_render;
    for (int s = 0; s < tracker.getModel(m).getNumSdfs(); ++s)
    {
        // Get the frame number for the SDF and it's transform w.r.t robot base
        int f = tracker.getModel(m).getSdfFrameNumber(s);
        uchar3 color = tracker.getModel(m).getSdfColor(s);
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
            meshColors.push_back(color); // SDF color
            meshModelids.push_back(m); // ID of the model for that particular mesh

            // Get their vertices and transform them using the given frame to model transform
            meshVertices.push_back(std::vector<float3>(mesh.nVertices));
            transformedMeshVertices.push_back(std::vector<float3>(mesh.nVertices));
            meshVerticesWMeshID.push_back(std::vector<float4>(mesh.nVertices));
            for(int i = 0; i < mesh.nVertices; ++i)
            {
                // Get the mesh vertex (local coords)
                float3 vertex;
                if (data->oldgrippermodel) {
                    vertex = mesh.vertices[i];
                }
                else {
                    // The mesh might have a rotation / translation / scale to get it to the current frame's reference.
                    float3 rotV   = geomTransform * mesh.vertices[i]; // Rotate / translate to get to frame reference
                    vertex        = make_float3(geomScale.x * rotV.x,
                                                geomScale.y * rotV.y,
                                                geomScale.z * rotV.z); // Scale the vertex
                }

                // Transform mesh vertex
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

    // Enable blending once
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    data->init_done = true;

    // Run till asked to terminate
    while (!pangolin::ShouldQuit() && !terminate_pangolin)
    {
        // General stuff
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        if (pangolin::HasResized())
        {
            pangolin::DisplayBase().ActivateScissorAndClear();
        }

        //////////////////////// ==== RENDER DEPTH IMAGE ==== ///////////////////////
        // Render depth image
        boost::mutex::scoped_lock render_lock(data->renderMutex);

        if (data->render)
        {
            /// == Update the pose of the robot(s)?
            for(int k = 0; k < valid_joint_names.size(); k++)
            {
                // Set baxter GT
                if (joint_name_to_pose_dim.find(valid_joint_names[k]) != joint_name_to_pose_dim.end())
                {
                    int pose_dim = joint_name_to_pose_dim[valid_joint_names[k]];
                    tracker.getPose(baxterID_render).getReducedArticulation()[pose_dim] = data->render_jts[k]; //fmod(data->render_jts[k], 2*M_PI); // TODO: Check
                }
            }

            // Update the pose so that the FK is computed properly
            tracker.updatePose(baxterID_render);

            /// == Update mesh vertices based on new pose
            data->mesh_transforms.clear();
            for (int k = 0; k < meshVertices.size(); k++)
            {
                // Get the SE3 transform for the frame
                int m = meshModelids[k];
                int f = meshFrameids[k];
                const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);
                data->mesh_transforms.push_back(dart::SE3(tfm.r0, tfm.r1, tfm.r2)); // Save the transforms

                // Transform the canonical vertices based on the new transform
                for(int j = 0; j < meshVertices[k].size(); ++j)
                {
                    transformedMeshVertices[k][j] = tfm * meshVertices[k][j];
                }

                // Upload to pangolin vertex buffer
                meshVertexAttributeBuffers[k][0]->Upload(transformedMeshVertices[k].data(), transformedMeshVertices[k].size()*sizeof(float3));
                meshVertexAttributeBuffers[k][1]->Upload(meshVerticesWMeshID[k].data(), meshVerticesWMeshID[k].size()*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
            }

            /// == Render a depth image
            glEnable(GL_DEPTH_TEST);
            renderer.renderMeshes<l2s::RenderDepth>(meshVertexAttributeBuffers, meshIndexBuffers);

            // Get the depth data as a float
            renderer.texture<l2s::RenderDepth>().Download(data->rendered_img);

            // == Convert the full-res depth image to 50% res
            data->depth_img_f     = cv::Mat(glHeight, glWidth, CV_32FC1, data->rendered_img);
            cv::resize(data->depth_img_f, data->depth_img_f_sub, cv::Size(data->imgWidth, data->imgHeight), 0, 0, CV_INTER_NN); // Do NN interpolation

            // Get float pointer
            float *p = data->rendered_img_sub;
            for(int i  = 0; i < data->depth_img_f_sub.rows; i++){
                memcpy(p, data->depth_img_f_sub.ptr(i), data->depth_img_f_sub.cols*sizeof(float));
                p += data->depth_img_f_sub.cols;
            }

            /// == Render a vertex map with the mesh ids (Had to do this to avoid weird bug with rendering model)
            renderer.renderMeshes<l2s::RenderVertMapWMeshID>(meshVertexAttributeBuffers, meshIndexBuffers);

            // Get the vertex image with mesh id
            renderer.texture<l2s::RenderVertMapWMeshID>().Download(data->rendered_vertmap, GL_RGBA, GL_FLOAT);
            glDisable(GL_DEPTH_TEST); // ===== DAMMMMM - Make sure not to disable this before u render the meshes!! ===== //

            // == Convert the full-res vertmap image to 50% res
            data->vertmap_f     = cv::Mat(glHeight, glWidth, CV_32FC4, data->rendered_vertmap);
            cv::resize(data->vertmap_f, data->vertmap_f_sub, cv::Size(data->imgWidth, data->imgHeight), 0, 0, CV_INTER_NN); // Do NN interpolation

            // Get float pointer
            float *p1 = data->rendered_vertmap_sub;
            int ct = 0;
            for(int r = 0; r < data->imgHeight; r++)
            {
                for(int c = 0; c < data->imgWidth; c++)
                {
                    cv::Vec4f v = data->vertmap_f_sub.at<cv::Vec4f>(r,c);
                    p1[ct+0] = v[0];
                    p1[ct+1] = v[1];
                    p1[ct+2] = v[2];
                    p1[ct+3] = v[3];
                    ct+=4;
                }
            }

            // Finished rendering
            data->render = false;
        }

        render_lock.unlock();

        ///////////////////////// SHOW RESULTS / GT ////////////////////////////

        // Update in the mutex
        boost::mutex::scoped_lock update_lock(data->dataMutex);

        /// GUI STUFF BEFORE DISPLAY
        // If input point cloud is not chosen, do not show any associations to it
        if (!showInit)
        {
            showInputAssoc = false;
            showPredAssoc = false;
        }

        // Network input
        if (!showNetInput)
        {
            showInputAssoc = false; // Do not show net input association
        }

        // Network prediction & gradient
        if (!showNetPred)
        {
            showNetGrad  = false;  // Do not show gradients
            showPredAssoc = false; // Do not show predicted association
        }

        // Network prediction & gradient (un-warped)
        if (!showNetPredUnWarped)
        {
            showNetGradUnWarped  = false; // Do not show gradients
        }

        //// DISPLAY PT CLOUDS /////

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
        glMultMatrixf(modelViewInv.data());

        //////////////////// ======= DRAW ALL GRADIENTS AND ASSOCIATIONS ========= ///////////////////////

        // Params
        int npts = data->imgHeight * data->imgWidth;
        int aden = arrowDensity;
        bool norm_grad     = normalizeGrad ? true : false;
        bool show_bg_assoc = showBGAssoc ? true : false;
        float scale = normalizeGrad ? scaleNormalizedGrad : scaleUnNormalizedGrad;

        // Draw
        glBegin(GL_LINES);
        for (int r = 0; r < data->imgHeight; r+=aden) // copy over from the flow matrix
        {
           for (int c = 0; c < data->imgWidth; c+=aden)
           {
               // Get pt index & init point
               int id = r * data->imgWidth + c;
               float xi = data->init_cloud[id + 0*npts];
               float yi = data->init_cloud[id + 1*npts];
               float zi = data->init_cloud[id + 2*npts];

               // Init -> Curr Input associations (Yellow - 110)
               if (showInputAssoc)
               {
                   // Get points from the current input cloud (warped to match the input)
                   float xi_a = data->warpedcurrinput_cloud[id + 0*npts];
                   float yi_a = data->warpedcurrinput_cloud[id + 1*npts];
                   float zi_a = data->warpedcurrinput_cloud[id + 2*npts];
                   int wid = (int)data->current_da_ids[id] - 1; // LUA is 1-indexed
                   bool BG = show_bg_assoc ? false : (wid == -2); // Highlights BG pts (don't care if we are asked to show those pts)
                   if (zi > 0 && !BG)
                   {
                       glColor3ub(255,255,0); // Y
                       glVertex3f(xi, yi, zi);
                       glVertex3f(xi_a, yi_a, zi_a);
                   }
               }

               // Init -> Curr Prediction associations (011)
               if (showPredAssoc)
               {
                   // ASSUMPTION: predicted pts are warped to be the same indexing as the initial point cloud
                   // Get points from the current predicted cloud (warped to match the input)
                   float xi_a = data->currpred_cloud[id + 0*npts];
                   float yi_a = data->currpred_cloud[id + 1*npts];
                   float zi_a = data->currpred_cloud[id + 2*npts];
                   int wid = (int)data->current_da_ids[id] - 1; // LUA is 1-indexed
                   bool BG = show_bg_assoc ? false : (wid == -2); // Highlights BG pts (don't care if we are asked to show those pts)
                   if (zi > 0 && !BG)
                   {
                       glColor3ub(0,255,255);
                       glVertex3f(xi, yi, zi);
                       glVertex3f(xi_a, yi_a, zi_a);
                   }
               }

               // Curr Gradient (111)
               if (showNetGrad)
               {
                   // Get predicted point
                   float xp = data->currpred_cloud[id + 0*npts];
                   float yp = data->currpred_cloud[id + 1*npts];
                   float zp = data->currpred_cloud[id + 2*npts];

                   //  Gradients for the predicted point
                   float xg = data->currpredgrad_cloud[id + 0*npts];
                   float yg = data->currpredgrad_cloud[id + 1*npts];
                   float zg = data->currpredgrad_cloud[id + 2*npts];
                   float norm = sqrt(xg*xg + yg*yg + zg*zg); // Norm of the gradient
                   float xgn = 0, ygn = 0, zgn = 0;
                   if (norm > 0)
                   {
                       float mnorm = (norm > 1e-9) ? norm : 1e-9;
                       xgn = norm_grad ? -xg/mnorm : -xg;
                       ygn = norm_grad ? -yg/mnorm : -yg;
                       zgn = norm_grad ? -zg/mnorm : -zg;
                   }

                   // Draw a line from prediction in direction of gradient (with scaling)
                   glColor3ub(255,255,255); // White
                   glVertex3f(xp, yp, zp);
                   glVertex3f(xp + scale*xgn, yp + scale * ygn, zp + scale * zgn);
               }

               // Curr Gradient (unwarped) (0.5,0.5,0.5)
               if (showNetGradUnWarped)
               {
                   // Get predicted point
                   float xp = data->currpreduw_cloud[id + 0*npts];
                   float yp = data->currpreduw_cloud[id + 1*npts];
                   float zp = data->currpreduw_cloud[id + 2*npts];

                   //  Gradients for the predicted point
                   float xg = data->currpredgraduw_cloud[id + 0*npts];
                   float yg = data->currpredgraduw_cloud[id + 1*npts];
                   float zg = data->currpredgraduw_cloud[id + 2*npts];
                   float norm = sqrt(xg*xg + yg*yg + zg*zg); // Norm of the gradient
                   float xgn = 0, ygn = 0, zgn = 0;
                   if (norm > 0)
                   {
                       float mnorm = (norm > 1e-9) ? norm : 1e-9;
                       xgn = norm_grad ? -xg/mnorm : -xg;
                       ygn = norm_grad ? -yg/mnorm : -yg;
                       zgn = norm_grad ? -zg/mnorm : -zg;
                   }

                   // Draw a line from prediction in direction of gradient (with scaling)
                   glColor3ub(128,128,128); // Gray
                   glVertex3f(xp, yp, zp);
                   glVertex3f(xp + scale*xgn, yp + scale * ygn, zp + scale * zgn);
               }
           }
        }
        glEnd();

        //////////////////// ======= DRAW ALL POINTS ========= ///////////////////////

        // Compute normals
        compute_normals(data->init_cloud, data->initnorm_cloud, data->imgWidth, data->imgHeight);
        compute_normals(data->currinput_cloud, data->currinputnorm_cloud, data->imgWidth, data->imgHeight);
        compute_normals(data->currpreduw_cloud, data->currpreduwnorm_cloud, data->imgWidth, data->imgHeight);
        compute_normals(data->final_cloud, data->finalnorm_cloud, data->imgWidth, data->imgHeight);
        compute_normals(data->finalobs_cloud, data->finalobsnorm_cloud, data->imgWidth, data->imgHeight);

        glPointSize(3.0);
        glBegin(GL_POINTS);
        for (int r = 0; r < data->imgHeight; r++) // copy over from the flow matrix
        {
           for (int c = 0; c < data->imgWidth; c++)
           {
               // Get pt index
               int id = r * data->imgWidth + c;

               // Input point cloud (blue)
               if (showInit)
               {
                   // Fixed color
                   //glColor3ub(255,0,0);

                   // Color based on 3D point
                   float3 pt = make_float3(data->init_cloud[id + 0*npts],
                                           data->init_cloud[id + 1*npts],
                                           data->init_cloud[id + 2*npts]);
                   float3 color = get_color(pt, cube_min, cube_max);
                   glColor3f(color.x,color.y,color.z);

                   // Render pt and normals
                   glVertex3f(data->init_cloud[id + 0*npts],
                              data->init_cloud[id + 1*npts],
                              data->init_cloud[id + 2*npts]);
                   //glNormal3f(data->initnorm_cloud[id + 0*npts],
                   //           data->initnorm_cloud[id + 1*npts],
                   //           data->initnorm_cloud[id + 2*npts]);
               }

               // Current (blue)
               if (showNetInput)
               {
                   // Fixed color
                   //glColor3f(0,0,1);

                   // Color based on 3D point
                   float3 pt = make_float3(data->currinput_cloud[id + 0*npts],
                                           data->currinput_cloud[id + 1*npts],
                                           data->currinput_cloud[id + 2*npts]);
                   float3 color = get_color(pt, cube_min, cube_max);
                   glColor3f(color.y,color.x,color.z); // flip R & G so that right arm is red & left is green

                   // Render pt and normals
                   glNormal3f(data->currinputnorm_cloud[id + 0*npts],
                              data->currinputnorm_cloud[id + 1*npts],
                              data->currinputnorm_cloud[id + 2*npts]);
                   glVertex3f(data->currinput_cloud[id + 0*npts],
                              data->currinput_cloud[id + 1*npts],
                              data->currinput_cloud[id + 2*npts]);

               }

               // Prediction (101)
               if (showNetPred)
               {
                   glColor3f(1,0,1);
                   glVertex3f(data->currpred_cloud[id + 0*npts],
                              data->currpred_cloud[id + 1*npts],
                              data->currpred_cloud[id + 2*npts]);
                   //glNormal3f(data->currprednorm_cloud[id + 0*npts],
                   //           data->currprednorm_cloud[id + 1*npts],
                   //           data->currprednorm_cloud[id + 2*npts]);
               }

               // Prediction unwarped (0.5,0,0.5)
               if (showNetPredUnWarped)
               {
                   glColor3f(0.5,0,0.5);
                   glVertex3f(data->currpreduw_cloud[id + 0*npts],
                              data->currpreduw_cloud[id + 1*npts],
                              data->currpreduw_cloud[id + 2*npts]);
                   //glNormal3f(data->currpreduwnorm_cloud[id + 0*npts],
                    //          data->currpreduwnorm_cloud[id + 1*npts],
                    //          data->currpreduwnorm_cloud[id + 2*npts]);
               }

               // Target (green)
               if (showFinal)
               {
                   // Fixed color
                   glColor3f(0,1,0);

                   // Render pts and normals
                   glNormal3f(data->finalnorm_cloud[id + 0*npts],
                              data->finalnorm_cloud[id + 1*npts],
                              data->finalnorm_cloud[id + 2*npts]);
                   glVertex3f(data->final_cloud[id + 0*npts],
                              data->final_cloud[id + 1*npts],
                              data->final_cloud[id + 2*npts]);
               }

               // Target Observation (green)
               if (showFinalObs)
               {
                   // Fixed color
                   glColor3f(0,1,0);

                   // Render pts and normals
                   glNormal3f(data->finalobsnorm_cloud[id + 0*npts],
                              data->finalobsnorm_cloud[id + 1*npts],
                              data->finalobsnorm_cloud[id + 2*npts]);
                   glVertex3f(data->finalobs_cloud[id + 0*npts],
                              data->finalobs_cloud[id + 1*npts],
                              data->finalobs_cloud[id + 2*npts]);
               }
           }
        }
        glEnd();

        // Draw the 3D origin frame
        //drawAxis(1);

        glPopMatrix(); // Remove inverse transform
        glColor4ub(255,255,255,255);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        glDisable(GL_LIGHTING);

        //////////////////// ======= SHOW ARM CONFIGS ========= ///////////////////////

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        camDisp.ActivateScissorAndClear(camState);

        for(int k = 0; k < valid_joint_names.size(); k++)
        {
            // Set baxter GT
            if (joint_name_to_pose_dim.find(valid_joint_names[k]) != joint_name_to_pose_dim.end())
            {
                int pose_dim = joint_name_to_pose_dim[valid_joint_names[k]];
                tracker.getPose(baxterID_in).getReducedArticulation()[pose_dim] = data->current_jts[k]; // was init_jts
                tracker.getPose(baxterID_tar).getReducedArticulation()[pose_dim] = data->final_jts[k];
                tracker.getPose(baxterID_curr).getReducedArticulation()[pose_dim] = data->current_jts[k];
            }
        }

        // Update the pose
        tracker.updatePose(baxterID_in);
        tracker.updatePose(baxterID_tar);
        tracker.updatePose(baxterID_curr);

        //////////////////// ======= SHOW MASKED PT CLOUD ========= ///////////////////////

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        maskDisp.ActivateScissorAndClear(camState);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor4ub(0xff,0xff,0xff,0xff);

        // Apply inverse of modelview to get pts in model frame
        glMultMatrixf(modelViewInv.data());

        // Create colorized version of mask
        cv::Mat float_mask(data->imgHeight, data->imgWidth, CV_32FC1, data->currmask_img);

        // Convert to grayscale and apply colormap
        cv::Mat gray_mask, colorized_mask;
        float_mask.convertTo(gray_mask, CV_8UC1, (255.0/data->nSE3));
        cv::applyColorMap(gray_mask, colorized_mask, maskColorMap);

        // Render input point cloud and color it based on the masks (opencv color map)
        glPointSize(3.0);
        glBegin(GL_POINTS);
        for (int r = 0; r < data->imgHeight; r++) // copy over from the flow matrix
        {
           for (int c = 0; c < data->imgWidth; c++)
           {
               // Get pt index
               int id = r * data->imgWidth + c;

               // Current point cloud
               // Color based on 3D point
               cv::Vec3b color = colorized_mask.at<cv::Vec3b>(r,c);
               glColor3ub(color[0],color[1],color[2]);

               // Render pt and normals
               if (useNormalsForMaskColor)
               {
                   glNormal3f(data->currinputnorm_cloud[id + 0*npts],
                              data->currinputnorm_cloud[id + 1*npts],
                              data->currinputnorm_cloud[id + 2*npts]);
               }

               // Plot point
               glVertex3f(data->currinput_cloud[id + 0*npts],
                          data->currinput_cloud[id + 1*npts],
                          data->currinput_cloud[id + 2*npts]);
           }
        }

        glEnd();

        // Draw the 3D origin frame
        //drawAxis(1);

        glPopMatrix(); // Remove inverse transform
        glColor4ub(255,255,255,255);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        glDisable(GL_LIGHTING);


        //////////////////// ======= SHOW Frames ========= ///////////////////////

        if (showAllFrames)
            for(int i = 0; i < data->nSE3; i++) *frameStatus[i] = true; // Reset options

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        poseDisp.ActivateScissorAndClear(camStatePose);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor4ub(0xff,0xff,0xff,0xff);

        // Apply inverse of modelview to get pts in model frame
        glMultMatrixf(modelViewInv.data());

        // Draw all the poses
        for(int i = 0; i < data->nSE3; i++)
        {
            // Display or not
            if(*frameStatus[i])
            {
                // Create the different SE3s
                dart::SE3 init = createSE3FromRt(&data->init_poses[i*12]);
                dart::SE3 tar  = createSE3FromRt(&data->tar_poses[i*12]);
                dart::SE3 curr = createSE3FromRt(&data->curr_poses[i*12]);
                if (showInitPoses) drawFrame(init, 0.5, frameLength, lineWidth);
                if (showTarPoses)  drawFrame(tar,  0.75, frameLength, lineWidth);
                if (showCurrPoses) drawFrame(curr, 1, frameLength, lineWidth);
            }
        }

        // Finish
        glPopMatrix(); // Remove inverse transform
        glColor4ub(255,255,255,255);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        glDisable(GL_LIGHTING);

        // Print model view matrix of camStatePose
        if (pangolin::Pushed(printPoseModelView))
            cout << camStatePose.GetModelViewMatrix();

        //////////////////// ======= Finish ========= ///////////////////////

        // Render frame
        renderPangolinFrame(tracker, camState, camDisp, modelAlphas);

        // Save frames to disk now
        if (updated_masks && save_frames)
        {
            std::string filename = save_frames_dir + "/render" + std::to_string(framectr);
            allDisp.SaveOnRender(filename);
            framectr++; // Increment frame counter
            updated_masks = false; // Since we have already saved the image for this particular mask update!
        }

        // ==  Finished updating
        update_lock.unlock();
        usleep(100);
     }
}

//////////////// == FFI functions for calling from LUA == /////////////////////
/// \brief initialize_viz - Initialize pangolin based visualizer
/// \param batchSize - Train / test batch size
/// \param imgHeight - Height of image
/// \param imgWidth  - Width of image
/// \param imgScale  - Scale that converts from "m" to image depth resolution
/// \param fx        - Focal length of camera (x)
/// \param fy        - Focal length of camera (y)
/// \param cx        - COP of camera (x)
/// \param cy        - COP of camera (y)
///
PangolinViz::PangolinViz(int seqLength, int imgHeight, int imgWidth, float imgScale, int nSE3,
                         float fx, float fy, float cx, float cy, float dt, int oldgrippermodel,
                         const std::string savedir)
{
    printf("==> [PANGOLIN_VIZ] Initializing data for visualizer \n");
    data = boost::shared_ptr<LuaData>(new LuaData(seqLength, imgHeight, imgWidth, imgScale, nSE3,
                                         fx, fy, cx, cy, dt, (oldgrippermodel != 0), savedir));

    /// ===== PANGOLIN viewer
    printf("==> [PANGOLIN_VIZ] Starting pangolin in a separate thread \n");
    pangolin_gui_thread.reset(new boost::thread(run_pangolin, data));

    while(!data->init_done) { usleep(100); }
    printf("==> [PANGOLIN_VIZ] Finished initializing pangolin visualizer \n");
    return;
}

cv::Mat project_ptcloud(const float *cloud, const float fx, const float fy,
                        const float cx, const float cy,
                        const int imgHeight, const int imgWidth)
{
    // Project point cloud
    cv::Mat image(imgHeight, imgWidth, CV_32FC1, cv::Scalar(HUGE_VALF));
    int npts = imgHeight * imgWidth;
    for(int r = 0; r < imgHeight; r++)
    {
        for(int c = 0; c < imgWidth; c++)
        {
            // Get input point
            int id = r * imgWidth + c;
            float x = cloud[id + 0*npts];
            float y = cloud[id + 1*npts];
            float z = cloud[id + 2*npts];
            if (z <= 0) continue;

            // Compute pixel coordinates based on focal length and COP
            float xpix = ((x/z) * fx) + cx;// + 1; // Points go from [0, row-1] & [0, col-1] in original data
            float ypix = ((y/z) * fy) + cy;// + 1;

            // Check projection success / Check limits / Do the depth test
            int xpixr = round(xpix); // Rounded off pixel col
            int ypixr = round(ypix); // Rounded off pixel row
            if (xpixr >= 0 && xpixr < imgWidth && ypixr >= 0 && ypixr < imgHeight)
            {
                // Do depth test:
                //   If z >= z at pixel, discard this point
                //   Else z at pixel = z
                float zo = image.at<float>(ypixr, xpixr);
                if ((zo == 0) || (z < zo)) {
                    image.at<float>(ypixr, xpixr) = z;
                }
            }
        }
    }

    // Reset all HUGE_VALF to 0
    for(int r = 0; r < imgHeight; r++)
        for(int c = 0; c < imgWidth; c++)
            if(image.at<float>(r,c) == HUGE_VALF)
                image.at<float>(r,c) = 0;

    return image;
}


////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief Initialize the problem
///
//float ** PangolinViz::initialize_problem(const float *init_jts, const float *final_jts)
void PangolinViz::initialize_problem(const float *start_jts, const float *goal_jts,
                                     float *start_pts, float *da_goal_pts)
{
    // Var
    int npts = data->imgHeight * data->imgWidth;

    /// ==== Render img 1 (@ input joints)
    // Lock
    boost::mutex::scoped_lock render_lock(data->renderMutex);

    // Copy config
    memcpy(data->render_jts, start_jts, 7 * sizeof(float));
    data->render = true;

    // Finished setting data
    render_lock.unlock();

    // Wait till the image is rendered
    while(data->render)
    {
        usleep(1000);
    }

    // Copy rendered data
    float *input_img     = new float[npts];
    float *input_vertmap = new float[npts * 4];
    memcpy(input_img, data->rendered_img_sub, npts * sizeof(float));
    memcpy(input_vertmap,data->rendered_vertmap_sub, 4 * npts * sizeof(float));

    // Saved initial depth image
    cv::Mat depth_img_sub;
    data->depth_img_f_sub.convertTo(depth_img_sub, CV_16UC1, 1e4); // Scale from m to 0.1 mm resolution and save as ushort
    cv::imwrite(data->savedir + "/depthinit.png", depth_img_sub); // Save depth image
    cout << "Saved init depth image at: " << data->savedir << "/depthinit.png" << endl;

    // Get the SE3s
    std::vector<dart::SE3> input_tfms;
    for (int i = 0; i < data->mesh_transforms.size(); i++)
    {
        input_tfms.push_back(dart::SE3(data->mesh_transforms[i].r0, data->mesh_transforms[i].r1, data->mesh_transforms[i].r2));
    }

    /// ==== Render img 2 (@ target joints)
    // Lock
    boost::mutex::scoped_lock render_lock1(data->renderMutex);

    // Copy config
    memcpy(data->render_jts, goal_jts, 7 * sizeof(float));
    data->render = true;

    // Finished setting data
    render_lock1.unlock();

    // Wait till the image is rendered
    while(data->render)
    {
        usleep(1000);
    }

    // Copy rendered data
    float *target_img     = new float[npts];
    float *target_vertmap = new float[npts * 4];
    memcpy(target_img, data->rendered_img_sub, npts * sizeof(float));
    memcpy(target_vertmap,data->rendered_vertmap_sub, 4 * npts * sizeof(float));

    // Get the SE3s
    std::vector<dart::SE3> target_tfms;
    for (int i = 0; i < data->mesh_transforms.size(); i++)
    {
        target_tfms.push_back(dart::SE3(data->mesh_transforms[i].r0, data->mesh_transforms[i].r1, data->mesh_transforms[i].r2));
    }

    /// Lock
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    /// ==== Convert input depth image to point cloud
    memset(data->init_cloud, 0, 3* npts * sizeof(float));
    for(int r = 0; r < data->imgHeight; r++)
    {
        for(int c = 0; c < data->imgWidth; c++)
        {
            // Get X & Y value
            int id = r * data->imgWidth + c;
            float x = (c - data->cx) / data->fx;
            float y = (r - data->cy) / data->fy;

            // Compute input point value
            float zi = (float) input_img[id];
            data->init_cloud[id + 0*npts] = x*zi;
            data->init_cloud[id + 1*npts] = y*zi;
            data->init_cloud[id + 2*npts] = zi;
        }
    }

    /// ==== Convert target depth image to point cloud
    memset(data->finalobs_cloud, 0, 3* npts * sizeof(float));
    for(int r = 0; r < data->imgHeight; r++)
    {
        for(int c = 0; c < data->imgWidth; c++)
        {
            // Get X & Y value
            int id = r * data->imgWidth + c;
            float x = (c - data->cx) / data->fx;
            float y = (r - data->cy) / data->fy;

            // Compute input point value
            float zi = (float) target_img[id];
            data->finalobs_cloud[id + 0*npts] = x*zi;
            data->finalobs_cloud[id + 1*npts] = y*zi;
            data->finalobs_cloud[id + 2*npts] = zi;
        }
    }

    /// ==== Compute target point cloud (input cloud + dense flow)
    memset(data->final_cloud, 0, 3* npts * sizeof(float));
    int ct = 0;
    for(int r = 0; r < data->imgHeight; r++)
    {
        for(int c = 0; c < data->imgWidth; c++, ct+=4)
        {
            // Get the mesh vertex in local co-ordinate frame
            // Point(0,1,2) is the local co-ordinate of the mesh vertex corresponding to the pixel
            // Point(3) is the ID of the mesh from which the vertex originated
            int id = r * data->imgWidth + c;
            const float *point = &(input_vertmap[ct]);
            if (std::round(point[3]) == 0) // Background - zero flow
                continue;

            // Get vertex point (for that pixel) and the mesh to which it belongs
            float3 localVertex = make_float3(point[0], point[1], point[2]);
            int meshid = (int) std::round(point[3]) - 1; // Reduce by 1 to get the ID

            // Transform the vertex from the local co-ordinate frame to the model @ t1 & t2
            float3 modelVertex1 = input_tfms[meshid] * localVertex;
            float3 modelVertex2 = target_tfms[meshid] * localVertex;

            // Get the flow vector in the robot model's frame of reference
            // This is just the difference between the vertex positions = (mV_t2 - mV_t1)
            // This is transformed to the camera frame of reference
            // Note: The flow is a vector, so the 4th homogeneous co-ordinate is zero
            Eigen::Vector4f modelFlow(modelVertex2.x - modelVertex1.x,
                                      modelVertex2.y - modelVertex1.y,
                                      modelVertex2.z - modelVertex1.z,
                                      0.0); // 4th co-ordinate = 0 so that we don't add any translation when transforming
            Eigen::Vector4f cameraFlow = data->modelView * modelFlow;

            // Target cloud = Input cloud + Flow
            data->final_cloud[id + 0*npts] = data->init_cloud[id + 0*npts] + cameraFlow(0);
            data->final_cloud[id + 1*npts] = data->init_cloud[id + 1*npts] + cameraFlow(1);
            data->final_cloud[id + 2*npts] = data->init_cloud[id + 2*npts] + cameraFlow(2);
        }
    }

    // Project target cloud to an image
    cv::Mat target_img_f_sub = project_ptcloud(data->final_cloud, data->fx, data->fy,
                                               data->cx, data->cy, data->imgHeight, data->imgWidth);
    cv::Mat target_img_sub;
    target_img_f_sub.convertTo(target_img_sub, CV_16UC1, 1e4); // Scale from m to 0.1 mm resolution and save as ushort
    cv::imwrite(data->savedir + "/depthtarget.png", target_img_sub); // Save depth image
    cout << "Saved target depth image at: " << data->savedir << "/depthtarget.png" << endl;

    /// ==== Copy input & target configs
    memcpy(data->currinput_cloud, data->init_cloud, 3*npts*sizeof(float)); // Init to current pt cloud
    memcpy(data->init_jts, start_jts, 7*sizeof(float));
    memcpy(data->current_jts, start_jts, 7*sizeof(float));    // Init to input
    memcpy(data->final_jts, goal_jts, 7*sizeof(float));

    // Copy to output
    memcpy(start_pts, data->init_cloud, 3*npts*sizeof(float)); // Copy start pt cloud
    memcpy(da_goal_pts, data->final_cloud, 3*npts*sizeof(float)); // Copy DA-Final pt cloud

    /// Unlock
    update_lock.unlock();

    // Return input & target cloud
    //float **out = (float**) malloc(sizeof(float *)*2);
    //out[0] = data->init_cloud;
    //out[1] = data->final_cloud;
    //return out;
}

//////////////////////
///
/// \brief terminate_viz - Kill the visualizer
///
PangolinViz::~PangolinViz()
{
    // Terminate visualizer (data will be deleted automatically as it is a shared ptr)
    terminate_pangolin = true;
    pangolin_gui_thread->join(); // Wait for thread to join
    printf("==> [PANGOLIN_VIZ] Terminated pangolin visualizer");
}

////////////////////////////////////////////////////////////////////////////////////////
/// == VIZ STUFF FOR RNNs

///
/// \brief update_viz - Get new data from the lua code
///
void PangolinViz::update_viz(const float *inputpts, const float *outputpts_gt,
                             const float *outputpts_pred, const float *jtangles_gt,
                             const float *jtangles_pred)
{
    // Copy over data to the existing vars
    // Update in the mutex
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy data
    memcpy(data->inputPts,       inputpts,       3 * data->imgHeight * data->imgWidth * sizeof(float));
    memcpy(data->outputPts_gt,   outputpts_gt,   3 * data->imgHeight * data->imgWidth * data->seqLength * sizeof(float));
    memcpy(data->outputPts_pred, outputpts_pred, 3 * data->imgHeight * data->imgWidth * data->seqLength * sizeof(float));
    memcpy(data->jtAngles_gt,    jtangles_gt,    7 * (data->seqLength+1) * sizeof(float));
    memcpy(data->jtAngles_pred,  jtangles_pred,  7 * (data->seqLength+1) * sizeof(float));

    // Finished updating
    update_lock.unlock();
}

////////////////////////////////////////////////////////////////////////////////////////
/// === HELPER FUNCTIONS FOR RENDERING DATA

void PangolinViz::render_arm(const float *config,
                             float *rendered_ptcloud)
{
    // Copy over the data and render
    boost::mutex::scoped_lock render_lock(data->renderMutex);

    // Set the render params
    memcpy(data->render_jts, config, 7 * sizeof(float));
    data->render = true;

    // Finished setting data
    render_lock.unlock();

    // Wait till the image is rendered
    while(data->render)
    {
        usleep(1000);
    }

    // Update lock
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Compute current 3D point cloud & update current jt angles
    int npts = data->imgHeight*data->imgWidth;
    memset(data->currinput_cloud, 0, 3*npts*sizeof(float));
    for(int r = 0; r < data->imgHeight; r++)
    {
        for(int c = 0; c < data->imgWidth; c++)
        {
            // Get X & Y value
            int id = r * data->imgWidth + c;
            float x = (c - data->cx) / data->fx;
            float y = (r - data->cy) / data->fy;

            // Compute input point value
            float zi = (float) data->rendered_img_sub[id];
            data->currinput_cloud[id + 0*npts] = x*zi;
            data->currinput_cloud[id + 1*npts] = y*zi;
            data->currinput_cloud[id + 2*npts] = zi;
        }
    }

    // Update curr jt angles
    memcpy(data->current_jts, config, 7*sizeof(float));

    // Copy to output
    memcpy(rendered_ptcloud, data->currinput_cloud, 3*npts*sizeof(float));
    //memcpy(rendered_vertmap, data->rendered_vertmap_sub, 4*npts*sizeof(float));

    // Unlock
    update_lock.unlock();

    // Return image and vertmap
    //float **out = (float**) malloc(sizeof(float *)*3);
    //out[0] = data->rendered_img_sub;
    //out[1] = data->rendered_vertmap_sub;
    //out[2] = data->currinput_cloud;
    //return out;
}


//////////////////////
///
/// \brief update_viz - Get new data from the lua code
///
void PangolinViz::update_da(const float *init_pts, const float *current_pts, const float *current_da_ids, const float * warpedcurrent_pts,
               const float *final_pts)
{
    // Update pangolin pts
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy data
    memcpy(data->init_cloud,                     init_pts,     3 * data->imgHeight * data->imgWidth * sizeof(float));
    memcpy(data->currinput_cloud,             current_pts,     3 * data->imgHeight * data->imgWidth * sizeof(float));
    memcpy(data->final_cloud,                   final_pts,     3 * data->imgHeight * data->imgWidth * sizeof(float));
    memcpy(data->current_da_ids,           current_da_ids,         data->imgHeight * data->imgWidth * sizeof(float));
    memcpy(data->warpedcurrinput_cloud, warpedcurrent_pts,     3 * data->imgHeight * data->imgWidth * sizeof(float));

    // Finished updating
    update_lock.unlock();
}

////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief Compute GT data associations between two configurations and render them
///
void PangolinViz::compute_gt_da(const float *input_jts,
                                const float *target_jts,
                                const int winsize,
                                const float thresh,
                                const float *final_pts,
                                float *gtwarped_out,
                                float *gtda_ids)
{
    // Var
    int npts = data->imgHeight * data->imgWidth;

    /// ==== Render img 1 (@ input joints)
    // Lock
    boost::mutex::scoped_lock render_lock(data->renderMutex);

    // Copy config
    memcpy(data->render_jts, input_jts, 7 * sizeof(float));
    data->render = true;

    // Finished setting data
    render_lock.unlock();

    // Wait till the image is rendered
    while(data->render)
    {
        usleep(1000);
    }

    // Copy rendered data
    float *input_img     = new float[npts];
    float *input_vertmap = new float[npts * 4];
    memcpy(input_img, data->rendered_img_sub, npts * sizeof(float));
    memcpy(input_vertmap,data->rendered_vertmap_sub, 4 * npts * sizeof(float));

    // Get the SE3s
    std::vector<dart::SE3> input_tfms;
    for (int i = 0; i < data->mesh_transforms.size(); i++)
    {
        input_tfms.push_back(dart::SE3(data->mesh_transforms[i].r0, data->mesh_transforms[i].r1, data->mesh_transforms[i].r2));
    }

    /// ==== Render img 2 (@ target joints)
    // Lock
    boost::mutex::scoped_lock render_lock1(data->renderMutex);

    // Copy config
    memcpy(data->render_jts, target_jts, 7 * sizeof(float));
    data->render = true;

    // Finished setting data
    render_lock1.unlock();

    // Wait till the image is rendered
    while(data->render)
    {
        usleep(1000);
    }

    // Copy rendered data
    float *target_img     = new float[npts];
    float *target_vertmap = new float[npts * 4];
    memcpy(target_img, data->rendered_img_sub, npts * sizeof(float));
    memcpy(target_vertmap,data->rendered_vertmap_sub, 4 * npts * sizeof(float));

    // Get the SE3s
    std::vector<dart::SE3> target_tfms;
    for (int i = 0; i < data->mesh_transforms.size(); i++)
    {
        target_tfms.push_back(dart::SE3(data->mesh_transforms[i].r0, data->mesh_transforms[i].r1, data->mesh_transforms[i].r2));
    }

    // Copy over local vertex values
    memcpy(data->local_1, input_vertmap, 4 * npts * sizeof(float));
    memcpy(data->local_2, target_vertmap, 4 * npts * sizeof(float));

    /// ==== Convert depth images to 3D point clouds
    float *input_pts = new float[3* npts];
    float *target_pts = new float[3* npts];
    memset(input_pts, 0, 3* npts * sizeof(float));
    memset(target_pts, 0, 3* npts * sizeof(float));
    for(int r = 0; r < data->imgHeight; r++)
    {
        for(int c = 0; c < data->imgWidth; c++)
        {
            // Get X & Y value
            int id = r * data->imgWidth + c;
            float x = (c - data->cx) / data->fx;
            float y = (r - data->cy) / data->fy;

            // Compute input point value
            float zi = (float) input_img[id];
            input_pts[id + 0*npts] = x*zi;
            input_pts[id + 1*npts] = y*zi;
            input_pts[id + 2*npts] = zi;

            // Compute target point value
            float zt = (float) target_img[id];
            target_pts[id + 0*npts] = x*zt;
            target_pts[id + 1*npts] = y*zt;
            target_pts[id + 2*npts] = zt;
        }
    }

    /// ==== Compute a data-association between the two images
    memset(data->gtwarped_out, 0, 3 * npts * sizeof(float));
    for(int i = 0; i < npts; i++) data->gtda_ids[i] = -1.0;

    // Setup extra params
    float sqthresh   = pow(thresh,2); // Threshold on squared distance
    int winhalfsize  = floor(winsize/2.0); // -winhalfsize -> (-winhalfsize + winsize-1)
    printf("%f, %d \n",sqthresh, winhalfsize);

    // Compute GT flow & flow using proj. GT DA
    float *flow_gt = new float[3 * npts]; for(int i = 0; i < 3*npts; i++) flow_gt[i] = 0.0;
    float *flow_da = new float[3 * npts]; for(int i = 0; i < 3*npts; i++) flow_da[i] = 0.0;

    // Project to get points that are visible in the next frame
    // Iterate over the images and compute the data-associations (using the vertex map images)
    float *mdist = new float[npts]; for(int i = 0; i < npts; i++) mdist[i] = HUGE_VALF;
    int *rpixs = new int[npts]; for(int i = 0; i < npts; i++) rpixs[i] = -1;
    int *cpixs = new int[npts]; for(int i = 0; i < npts; i++) cpixs[i] = -1;
    int bigflow_gt = 0;
    for(int r = 0; r < data->imgHeight; r++)
    {
        for(int c = 0; c < data->imgWidth; c++)
        {
            // Get input point value (local co-ords)
            int id = r * data->imgWidth + c;
            float xi = input_vertmap[id*4+0];
            float yi = input_vertmap[id*4+1];
            float zi = input_vertmap[id*4+2];
            int mi = (int)round(input_vertmap[id*4+3]);

            // In case the ID is background, then skip DA (we need to check for z < 0 => this is local frame of reference, not camera)
            if (mi == 0) continue;

            // Find the 3D point where this vertex projects onto in the current frame
            // Get vertex point (for that pixel) and the mesh to which it belongs
            float3 localVertex = make_float3(xi, yi, zi);
            int meshid = mi - 1; // Reduce by 1 to get the ID

            // Get target 3D point (in camera frame)
            float3 modelVertex_t = target_tfms[meshid] * localVertex;
            Eigen::Vector4f modelVertex_t_1(modelVertex_t.x, modelVertex_t.y, modelVertex_t.z, 1);
            Eigen::Vector4f cameraVertex_t = data->modelView * modelVertex_t_1;

            // Compute flow
            float xp = cameraVertex_t(0);
            float yp = cameraVertex_t(1);
            float zp = cameraVertex_t(2);
            flow_gt[id+0*npts] = xp - input_pts[id+0*npts];
            flow_gt[id+1*npts] = yp - input_pts[id+1*npts];
            flow_gt[id+2*npts] = zp - input_pts[id+2*npts];
            float mag = pow(pow(flow_gt[id+0*npts],2) + pow(flow_gt[id+1*npts],2) + pow(flow_gt[id+2*npts],2), 0.5);
            if (mag > 5e-3) bigflow_gt++;

            // Project target 3D point (in cam frame) onto canvas to get approx pixel location to search for DA
            int cpix = (int) round((xp/zp)*data->fx + data->cx);
            int rpix = (int) round((yp/zp)*data->fy + data->cy);
            if (rpix < 0 || rpix >= data->imgHeight || cpix < 0 || cpix >= data->imgWidth) continue;

            // Check if point is visible (if it has the least z)
            int cid = rpix * data->imgWidth + cpix;
            if (zp < mdist[cid])
            {
                mdist[cid] = zp;
                rpixs[cid] = r;
                cpixs[cid] = c;
            }
        }
    }

    // Iterate over the images and compute the data-associations (using the vertex map images)
    float max_flowerr = 0;
    int nonbg = 0, numass = 0, bigerr = 0;
    int bigflow_ass = 0;
    for(int cid = 0; cid < npts; cid++)
    {
        // Find input point that projected to this pixel
        int r = rpixs[cid];
        int c = cpixs[cid];
        if (r == -1 && c == -1) continue;
        nonbg++; // Not a BG point

        // Get input point value
        int id = r * data->imgWidth + c;
        float xi = input_vertmap[id*4+0];
        float yi = input_vertmap[id*4+1];
        float zi = input_vertmap[id*4+2];
        int mi = (int)round(input_vertmap[id*4+3]);

        // Get rpix and cpix
        int cpix = cid % data->imgWidth;
        int rpix = (cid - cpix) / data->imgWidth;

        // Check in a region around this point
        float mindist = HUGE_VALF;
        int mintr = -1, mintc = -1;
        for (int tr = (rpix-winhalfsize); tr < (rpix-winhalfsize+winsize); tr++)
        {
            for (int tc = (cpix-winhalfsize); tc < (cpix-winhalfsize+winsize); tc++)
            {
                // Check limits
                if (tr < 0 || tr >= data->imgHeight || tc < 0 || tc >= data->imgWidth) continue;

                // Get target value
                int tid = tr * data->imgWidth + tc;
                float xt = target_vertmap[tid*4+0];
                float yt = target_vertmap[tid*4+1];
                float zt = target_vertmap[tid*4+2];
                int mt = (int)round(target_vertmap[tid*4+3]);

                // Compare only in the same mesh, if not continue
                if (mt != mi) continue;

                // Now check distance in local-coordinates
                // If this is closer than previous NN & also less than the outlier threshold, count for loss
                float dist = pow(xi-xt, 2) + pow(yi-yt, 2) + pow(zi-zt, 2);
                if ((dist < mindist) && (dist < sqthresh))
                {
                    mindist = dist;
                    mintr = tr;
                    mintc = tc;
                }
            }
        }

        // In case we found a match, update outputs
        if(mintr != -1 && mintc != -1)
        {
            // == Update DA in camera co-ordinates
            // Get the closest target depth value and convert to 3D point based on focal length etc
            int tid = mintr * data->imgWidth + mintc;
            float x = (mintc - data->cx) / data->fx;
            float y = (mintr - data->cy) / data->fy;
            float z = (float) target_img[tid];

            // Save this point at the corresponding row/col of the input
            data->gtwarped_out[id + 0*npts] = x*z;
            data->gtwarped_out[id + 1*npts] = y*z;
            data->gtwarped_out[id + 2*npts] = z;

            // Add the index of the target point to the input point's position
            data->gtda_ids[id] = tid; //Python is 0-indexed ////// Lua is 1-indexed

            // == Compute flow based on proj. DA & check error against GT flow
            flow_da[id + 0*npts] = x*z - input_pts[id+0*npts];
            flow_da[id + 1*npts] = y*z - input_pts[id+1*npts];
            flow_da[id + 2*npts] =   z - input_pts[id+2*npts];

            // Get error
            float dx = flow_da[id + 0*npts] - flow_gt[id + 0*npts];
            float dy = flow_da[id + 1*npts] - flow_gt[id + 1*npts];
            float dz = flow_da[id + 2*npts] - flow_gt[id + 2*npts];
            float er = pow(pow(dx,2)+pow(dy,2)+pow(dz,2),0.5);
            if (er > max_flowerr) max_flowerr = er;
            if (er > 5e-3) bigerr++;

            // == Check if the point has a big flow
            float mag_gt = pow(pow(flow_gt[id+0*npts],2) + pow(flow_gt[id+1*npts],2) + pow(flow_gt[id+2*npts],2), 0.5);
            if (mag_gt > 1e-2) bigflow_ass++;

            // == Update DA in local co-ordinates
            float xt = target_vertmap[tid*4+0];
            float yt = target_vertmap[tid*4+1];
            float zt = target_vertmap[tid*4+2];
            int mt = (int)round(target_vertmap[tid*4+3]);

            // Save match
            data->local_matches[id*4+0] = xt;
            data->local_matches[id*4+1] = yt;
            data->local_matches[id*4+2] = zt;
            data->local_matches[id*4+3] = mt;
            numass++; // Successful association
        }
    }
    printf("Total: %d, Non-BG: %d, Num Ass: %d, Flow > 0.005m: %d/%d = %f, Err > 0.005m: %d. Max Err: %f \n", npts, nonbg,
           numass, bigflow_ass, bigflow_gt, bigflow_ass * (1.0/bigflow_gt), bigerr, max_flowerr);

    // Update the DA
    update_da(input_pts, target_pts, data->gtda_ids, data->gtwarped_out, final_pts);

    // Free temp memory
    free(input_img);
    free(input_pts);
    free(input_vertmap);
    free(target_img);
    free(target_pts);
    free(target_vertmap);
    free(flow_gt);
    free(flow_da);

    // Memcpy to output data
    memcpy(gtwarped_out, data->gtwarped_out, 3 * npts * sizeof(float));
    memcpy(gtda_ids,     data->gtda_ids,     npts * sizeof(float));

    // Return
    //data->gtda_res[0] = data->gtwarped_out;
    //data->gtda_res[1] = data->gtda_ids;
    //return data->gtda_res;
}

////////////////////////////////////////////////////////////////////////////////////////
/// === SHOW PREDICTIONS VS GRADIENTS

///
/// \brief Predicted points & gradients along with predicted associations (assumes that they are warped to be in same
/// indexing as the initial point cloud against which it is tracked)
///
void PangolinViz::update_pred_pts(const float *net_preds, const float *net_grads)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy vars
    memcpy(data->currpred_cloud,      net_preds,     3 * data->imgHeight * data->imgWidth * sizeof(float));
    memcpy(data->currpredgrad_cloud,  net_grads,     3 * data->imgHeight * data->imgWidth * sizeof(float));

    // Unlock mutex so that display happens
    update_lock.unlock();
}

//////////////////////////////
///
/// \brief Predicted points & gradients along with predicted associations (unwarped)
///
void PangolinViz::update_pred_pts_unwarped(const float *net_preds, const float *net_grads)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy vars
    memcpy(data->currpreduw_cloud,      net_preds,     3 * data->imgHeight * data->imgWidth * sizeof(float));
    memcpy(data->currpredgraduw_cloud,  net_grads,     3 * data->imgHeight * data->imgWidth * sizeof(float));

    // Unlock mutex so that display happens
    update_lock.unlock();
}

//////////////////////////////
///
/// \brief Init & Tar poses for SE3-Control
///
void PangolinViz::initialize_poses(const float *init_poses, const float *tar_poses)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy vars
    memcpy(data->init_poses, init_poses, data->nSE3 * 12 * sizeof(float));
    memcpy(data->tar_poses,  tar_poses,  data->nSE3 * 12 * sizeof(float));

    // Unlock mutex so that display happens
    update_lock.unlock();
}

//////////////////////////////
///
/// \brief Updated predicted masks & Curr poses for SE3-Control
///
void PangolinViz::update_masklabels_and_poses(const float *curr_masks, const float *curr_poses)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy vars
    memcpy(data->currmask_img, curr_masks,  data->imgHeight * data->imgWidth * sizeof(float));
    memcpy(data->curr_poses,   curr_poses,  data->nSE3 * 12 * sizeof(float));
    assert(!updated_masks && "Check code - frame has not been saved after updating the masks previously");
    updated_masks = true;

    // Unlock mutex so that display happens
    update_lock.unlock();
}

//////////////////////////////
///
/// \brief Start saving rendered frames to disk
///
void PangolinViz::start_saving_frames(const std::string framesavedir)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Update flag & the save directory
    save_frames_dir = framesavedir;
    save_frames = true;

    // Unlock mutex so that display happens
    update_lock.unlock();
}


//////////////////////////////
///
/// \brief Stop saving rendered frames to disk
///
void PangolinViz::stop_saving_frames()
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Update flag
    save_frames = false;

    // Unlock mutex so that display happens
    update_lock.unlock();
}
