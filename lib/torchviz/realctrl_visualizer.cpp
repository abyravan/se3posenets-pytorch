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
#include "realctrl_visualizer.hpp"

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

// Get max val
std::vector<float3> colors = {make_float3(1,1,1),
                              make_float3(1,0,0),
                              make_float3(0,1,0),
                              make_float3(0,0,1),
                              make_float3(1,1,0),
                              make_float3(1,0,1),
                              make_float3(0,1,1),
                              make_float3(0.75,0.25,0.5)
                             };

// Saving files to disk
std::string save_frames_dir;
bool save_frames = false;
int framectr = 0;
bool updated_masks = false;

// Data copy
boost::shared_ptr<PyRealData> datac;
pangolin::DataLog log1, log2;
pangolin::Plotter plotter1, plotter2;

void run_pangolin(const boost::shared_ptr<PyRealData> data)
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

    // -=-=-=- pangolin window setup -=-=-=-
    pangolin::CreatePanel("gui").SetBounds(0.5,1,0,pangolin::Attach::Pix(panelWidth));
    pangolin::CreatePanel("se3gui").SetBounds(0.0,0.5,0,pangolin::Attach::Pix(panelWidth));

    // Add labels
    std::vector<std::string> jterr_labels;
    jterr_labels.push_back(std::string("Joint 1"));
    jterr_labels.push_back(std::string("Joint 2"));
    jterr_labels.push_back(std::string("Joint 3"));
    jterr_labels.push_back(std::string("Joint 4"));
    jterr_labels.push_back(std::string("Joint 5"));
    jterr_labels.push_back(std::string("Joint 6"));
    jterr_labels.push_back(std::string("Joint 7"));
    log1.SetLabels(jterr_labels);

    // OpenGL 'view' of data. We might have many views of the same data.
    plotter1 = Pangolin::Plotter(&log1,0.0f,200.0f,0.0f,75.0f,10.0f,5.0f);
    plotter1.SetBounds(0.5f, 0.75f, pangolin::Attach::Pix(panelWidth), 0.5f);
//    plotter1.Track("$i");

//    // Add some sample annotations to the plot
//    plotter1.AddMarker(pangolin::Marker::Horizontal, -1000, pangolin::Marker::LessThan, pangolin::Colour::Blue().WithAlpha(1.0f) );
//    plotter1.AddMarker(pangolin::Marker::Horizontal,  -900, pangolin::Marker::GreaterThan, pangolin::Colour::Red().WithAlpha(1.0f) );
//    plotter1.AddMarker(pangolin::Marker::Horizontal,  -800, pangolin::Marker::Equal, pangolin::Colour::Green().WithAlpha(1.0f) );
//    plotter1.AddMarker(pangolin::Marker::Horizontal,  -700, pangolin::Marker::LessThan, pangolin::Colour::White().WithAlpha(1.0f) );
//    plotter1.AddMarker(pangolin::Marker::Horizontal,  -600, pangolin::Marker::GreaterThan, pangolin::Colour(0,1,1).WithAlpha(1.0f) );
//    plotter1.AddMarker(pangolin::Marker::Horizontal,  -500, pangolin::Marker::Equal, pangolin::Colour(1,1,0).WithAlpha(1.0f) );
//    plotter1.AddMarker(pangolin::Marker::Horizontal,  -400, pangolin::Marker::LessThan, pangolin::Colour(1,0,1).WithAlpha(1.0f) );

    // === Second plotter (pose error)
    // Labels
    std::vector<std::string> poserr_labels;
    poserr_labels.push_back(std::string("Pose error"));
    log2.SetLabels(poserr_labels);

    // OpenGL 'view' of data. We might have many views of the same data.
    plotter2 = pangolin::Plotter(&log2,0.0f,200.0f,0.0f,150.0f,10.0f,10.0f);
    plotter2.SetBounds(0.5f, 0.75f, 0.5f, 1.0f);
//    plotter2.Track("$i");

//    // Add some sample annotations to the plot
//    plotter2.AddMarker(pangolin::Marker::Horizontal, -1000, pangolin::Marker::LessThan, pangolin::Colour::White().WithAlpha(1.0f) );

    // =====
    // Create a display renderer
    pangolin::OpenGlRenderState camState(glK_pangolin);
    pangolin::OpenGlRenderState camStatePose(glK_pangolin);
    pangolin::View & pcDisp = pangolin::Display("pointcloud").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & maskDisp = pangolin::Display("mask").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & poseDisp = pangolin::Display("pose").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camStatePose));
    pangolin::View & disp3d = pangolin::Display("3d")
            .SetBounds(0.0, 0.5f, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pcDisp)
            .AddDisplay(poseDisp)
            .AddDisplay(maskDisp);
    pangolin::View & allDisp = pangolin::Display("multi")
            .SetBounds(0.0, 0.75f, pangolin::Attach::Pix(panelWidth), 1.0)
            .AddDisplay(plotter1)
            .AddDisplay(plotter2)
            .AddDisplay(disp3d);

    // Cam Disp is separate from others
    pangolin::View & camDisp = pangolin::Display("cam")
            .SetBounds(0.75, 1.0, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetAspect(glWidth*1.0f/(glHeight*1.0f))
            .SetHandler(new pangolin::Handler3D(camState));

    /// ===== Pangolin options

    // Options for color map
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
    const std::string modelFile = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter_rosmesh.xml";

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

    // Assuming that the joints are fixed - change later
    std::vector<std::string> valid_joint_names = {"right_s0", "right_s1", "right_e0", "right_e1", "right_w0",
                                                  "right_w1", "right_w2"};
    std::vector<char> modelAlphas = {(char)255, (char)128}; // GT model alpha = 0.5, pred = 1, render = 0

    /// == Pre-process to compute the mesh vertices and indices for all the robot parts
    std::vector<std::vector<float3> > meshVertices, transformedMeshVertices;
    std::vector<std::vector<float4> > meshVerticesWMeshID;
    std::vector<uchar3> meshColors;
    std::vector<pangolin::GlBuffer> meshIndexBuffers;
    std::vector<std::vector<pangolin::GlBuffer *> > meshVertexAttributeBuffers;
    std::vector<int> meshFrameids, meshModelids;

    // Get the model
    int m = baxterID_in;
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
                // The mesh might have a rotation / translation / scale to get it to the current frame's reference.
                float3 rotV   = geomTransform * mesh.vertices[i]; // Rotate / translate to get to frame reference
                float3 vertex = make_float3(geomScale.x * rotV.x,
                                            geomScale.y * rotV.y,
                                            geomScale.z * rotV.z); // Scale the vertex

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

    // Keep copies of all data
    datac = boost::shared_ptr<PyRealData>(new PyRealData(data->imgHeight, data->imgWidth,
                                                         data->imgScale, data->nSE3,
                                                         data->fx, data->fy,
                                                         data->cx, data->cy, data->savedir));

    // Run till asked to terminate
    int prevctr = 0;
    while (!pangolin::ShouldQuit() && !terminate_pangolin)
    {
        // General stuff
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        if (pangolin::HasResized())
        {
            pangolin::DisplayBase().ActivateScissorAndClear();
        }

        //////////////////////// ==== COPY DATA HERE ==== /////////////////////////////
        boost::mutex::scoped_lock data_lock(data->dataMutex);

        datac->copyData(data);

        data_lock.unlock();

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
                    tracker.getPose(baxterID_in).getReducedArticulation()[pose_dim] = data->render_jts[k]; //fmod(data->render_jts[k], 2*M_PI); // TODO: Check
                }
            }

            // Update the pose so that the FK is computed properly
            tracker.updatePose(baxterID_in);

            /// == Update mesh vertices based on new pose
            std::vector<dart::SE3> mesh_transforms;
            for (int k = 0; k < meshVertices.size(); k++)
            {
                // Get the SE3 transform for the frame
                int m = meshModelids[k];
                int f = meshFrameids[k];
                const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);
                mesh_transforms.push_back(dart::SE3(tfm.r0, tfm.r1, tfm.r2)); // Save the transforms

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

            /// == Render a vertex map with the mesh ids (Had to do this to avoid weird bug with rendering model)
            renderer.renderMeshes<l2s::RenderVertMapWMeshID>(meshVertexAttributeBuffers, meshIndexBuffers);

            // Get the vertex image with mesh id
            renderer.texture<l2s::RenderVertMapWMeshID>().Download(data->rendered_vertmap, GL_RGBA, GL_FLOAT);
            glDisable(GL_DEPTH_TEST); // ===== DAMMMMM - Make sure not to disable this before u render the meshes!! ===== //

            // Get float pointer
            cv::Mat depth_f     = cv::Mat(glHeight, glWidth, CV_32FC1, data->rendered_img);
            cv::Mat vertmap_f   = cv::Mat(glHeight, glWidth, CV_32FC4, data->rendered_vertmap);
            float *p  = data->rendered_img_sub;
            float *p1 = data->rendered_vertmap_sub;
            int ct = 0;
            for(int r = 0; r < data->imgHeight; r++)
            {
                for(int c = 0; c < data->imgWidth; c++)
                {
                    cv::Vec4f v = vertmap_f.at<cv::Vec4f>(2*r,2*c);
                    p1[ct+0] = v[0];
                    p1[ct+1] = v[1];
                    p1[ct+2] = v[2];
                    p1[ct+3] = v[3];
                    ct+=4;

                    p[r*data->imgWidth+c] = depth_f.at<float>(2*r,2*c);
                }
            }

            // Finished rendering
            data->render = false;
        }

        render_lock.unlock();

        ///////////////////////// SHOW RESULTS / GT ////////////////////////////

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

        //////////////////// ======= DRAW ALL POINTS ========= ///////////////////////

        // Params
        int npts = datac->imgHeight * datac->imgWidth;

        // Compute normals
        compute_normals(datac->init_cloud, datac->initnorm_cloud, datac->imgWidth, datac->imgHeight);
        compute_normals(datac->curr_cloud, datac->currnorm_cloud, datac->imgWidth, datac->imgHeight);
        compute_normals(datac->final_cloud, datac->finalnorm_cloud, datac->imgWidth, datac->imgHeight);

        glPointSize(3.0);
        glBegin(GL_POINTS);
        for (int r = 0; r < datac->imgHeight; r++) // copy over from the flow matrix
        {
           for (int c = 0; c < datac->imgWidth; c++)
           {
               // Get pt index
               int id = r * datac->imgWidth + c;

               // ==== Input point cloud (blue)
               // Color based on 3D point
               float3 pti = make_float3(datac->init_cloud[id + 0*npts],
                                       datac->init_cloud[id + 1*npts],
                                       datac->init_cloud[id + 2*npts]);
               float3 colori = get_color(pti, cube_min, cube_max);
               glColor3f(colori.x,colori.y,colori.z);

               // Render pt and normals
               glVertex3f(datac->init_cloud[id + 0*npts],
                          datac->init_cloud[id + 1*npts],
                          datac->init_cloud[id + 2*npts]);
               glNormal3f(datac->initnorm_cloud[id + 0*npts],
                          datac->initnorm_cloud[id + 1*npts],
                          datac->initnorm_cloud[id + 2*npts]);

               // ==== Current (blue)
               // Color based on 3D point
               float3 ptc = make_float3(datac->curr_cloud[id + 0*npts],
                                       datac->curr_cloud[id + 1*npts],
                                       datac->curr_cloud[id + 2*npts]);
               float3 colorc = get_color(ptc, cube_min, cube_max);
               glColor3f(colorc.y,colorc.x,colorc.z); // flip R & G so that right arm is red & left is green

               // Render pt and normals
               glNormal3f(datac->currnorm_cloud[id + 0*npts],
                          datac->currnorm_cloud[id + 1*npts],
                          datac->currnorm_cloud[id + 2*npts]);
               glVertex3f(datac->curr_cloud[id + 0*npts],
                          datac->curr_cloud[id + 1*npts],
                          datac->curr_cloud[id + 2*npts]);


               // ==== Target (green)
               // Color based on 3D point
               float3 ptf = make_float3(datac->final_cloud[id + 0*npts],
                                        datac->final_cloud[id + 1*npts],
                                        datac->final_cloud[id + 2*npts]);
               float3 colorf = get_color(ptf, cube_min, cube_max);
               glColor3f(colorf.y,colorf.x,colorf.z); // flip R & G so that right arm is red & left is green

               // Render pt and normals
               glNormal3f(datac->finalnorm_cloud[id + 0*npts],
                          datac->finalnorm_cloud[id + 1*npts],
                          datac->finalnorm_cloud[id + 2*npts]);
               glVertex3f(datac->final_cloud[id + 0*npts],
                          datac->final_cloud[id + 1*npts],
                          datac->final_cloud[id + 2*npts]);
           }
        }
        glEnd();

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
                tracker.getPose(baxterID_in).getReducedArticulation()[pose_dim] = datac->curr_jts[k]; // was init_jts
                tracker.getPose(baxterID_tar).getReducedArticulation()[pose_dim] = datac->final_jts[k];
            }
        }

        // Update the pose
        tracker.updatePose(baxterID_in);
        tracker.updatePose(baxterID_tar);

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

        // Render input point cloud and color it based on the masks (opencv color map)
        glPointSize(3.0);
        glBegin(GL_POINTS);
        for (int r = 0; r < datac->imgHeight; r++) // copy over from the flow matrix
        {
           for (int c = 0; c < datac->imgWidth; c++)
           {
               // Get pt index
               int id = r * datac->imgWidth + c;

               // Color based on 3D point
               int maxid = -1; float maxval = -HUGE_VAL;
               for (int k = 0; k < datac->nSE3; k++)
               {
                   if (datac->curr_masks[id + k*npts] > maxval)
                   {
                       maxval = datac->curr_masks[id + k*npts];
                       maxid = k;
                   }
               }
               float3 color = colors[maxid];
               glColor3f(color.x,color.y,color.z);

               // Render pt and normals
               if (useNormalsForMaskColor)
               {
                   glNormal3f(datac->currnorm_cloud[id + 0*npts],
                              datac->currnorm_cloud[id + 1*npts],
                              datac->currnorm_cloud[id + 2*npts]);
               }

               // Plot point
               glVertex3f(datac->curr_cloud[id + 0*npts],
                          datac->curr_cloud[id + 1*npts],
                          datac->curr_cloud[id + 2*npts]);
           }
        }

        glEnd();

        glPopMatrix(); // Remove inverse transform
        glColor4ub(255,255,255,255);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        glDisable(GL_LIGHTING);


        //////////////////// ======= SHOW Frames ========= ///////////////////////

        if (showAllFrames)
            for(int i = 0; i < datac->nSE3; i++) *frameStatus[i] = true; // Reset options

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
        for(int i = 0; i < datac->nSE3; i++)
        {
            // Display or not
            if(*frameStatus[i])
            {
                // Create the different SE3s
                dart::SE3 init = createSE3FromRt(&datac->init_poses[i*12]);
                dart::SE3 tar  = createSE3FromRt(&datac->final_poses[i*12]);
                dart::SE3 curr = createSE3FromRt(&datac->curr_poses[i*12]);
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

        //////////////////// ======= Plot errors ========= ///////////////////////

        // Get current counter
        int ctr = datac->deg_errors.size();
        if (ctr == prevctr && ctr > 0)
        {
//            // Joint angle error
//            for (int i = 0; i < datac->deg_errors.size(); i++)
//            {
//                std::vector<float> d = data->deg_errors[i];
//                for (int j = 0; j < d.size(); j++)
//                    log1.Log(d[0], d[1], d[2], d[3], d[4], d[5], d[6]);
//            }

            // Pose error
//            for (int i = 0; i < datac->pose_errors.size(); i++)
            log2.Log(datac->pose_errors.back());

            // Prev ctr update
            prevctr = ctr;
        }

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
RealCtrlViz::RealCtrlViz(int imgHeight, int imgWidth, float imgScale, int nSE3,
                         float fx, float fy, float cx, float cy,
                         const std::string savedir)
{
    printf("==> [REAL-CTRL VIZ] Initializing data for visualizer \n");
    data = boost::shared_ptr<PyRealData>(new PyRealData(imgHeight, imgWidth, imgScale, nSE3,
                                                        fx, fy, cx, cy, savedir));

    /// ===== PANGOLIN viewer
    printf("==> [REAL-CTRL VIZ] Starting pangolin in a separate thread \n");
    pangolin_gui_thread.reset(new boost::thread(run_pangolin, data));

    while(!data->init_done) { usleep(100); }
    printf("==> [REAL-CTRL VIZ] Finished initializing pangolin visualizer \n");
    return;
}

//////////////////////
///
/// \brief terminate_viz - Kill the visualizer
///
RealCtrlViz::~RealCtrlViz()
{
    // Terminate visualizer (data will be deleted automatically as it is a shared ptr)
    terminate_pangolin = true;
    pangolin_gui_thread->join(); // Wait for thread to join
    printf("==> [REAL-CTRL VIZ] Terminated pangolin visualizer");
}

////////////////////////////////////////////////////////////////////////////////////////
/// === HELPER FUNCTIONS FOR RENDERING DATA

void RealCtrlViz::render_arm(const float *config,
                             float *rendered_ptcloud,
                             float *rendered_labels)
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
    int ct = 0;
    for(int r = 0; r < data->imgHeight; r++)
    {
        for(int c = 0; c < data->imgWidth; c++, ct+=4)
        {
            // Get X & Y value
            int id = r * data->imgWidth + c;
            float x = (c - data->cx) / data->fx;
            float y = (r - data->cy) / data->fy;

            // Compute input point value
            float zi = (float) data->rendered_img_sub[id];
            data->curr_cloud[id + 0*npts] = x*zi;
            data->curr_cloud[id + 1*npts] = y*zi;
            data->curr_cloud[id + 2*npts] = zi;

            // Setup current mask
            data->curr_labels[id] = data->rendered_vertmap_sub[ct+3];
        }
    }

    // Update curr jt angles
    memcpy(data->curr_jts, config, 7*sizeof(float));

    // Copy to output
    memcpy(rendered_ptcloud, data->curr_cloud,  3*npts*sizeof(float));
    memcpy(rendered_labels,  data->curr_labels, npts*sizeof(float));

    // Unlock
    update_lock.unlock();
}

////////////////////////////////////////////////////////////////////////////////////////
/// === SHOW PREDICTIONS VS GRADIENTS

//////////////////////////////
///
/// \brief Updated predicted masks & Curr poses for SE3-Control
///
void RealCtrlViz::update_real_curr(const float *curr_angles,
                                   const float *curr_ptcloud,
                                   const float *curr_poses,
                                   const float *curr_masks,
                                   const float curr_pose_error,
                                   const float *curr_deg_errors)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy vars
    memcpy(data->curr_jts,   curr_angles,  7 * sizeof(float));
    memcpy(data->curr_cloud, curr_ptcloud, data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->curr_masks, curr_masks,   data->imgHeight * data->imgWidth * data->nSE3 * sizeof(float));
    memcpy(data->curr_poses, curr_poses,   data->nSE3 * 12 * sizeof(float));
    assert(!updated_masks && "Check code - frame has not been saved after updating the masks previously");
    updated_masks = true;

    // Append to the errors
    data->pose_errors.push_back(curr_pose_error);
    data->deg_errors.push_back(std::vector<float>(curr_deg_errors, curr_deg_errors+7));

    // Unlock mutex so that display happens
    update_lock.unlock();
}

//////////////////////////////
///
/// \brief Updated predicted masks & Curr poses for SE3-Control
///
void RealCtrlViz::update_real_init(const float *start_angles, const float *start_ptcloud,
                                   const float *start_poses, const float *start_masks,
                                   const float *goal_angles, const float *goal_ptcloud,
                                   const float *goal_poses, const float *goal_masks,
                                   const float start_pose_error, const float *start_deg_errors)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy vars
    memcpy(data->init_jts,    start_angles,  7 * sizeof(float));
    memcpy(data->final_jts,   goal_angles,   7 * sizeof(float));
    memcpy(data->init_poses,  start_poses,   data->nSE3 * 12 * sizeof(float));
    memcpy(data->final_poses, goal_poses,    data->nSE3 * 12 * sizeof(float));
    memcpy(data->init_cloud,  start_ptcloud, data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->final_cloud, goal_ptcloud,  data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->init_masks,  start_masks,   data->imgHeight * data->imgWidth * data->nSE3 * sizeof(float));
    memcpy(data->final_masks, goal_masks,    data->imgHeight * data->imgWidth * data->nSE3 * sizeof(float));

    // Reset the loggers, dynamically change the max values of the loggers here
    this->reset();
    plotter2->SetView(pangolin::XYRangef(Pangolin::Range<float>(0.0f, 200.0f), Pangolin::Range<float>(0.0f, start_pose_error)));

    // Append to the errors
    data->pose_errors.push_back(start_pose_error);
    data->deg_errors.push_back(std::vector<float>(start_deg_errors, start_deg_errors+7));

    // Unlock mutex so that display happens
    update_lock.unlock();
}

//////////////////////////////
///
/// \brief Updated predicted masks & Curr poses for SE3-Control
///
void RealCtrlViz::reset()
{
    data->deg_errors.clear();
    data->pose_errors.clear();
    log1.Clear();
    log2.Clear();
}

//////////////////////////////
///
/// \brief Start saving rendered frames to disk
///
void RealCtrlViz::start_saving_frames(const std::string framesavedir)
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
void RealCtrlViz::stop_saving_frames()
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Update flag
    save_frames = false;

    // Unlock mutex so that display happens
    update_lock.unlock();
}
