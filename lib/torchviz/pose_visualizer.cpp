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
#include "pose_visualizer.hpp"

using namespace std;

//////////////// == Static global variables == /////////////////////
boost::mutex dataMutex;

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

///////// == Draw a 3D frame
void drawFrame(const dart::SE3 se3, const float colorScale, const float frameLength, const float lineWidth, const float3 color)
{
    // Get the 3D points (x,y,z axis at frameLength distance from the center)
    float4 c = se3 * make_float4(0,0,0,1); // Center
    float4 x = se3 * make_float4(frameLength,0,0,1); // add translation to center (x-axis)
    float4 y = se3 * make_float4(0,frameLength,0,1); // add translation to center (y-axis)
    float4 z = se3 * make_float4(0,0,frameLength,1); // add translation to center (z-axis)

    // Line width
    glLineWidth(lineWidth);

    // Draw x-axis
    glColor3f(colorScale*color.x, colorScale*color.y, colorScale*color.z);
    glBegin(GL_LINES);
    glVertex3f(c.x, c.y, c.z);
    glVertex3f(x.x, x.y, x.z);
    glEnd();

    // Draw y-axis
    glColor3f(colorScale*color.x, colorScale*color.y, colorScale*color.z);
    glBegin(GL_LINES);
    glVertex3f(c.x, c.y, c.z);
    glVertex3f(y.x, y.y, y.z);
    glEnd();

    // Draw z-axis
    glColor3f(colorScale*color.x, colorScale*color.y, colorScale*color.z);
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

int glWidth = 640;
int glHeight = 480;
pangolin::Var<bool> *sliderControlled;

void run_pangolin(const boost::shared_ptr<LuaData> data)
{

    /// ===== Set up a DART tracker with the baxter model

    // Setup OpenGL/CUDA/Pangolin stuff - Has to happen before DART tracker initialization
    //cudaGLSetGLDevice(0);
    //cudaDeviceReset();
    const float totalwidth = 2*glWidth + panelWidth;
    const float totalheight = glHeight*3;
    pangolin::CreateWindowAndBind("GD_Baxter: Results",totalwidth,totalheight);
    printf("Initialized Pangolin GUI  \n");

    /// ===== Pangolin initialization
    /// Pangolin mirrors the display, so we need to use TopLeft direndered_imgsplay for it. Our rendering needs BottomLeft

    // -=-=-=- pangolin window setup -=-=-=-
    pangolin::CreatePanel("gui").SetBounds(0.5,1.0,0, pangolin::Attach::Pix(panelWidth));
    pangolin::CreatePanel("config").SetBounds(0.0,0.5,0, pangolin::Attach::Pix(panelWidth));

    // Use default params as for rendering - what we used for the actual dataset rendering (rather than subsampled version)
    float glFLx = 589.3664541825391;// not sure what to do about these dimensions
    float glFLy = 589.3664541825391;// not sure what to do about these dimensions
    float glPPx = 320.5;
    float glPPy = 240.5;
    pangolin::OpenGlMatrixSpec glK_pangolin = pangolin::ProjectionMatrixRDF_TopLeft(glWidth,glHeight,glFLx,glFLy,glPPx,glPPy,0.01,1000);

    // Create a display renderer
    pangolin::OpenGlRenderState camStatePose(glK_pangolin);
    pangolin::View & poseDisp = pangolin::Display("pose")
            .SetBounds(0.0, 0.5, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetAspect(glWidth*1.0f/(glHeight*1.0f))
            .SetHandler(new pangolin::Handler3D(camStatePose));

    // Cam Disp is separate from others
    pangolin::OpenGlRenderState camState(glK_pangolin);
    pangolin::View & camDisp  = pangolin::Display("cam").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & maskDisp = pangolin::Display("mask").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camStatePose));
    pangolin::View & allDisp1 = pangolin::Display("cammask")
            .SetBounds(0.5, 1.0, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(camDisp)
            .AddDisplay(maskDisp);

    /// ===== Pangolin options

    // == GUI options
    pangolin::Var<float> frameLength("gui.frameLength",0.05,0.01,0.2);
    pangolin::Var<float> lineWidth("gui.lineWidth",3,1,10);
    pangolin::Var<bool> showAllFrames("gui.showAllFrames",false,true);
    pangolin::Var<bool> **frameStatus = new pangolin::Var<bool>*[data->nSE3];
    for(int i = 0; i < data->nSE3; i++)
        frameStatus[i] = new pangolin::Var<bool>(dart::stringFormat("gui.showFrame%d",i),true,true);

    // Pose display
    pangolin::Var<bool> showPose("gui.showPose",true,true);
    pangolin::Var<bool> showPredPose("gui.showPredPose",true,true);

    /// ====== Setup model view matrix from disk
    // Model view matrix
    Eigen::Matrix4f modelView = Eigen::Matrix4f::Identity();
    Eigen::read_binary("/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/cameramodelview.dat", modelView);

    // Set cam state
    camState.SetModelViewMatrix(modelView);
    camDisp.SetHandler(new pangolin::Handler3D(camState));

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
    Eigen::Matrix4f modelViewInv = modelView.inverse();

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

    // Assuming that the joints are fixed - change later
    std::vector<std::string> valid_joint_names = {"right_s0", "right_s1", "right_e0", "right_e1", "right_w0",
                                                  "right_w1", "right_w2"};
    std::vector<char> modelAlphas = {(char)128, (char)64}; // GT model alpha = 0.5, pred = 1, render = 0

    // Set controls for moving the robot's joints
    sliderControlled = new pangolin::Var<bool>("config.sliderControlled",false,true);
    pangolin::Var<float> **poseVars    = new pangolin::Var<float>*[valid_joint_names.size()];
    for (int i = 0; i < valid_joint_names.size(); i++)
    {
        // Get max/min vals for joint
        int dim = joint_name_to_pose_dim[valid_joint_names[i]];
        float min_lim = tracker.getModel(baxterID_1).getJointMin(dim);
        float max_lim = tracker.getModel(baxterID_1).getJointMax(dim);
        poseVars[i] = new pangolin::Var<float>(dart::stringFormat("config.jt%d",i+1),
                                               min_lim+0.5*(max_lim-min_lim),min_lim,max_lim);
    }

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
        boost::mutex::scoped_lock data_lock(dataMutex);

        /// ================ Update the robot configs
        float *config = data->config;
        if(*sliderControlled)
        {
            // Config = slider vals
            for(int k = 0; k < 7; k++)
                data->config[k] = *poseVars[k];
        }
        else
        {
            // Config = slider vals
            for(int k = 0; k < 7; k++)
                *poseVars[k] = data->config[k];
        }

        // Update config
        for(int k = 0; k < valid_joint_names.size(); k++)
        {
            // Set baxter GT
            if (joint_name_to_pose_dim.find(valid_joint_names[k]) != joint_name_to_pose_dim.end())
            {
                int pose_dim = joint_name_to_pose_dim[valid_joint_names[k]];
                tracker.getPose(baxterID_1).getReducedArticulation()[pose_dim] = config[k]; // q_1
            }
        }

        // Update the pose so that the FK is computed properly
        tracker.updatePose(baxterID_1);

        /// == Update mesh vertices based on new pose
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

        /// == Render a depth image
        glEnable(GL_DEPTH_TEST);
        renderer.renderMeshes<l2s::RenderDepth>(meshVertexAttributeBuffers, meshIndexBuffers);

        // Get the depth data as a float
        renderer.texture<l2s::RenderDepth>().Download(data->rendered_img);
        glDisable(GL_DEPTH_TEST);

        // Compute pt cloud
        int npts = 240 * 320;
        int r1 = 0;
        for(int r = 0; r < 480; r+=2)
        {
            int c1 = 0;
            for(int c = 0; c < 640; c+=2)
            {
                // Get X & Y value
                int id  = r * 640 + c;
                int id1 = r1 * 320 + c1;
                float x = (c - glPPx) / glFLx;
                float y = (r - glPPy) / glFLy;

                // Compute input point value & set in PCL
                float zi = (float) data->rendered_img[id];
                data->ptcloud[id1 + 0*npts] = x*zi;
                data->ptcloud[id1 + 1*npts] = y*zi;
                data->ptcloud[id1 + 2*npts] = zi;
                c1++;
            }
            r1++;
        }

        //////////////////// ======= SHOW MASKED CURRENT PT CLOUD ========= ///////////////////////

        // Get max val
        std::vector<float3> colors = {make_float3(0.75,0.25,0.5),
                                      make_float3(1,0,0),
                                      make_float3(0,1,0),
                                      make_float3(0,0,1),
                                      make_float3(1,1,0),
                                      make_float3(1,0,1),
                                      make_float3(0,1,1),
                                      make_float3(1,1,1)
                                     };

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        maskDisp.ActivateScissorAndClear(camState);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_NORMALIZE);
        glEnable(GL_COLOR_MATERIAL);
        glColor4ub(0xff,0xff,0xff,0xff);

        // Apply inverse of modelview to get pts in model frame
        glMultMatrixf(modelViewInv.data());

        // Render input point cloud and color it based on the masks (opencv color map)
        glPointSize(3.0);
        glBegin(GL_POINTS);
        for (int r = 0; r < 240; r++) // copy over from the flow matrix
        {
           for (int c = 0; c < 320; c++)
           {
               // Get pt index
               int id = r * 320 + c;

               // Color based on 3D point
               int maxid = -1; float maxval = -HUGE_VAL;
               for (int k = 0; k < data->nSE3; k++)
               {
                   if (data->predmask[id + k*npts] > maxval)
                   {
                       maxval = data->predmask[id + k*npts];
                       maxid = k;
                   }
               }
               float3 color = colors[maxid];
               glColor3f(color.x,color.y,color.z);

               // Plot point
               glVertex3f(data->ptcloud[id + 0*npts],
                          data->ptcloud[id + 1*npts],
                          data->ptcloud[id + 2*npts]);
           }
        }

        glEnd();

        glPopMatrix(); // Remove inverse transform
        glColor4ub(255,255,255,255);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_NORMALIZE);

        /// ================ Render the poses
        if (showAllFrames)
            for(int i = 0; i < data->nSE3; i++) *frameStatus[i] = true; // Reset options

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        poseDisp.ActivateScissorAndClear(camStatePose);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        //glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        //glEnable(GL_LIGHTING);
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
                dart::SE3 p1   = createSE3FromRt(&data->pose[i*data->se3Dim]); // Gt pose 1
                dart::SE3 p1_p = createSE3FromRt(&data->predpose[i*data->se3Dim]); // Pred pose 1

                // Render the different frames
                if (showPose) drawFrame(p1, 0.5, frameLength, lineWidth, colors[i]);
                if (showPredPose) drawFrame(p1_p, 1.0, frameLength, lineWidth, colors[i]);
            }
        }
        // Finish
        glPopMatrix(); // Remove inverse transform
        glColor4ub(255,255,255,255);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_DEPTH_TEST);
        //glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        //glDisable(GL_LIGHTING);

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
PangolinPoseViz::PangolinPoseViz()
{
    printf("==> [PANGOLIN_VIZ] Initializing data for visualizer \n");
    data = boost::shared_ptr<LuaData>(new LuaData());

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
PangolinPoseViz::~PangolinPoseViz()
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
void PangolinPoseViz::update_viz(const float *pose, const float *predpose, const float *predmask,
                                 float *config, float *ptcloud)
{

    // === Get data lock
    boost::mutex::scoped_lock data_mutex(dataMutex);

    memcpy(data->pose, pose, data->nSE3 * data->se3Dim * sizeof(float));
    memcpy(data->predpose, predpose, data->nSE3 * data->se3Dim * sizeof(float));
    memcpy(data->predmask, predmask, data->nSE3 * 240 * 320 * sizeof(float));
    memcpy(config, data->config, 7 * sizeof(float));
    memcpy(ptcloud, data->ptcloud, 3*240*320* sizeof(float));

    // Unlock mutex
    data_mutex.unlock();
}

