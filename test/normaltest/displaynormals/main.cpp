#include <pangolin/pangolin.h>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <iostream>

int main( int /*argc*/, char** /*argv*/ )
{  
    pangolin::CreateWindowAndBind("Main",640*2,480);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
//    pangolin::OpenGlRenderState s_cam(
//        pangolin::ProjectionMatrix(640,480,589,589,320,240,0.2,100)
//    );
    // Use default params as for rendering - what we used for the actual dataset rendering (rather than subsampled version)
    int glWidth  = 640;
    int glHeight = 480;
    float glFLx = 589.3664541825391;// not sure what to do about these dimensions
    float glFLy = 589.3664541825391;// not sure what to do about these dimensions
    float glPPx = 320.5;
    float glPPy = 240.5;
    pangolin::OpenGlMatrixSpec glK_pangolin = pangolin::ProjectionMatrixRDF_TopLeft(glWidth,glHeight,glFLx,glFLy,glPPx,glPPy,0.01,1000);

    int panelWidth = 180;
    pangolin::CreatePanel("gui").SetBounds(0.0,1,0,pangolin::Attach::Pix(panelWidth));
    static pangolin::Var<int> arrowDensity("gui.arrowDensity",4,1,10);
    static pangolin::Var<float> arrowLength("gui.arrowLength",0.025,0.005,0.1);

    pangolin::OpenGlRenderState camState1(glK_pangolin);
    pangolin::OpenGlRenderState camState2(glK_pangolin);
    pangolin::View & camDisp1 = pangolin::Display("cam1")
            .SetBounds(0.0, 1.0, 0.0, 1.0)
            .SetAspect(glWidth*1.0f/(glHeight*1.0f))
            .SetHandler(new pangolin::Handler3D(camState1));
    pangolin::View & camDisp2 = pangolin::Display("cam2")
            .SetBounds(0.0, 1.0, 0.0, 1.0)
            .SetAspect(glWidth*1.0f/(glHeight*1.0f))
            .SetHandler(new pangolin::Handler3D(camState2));
    pangolin::View & allDisp = pangolin::Display("multi")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(panelWidth), 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(camDisp1)
            .AddDisplay(camDisp2);

    // Set cam state for pose
    Eigen::Matrix4f modelView = Eigen::Matrix4f::Identity();
//    modelView << 1.0000, -0.0000, -0.0000, -0.0250,
//                -0.0000, -0.5000,  0.8660, -1.4000,
//                -0.0000, -0.8660, -0.5000,  1.7500,
//                 0.0000,  0.0000,  0.0000,  1.0000;
    modelView << -0.0309999,     0.999487,  -0.00799519,     0.152131,
                  0.24277,   -0.000230283,    -0.970084,     0.158486,
                 -0.969588,    -0.0320134,    -0.242639,      1.77066,
                         0,           0,            0,              1;
    Eigen::Matrix4f modelViewInv = modelView.inverse();
    camState1.SetModelViewMatrix(modelView);
    camState2.SetModelViewMatrix(modelView);

    // Read depth and normal images (16UC3)
    cv::Mat cloud1   = cv::imread("cloud1.png", -1);
    cv::Mat cloud2   = cv::imread("cloud2.png", -1);
    cv::Mat normals1 = cv::imread("normals1.png", -1);
    cv::Mat normals2 = cv::imread("normals2.png", -1);
//    std::cout << cloud.depth() << " " << cloud.channels() << std::endl;
//    std::cout << normals.depth() << " " << normals.channels() << std::endl;

    //cv::Mat cloud_u, normals_u;
    //cloud.convertTo(cloud_u, CV_16SC3);
    //normals.convertTo(normals_u, CV_16SC3);

    while( !pangolin::ShouldQuit() )
    {
//        // Clear screen and activate view to render into
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//        camDisp.ActivateScissorAndClear(camState);
////        glMultMatrixf(modelViewInv.data());

        int aden = arrowDensity;

        ///// CLOUD 1
        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        camDisp1.ActivateScissorAndClear(camState1);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor4ub(0xff,0xff,0xff,0xff);

        // Apply inverse of modelview to get pts in model frame
        glMultMatrixf(modelViewInv.data());

        // Show vertmap
        glBegin(GL_POINTS);
        for(int r = 0; r < 240; r+=aden)
        {
            for (int c = 0; c < 320; c+=aden)
            {
                cv::Vec3w n = normals1.at<cv::Vec3w>(r,c);
                cv::Vec3w x = cloud1.at<cv::Vec3w>(r,c);
                glNormal3f(((short) n(0)) * 1e-4,
                           ((short) n(1)) * 1e-4,
                           ((short) n(2)) * 1e-4);
                glVertex3f(((short) x(0)) * 1e-4,
                           ((short) x(1)) * 1e-4,
                           ((short) x(2)) * 1e-4);
            }
        }
        glEnd();

        // Show normal map
        glBegin(GL_LINES);
        for(int r = 0; r < 240; r+=aden)
        {
            for (int c = 0; c < 320; c+=aden)
            {
                cv::Vec3w n = normals1.at<cv::Vec3w>(r,c);
                cv::Vec3w x = cloud1.at<cv::Vec3w>(r,c);
                glColor3f(1.0,0.0,0.0);
                glVertex3f(((short) x(0)) * 1e-4,
                           ((short) x(1)) * 1e-4,
                           ((short) x(2)) * 1e-4);
                glVertex3f(((short) x(0)) * 1e-4 + ((short) n(0)) * 1e-4 * arrowLength,
                           ((short) x(1)) * 1e-4 + ((short) n(1)) * 1e-4 * arrowLength,
                           ((short) x(2)) * 1e-4 + ((short) n(2)) * 1e-4 * arrowLength);
            }
        }
        glEnd();

        ///// CLOUD 2
        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        camDisp2.ActivateScissorAndClear(camState2);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor4ub(0xff,0xff,0xff,0xff);

        // Apply inverse of modelview to get pts in model frame
        glMultMatrixf(modelViewInv.data());

        // Show vertmap
        glBegin(GL_POINTS);
        for(int r = 0; r < 240; r+=aden)
        {
            for (int c = 0; c < 320; c+=aden)
            {
                cv::Vec3w n = normals2.at<cv::Vec3w>(r,c);
                cv::Vec3w x = cloud2.at<cv::Vec3w>(r,c);
                glNormal3f(((short) n(0)) * 1e-4,
                           ((short) n(1)) * 1e-4,
                           ((short) n(2)) * 1e-4);
                glVertex3f(((short) x(0)) * 1e-4,
                           ((short) x(1)) * 1e-4,
                           ((short) x(2)) * 1e-4);
            }
        }
        glEnd();

        // Show normal map
        glBegin(GL_LINES);
        for(int r = 0; r < 240; r+=aden)
        {
            for (int c = 0; c < 320; c+=aden)
            {
                cv::Vec3w n = normals2.at<cv::Vec3w>(r,c);
                cv::Vec3w x = cloud2.at<cv::Vec3w>(r,c);
                glColor3f(1.0,0.0,0.0);
                glVertex3f(((short) x(0)) * 1e-4,
                           ((short) x(1)) * 1e-4,
                           ((short) x(2)) * 1e-4);
                glVertex3f(((short) x(0)) * 1e-4 + ((short) n(0)) * 1e-4 * arrowLength,
                           ((short) x(1)) * 1e-4 + ((short) n(1)) * 1e-4 * arrowLength,
                           ((short) x(2)) * 1e-4 + ((short) n(2)) * 1e-4 * arrowLength);
            }
        }
        glEnd();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
    
    return 0;
}
