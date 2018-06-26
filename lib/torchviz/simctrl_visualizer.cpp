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
#include "simctrl_visualizer.hpp"

using namespace std;


std::vector<std::vector<float> > colormap = {{0.403921568627,0.0,0.121568627451},
                                             {0.415455594002,0.00369088811995,0.123414071511},
                                             {0.426989619377,0.00738177623991,0.125259515571},
                                             {0.438523644752,0.0110726643599,0.127104959631},
                                             {0.450057670127,0.0147635524798,0.128950403691},
                                             {0.461591695502,0.0184544405998,0.130795847751},
                                             {0.473125720877,0.0221453287197,0.132641291811},
                                             {0.484659746251,0.0258362168397,0.134486735871},
                                             {0.496193771626,0.0295271049596,0.136332179931},
                                             {0.507727797001,0.0332179930796,0.138177623991},
                                             {0.519261822376,0.0369088811995,0.140023068051},
                                             {0.530795847751,0.0405997693195,0.141868512111},
                                             {0.542329873126,0.0442906574394,0.143713956171},
                                             {0.553863898501,0.0479815455594,0.145559400231},
                                             {0.565397923875,0.0516724336794,0.147404844291},
                                             {0.57693194925,0.0553633217993,0.149250288351},
                                             {0.588465974625,0.0590542099193,0.151095732411},
                                             {0.6,0.0627450980392,0.152941176471},
                                             {0.611534025375,0.0664359861592,0.154786620531},
                                             {0.62306805075,0.0701268742791,0.156632064591},
                                             {0.634602076125,0.0738177623991,0.158477508651},
                                             {0.646136101499,0.077508650519,0.16032295271},
                                             {0.657670126874,0.081199538639,0.16216839677},
                                             {0.669204152249,0.0848904267589,0.16401384083},
                                             {0.680738177624,0.0885813148789,0.16585928489},
                                             {0.692272202999,0.0922722029988,0.16770472895},
                                             {0.700807381776,0.0996539792388,0.171241830065},
                                             {0.706343713956,0.110726643599,0.176470588235},
                                             {0.711880046136,0.121799307958,0.181699346405},
                                             {0.717416378316,0.132871972318,0.186928104575},
                                             {0.722952710496,0.143944636678,0.192156862745},
                                             {0.728489042676,0.155017301038,0.197385620915},
                                             {0.734025374856,0.166089965398,0.202614379085},
                                             {0.739561707036,0.177162629758,0.207843137255},
                                             {0.745098039216,0.188235294118,0.213071895425},
                                             {0.750634371396,0.199307958478,0.218300653595},
                                             {0.756170703576,0.210380622837,0.223529411765},
                                             {0.761707035755,0.221453287197,0.228758169935},
                                             {0.767243367935,0.232525951557,0.233986928105},
                                             {0.772779700115,0.243598615917,0.239215686275},
                                             {0.778316032295,0.254671280277,0.244444444444},
                                             {0.783852364475,0.265743944637,0.249673202614},
                                             {0.789388696655,0.276816608997,0.254901960784},
                                             {0.794925028835,0.287889273356,0.260130718954},
                                             {0.800461361015,0.298961937716,0.265359477124},
                                             {0.805997693195,0.310034602076,0.270588235294},
                                             {0.811534025375,0.321107266436,0.275816993464},
                                             {0.817070357555,0.332179930796,0.281045751634},
                                             {0.822606689735,0.343252595156,0.286274509804},
                                             {0.828143021915,0.354325259516,0.291503267974},
                                             {0.833679354095,0.365397923875,0.296732026144},
                                             {0.839215686275,0.376470588235,0.301960784314},
                                             {0.843829296424,0.38708189158,0.310111495579},
                                             {0.848442906574,0.397693194925,0.318262206844},
                                             {0.853056516724,0.40830449827,0.326412918108},
                                             {0.857670126874,0.418915801615,0.334563629373},
                                             {0.862283737024,0.42952710496,0.342714340638},
                                             {0.866897347174,0.440138408304,0.350865051903},
                                             {0.871510957324,0.450749711649,0.359015763168},
                                             {0.876124567474,0.461361014994,0.367166474433},
                                             {0.880738177624,0.471972318339,0.375317185698},
                                             {0.885351787774,0.482583621684,0.383467896963},
                                             {0.889965397924,0.493194925029,0.391618608228},
                                             {0.894579008074,0.503806228374,0.399769319493},
                                             {0.899192618224,0.514417531719,0.407920030757},
                                             {0.903806228374,0.525028835063,0.416070742022},
                                             {0.908419838524,0.535640138408,0.424221453287},
                                             {0.913033448674,0.546251441753,0.432372164552},
                                             {0.917647058824,0.556862745098,0.440522875817},
                                             {0.922260668973,0.567474048443,0.448673587082},
                                             {0.926874279123,0.578085351788,0.456824298347},
                                             {0.931487889273,0.588696655133,0.464975009612},
                                             {0.936101499423,0.599307958478,0.473125720877},
                                             {0.940715109573,0.609919261822,0.481276432141},
                                             {0.945328719723,0.620530565167,0.489427143406},
                                             {0.949942329873,0.631141868512,0.497577854671},
                                             {0.954555940023,0.641753171857,0.505728565936},
                                             {0.957554786621,0.651211072664,0.515109573241},
                                             {0.958938869666,0.659515570934,0.525720876586},
                                             {0.96032295271,0.667820069204,0.536332179931},
                                             {0.961707035755,0.676124567474,0.546943483276},
                                             {0.9630911188,0.684429065744,0.557554786621},
                                             {0.964475201845,0.692733564014,0.568166089965},
                                             {0.96585928489,0.701038062284,0.57877739331},
                                             {0.967243367935,0.709342560554,0.589388696655},
                                             {0.96862745098,0.717647058824,0.6},
                                             {0.970011534025,0.725951557093,0.610611303345},
                                             {0.97139561707,0.734256055363,0.62122260669},
                                             {0.972779700115,0.742560553633,0.631833910035},
                                             {0.97416378316,0.750865051903,0.642445213379},
                                             {0.975547866205,0.759169550173,0.653056516724},
                                             {0.97693194925,0.767474048443,0.663667820069},
                                             {0.978316032295,0.775778546713,0.674279123414},
                                             {0.97970011534,0.784083044983,0.684890426759},
                                             {0.981084198385,0.792387543253,0.695501730104},
                                             {0.98246828143,0.800692041522,0.706113033449},
                                             {0.983852364475,0.808996539792,0.716724336794},
                                             {0.98523644752,0.817301038062,0.727335640138},
                                             {0.986620530565,0.825605536332,0.737946943483},
                                             {0.98800461361,0.833910034602,0.748558246828},
                                             {0.989388696655,0.842214532872,0.759169550173},
                                             {0.9907727797,0.850519031142,0.769780853518},
                                             {0.992156862745,0.858823529412,0.780392156863},
                                             {0.991234140715,0.863129565552,0.787773933103},
                                             {0.990311418685,0.867435601692,0.795155709343},
                                             {0.989388696655,0.871741637832,0.802537485582},
                                             {0.988465974625,0.876047673972,0.809919261822},
                                             {0.987543252595,0.880353710111,0.817301038062},
                                             {0.986620530565,0.884659746251,0.824682814302},
                                             {0.985697808535,0.888965782391,0.832064590542},
                                             {0.984775086505,0.893271818531,0.839446366782},
                                             {0.983852364475,0.897577854671,0.846828143022},
                                             {0.982929642445,0.901883890811,0.854209919262},
                                             {0.982006920415,0.906189926951,0.861591695502},
                                             {0.981084198385,0.910495963091,0.868973471742},
                                             {0.980161476355,0.914801999231,0.876355247982},
                                             {0.979238754325,0.919108035371,0.883737024221},
                                             {0.978316032295,0.923414071511,0.891118800461},
                                             {0.977393310265,0.927720107651,0.898500576701},
                                             {0.976470588235,0.932026143791,0.905882352941},
                                             {0.975547866205,0.936332179931,0.913264129181},
                                             {0.974625144175,0.940638216071,0.920645905421},
                                             {0.973702422145,0.944944252211,0.928027681661},
                                             {0.972779700115,0.949250288351,0.935409457901},
                                             {0.971856978085,0.953556324491,0.942791234141},
                                             {0.970934256055,0.957862360631,0.950173010381},
                                             {0.970011534025,0.96216839677,0.957554786621},
                                             {0.969088811995,0.96647443291,0.96493656286},
                                             {0.965705497885,0.967243367935,0.968089196463},
                                             {0.959861591696,0.964475201845,0.967012687428},
                                             {0.954017685506,0.961707035755,0.965936178393},
                                             {0.948173779316,0.958938869666,0.964859669358},
                                             {0.942329873126,0.956170703576,0.963783160323},
                                             {0.936485966936,0.953402537486,0.962706651288},
                                             {0.930642060746,0.950634371396,0.961630142253},
                                             {0.924798154556,0.947866205306,0.960553633218},
                                             {0.918954248366,0.945098039216,0.959477124183},
                                             {0.913110342176,0.942329873126,0.958400615148},
                                             {0.907266435986,0.939561707036,0.957324106113},
                                             {0.901422529796,0.936793540946,0.956247597078},
                                             {0.895578623606,0.934025374856,0.955171088043},
                                             {0.889734717416,0.931257208766,0.954094579008},
                                             {0.883890811226,0.928489042676,0.953018069973},
                                             {0.878046905037,0.925720876586,0.951941560938},
                                             {0.872202998847,0.922952710496,0.950865051903},
                                             {0.866359092657,0.920184544406,0.949788542868},
                                             {0.860515186467,0.917416378316,0.948712033833},
                                             {0.854671280277,0.914648212226,0.947635524798},
                                             {0.848827374087,0.911880046136,0.946559015763},
                                             {0.842983467897,0.909111880046,0.945482506728},
                                             {0.837139561707,0.906343713956,0.944405997693},
                                             {0.831295655517,0.903575547866,0.943329488658},
                                             {0.825451749327,0.900807381776,0.942252979623},
                                             {0.819607843137,0.898039215686,0.941176470588},
                                             {0.809919261822,0.893118031526,0.938408304498},
                                             {0.800230680507,0.888196847366,0.935640138408},
                                             {0.790542099193,0.883275663206,0.932871972318},
                                             {0.780853517878,0.878354479047,0.930103806228},
                                             {0.771164936563,0.873433294887,0.927335640138},
                                             {0.761476355248,0.868512110727,0.924567474048},
                                             {0.751787773933,0.863590926567,0.921799307958},
                                             {0.742099192618,0.858669742407,0.919031141869},
                                             {0.732410611303,0.853748558247,0.916262975779},
                                             {0.722722029988,0.848827374087,0.913494809689},
                                             {0.713033448674,0.843906189927,0.910726643599},
                                             {0.703344867359,0.838985005767,0.907958477509},
                                             {0.693656286044,0.834063821607,0.905190311419},
                                             {0.683967704729,0.829142637447,0.902422145329},
                                             {0.674279123414,0.824221453287,0.899653979239},
                                             {0.664590542099,0.819300269127,0.896885813149},
                                             {0.654901960784,0.814379084967,0.894117647059},
                                             {0.645213379469,0.809457900807,0.891349480969},
                                             {0.635524798155,0.804536716647,0.888581314879},
                                             {0.62583621684,0.799615532488,0.885813148789},
                                             {0.616147635525,0.794694348328,0.883044982699},
                                             {0.60645905421,0.789773164168,0.880276816609},
                                             {0.596770472895,0.784851980008,0.877508650519},
                                             {0.58708189158,0.779930795848,0.874740484429},
                                             {0.577393310265,0.775009611688,0.871972318339},
                                             {0.56647443291,0.768704344483,0.868512110727},
                                             {0.554325259516,0.761014994233,0.864359861592},
                                             {0.542176086121,0.753325643983,0.860207612457},
                                             {0.530026912726,0.745636293733,0.856055363322},
                                             {0.517877739331,0.737946943483,0.851903114187},
                                             {0.505728565936,0.730257593233,0.847750865052},
                                             {0.493579392541,0.722568242983,0.843598615917},
                                             {0.481430219146,0.714878892734,0.839446366782},
                                             {0.469281045752,0.707189542484,0.835294117647},
                                             {0.457131872357,0.699500192234,0.831141868512},
                                             {0.444982698962,0.691810841984,0.826989619377},
                                             {0.432833525567,0.684121491734,0.822837370242},
                                             {0.420684352172,0.676432141484,0.818685121107},
                                             {0.408535178777,0.668742791234,0.814532871972},
                                             {0.396386005383,0.661053440984,0.810380622837},
                                             {0.384236831988,0.653364090734,0.806228373702},
                                             {0.372087658593,0.645674740484,0.802076124567},
                                             {0.359938485198,0.637985390235,0.797923875433},
                                             {0.347789311803,0.630296039985,0.793771626298},
                                             {0.335640138408,0.622606689735,0.789619377163},
                                             {0.323490965013,0.614917339485,0.785467128028},
                                             {0.311341791619,0.607227989235,0.781314878893},
                                             {0.299192618224,0.599538638985,0.777162629758},
                                             {0.287043444829,0.591849288735,0.773010380623},
                                             {0.274894271434,0.584159938485,0.768858131488},
                                             {0.262745098039,0.576470588235,0.764705882353},
                                             {0.257516339869,0.56955017301,0.761168781238},
                                             {0.252287581699,0.562629757785,0.757631680123},
                                             {0.247058823529,0.555709342561,0.754094579008},
                                             {0.241830065359,0.548788927336,0.750557477893},
                                             {0.23660130719,0.541868512111,0.747020376778},
                                             {0.23137254902,0.534948096886,0.743483275663},
                                             {0.22614379085,0.528027681661,0.739946174548},
                                             {0.22091503268,0.521107266436,0.736409073433},
                                             {0.21568627451,0.514186851211,0.732871972318},
                                             {0.21045751634,0.507266435986,0.729334871203},
                                             {0.20522875817,0.500346020761,0.725797770088},
                                             {0.2,0.493425605536,0.722260668973},
                                             {0.19477124183,0.486505190311,0.718723567859},
                                             {0.18954248366,0.479584775087,0.715186466744},
                                             {0.18431372549,0.472664359862,0.711649365629},
                                             {0.17908496732,0.465743944637,0.708112264514},
                                             {0.17385620915,0.458823529412,0.704575163399},
                                             {0.16862745098,0.451903114187,0.701038062284},
                                             {0.16339869281,0.444982698962,0.697500961169},
                                             {0.158169934641,0.438062283737,0.693963860054},
                                             {0.152941176471,0.431141868512,0.690426758939},
                                             {0.147712418301,0.424221453287,0.686889657824},
                                             {0.142483660131,0.417301038062,0.683352556709},
                                             {0.137254901961,0.410380622837,0.679815455594},
                                             {0.132026143791,0.403460207612,0.676278354479},
                                             {0.127258746636,0.395847750865,0.668742791234},
                                             {0.122952710496,0.387543252595,0.657208765859},
                                             {0.118646674356,0.379238754325,0.645674740484},
                                             {0.114340638216,0.370934256055,0.63414071511},
                                             {0.110034602076,0.362629757785,0.622606689735},
                                             {0.105728565936,0.354325259516,0.61107266436},
                                             {0.101422529796,0.346020761246,0.599538638985},
                                             {0.0971164936563,0.337716262976,0.58800461361},
                                             {0.0928104575163,0.329411764706,0.576470588235},
                                             {0.0885044213764,0.321107266436,0.56493656286},
                                             {0.0841983852364,0.312802768166,0.553402537486},
                                             {0.0798923490965,0.304498269896,0.541868512111},
                                             {0.0755863129566,0.296193771626,0.530334486736},
                                             {0.0712802768166,0.287889273356,0.518800461361},
                                             {0.0669742406767,0.279584775087,0.507266435986},
                                             {0.0626682045367,0.271280276817,0.495732410611},
                                             {0.0583621683968,0.262975778547,0.484198385236},
                                             {0.0540561322568,0.254671280277,0.472664359862},
                                             {0.0497500961169,0.246366782007,0.461130334487},
                                             {0.0454440599769,0.238062283737,0.449596309112},
                                             {0.041138023837,0.229757785467,0.438062283737},
                                             {0.036831987697,0.221453287197,0.426528258362},
                                             {0.0325259515571,0.213148788927,0.414994232987},
                                             {0.0282199154171,0.204844290657,0.403460207612},
                                             {0.0239138792772,0.196539792388,0.391926182238},
                                             {0.0196078431373,0.188235294118,0.380392156863}};

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

///////// == Draw a 3D frame
void drawDashedFrame(const dart::SE3 se3, const float colorScale, const float frameLength, const float lineWidth, const float3 color)
{
    // Get the 3D points (x,y,z axis at frameLength distance from the center)
    float4 c = se3 * make_float4(0,0,0,1); // Center
    float4 x = se3 * make_float4(frameLength,0,0,1); // add translation to center (x-axis)
    float4 y = se3 * make_float4(0,frameLength,0,1); // add translation to center (y-axis)
    float4 z = se3 * make_float4(0,0,frameLength,1); // add translation to center (z-axis)

    // Line width
    glLineWidth(lineWidth);

    glPushAttrib(GL_ENABLE_BIT);
    glLineStipple(4, 0xAAAA);
    glEnable(GL_LINE_STIPPLE);

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

    glDisable(GL_LINE_STIPPLE);
    glPopAttrib();
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
//    //glShadeModel (GL_SMOOTH);
//    float4 lightPosition = make_float4(normalize(make_float3(-0.4405,-0.5357,-0.619)),0);
//    GLfloat light_ambient[] = { 0.3, 0.3, 0.3, 1.0 };
//    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
//    glLightfv(GL_LIGHT0, GL_POSITION, (float*)&lightPosition);

    camDisp.ActivateScissorAndClear(camState);

    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
//    glEnable(GL_LIGHTING);

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
//    glDisable(GL_LIGHT0);
    glDisable(GL_NORMALIZE);
//    glDisable(GL_LIGHTING);
    glColor4ub(255,255,255,255);

    /// Finish frame
    pangolin::FinishFrame();
}

/*************** TIMESPEC HELPER STUFF ***************/
int timespec_sub(struct timespec * dst, const struct timespec * src)
{
   dst->tv_sec -= src->tv_sec;
   dst->tv_nsec -= src->tv_nsec;
   if (dst->tv_nsec < 0)
   {
      dst->tv_sec -= 1;
      dst->tv_nsec += 1000000000;
   }
   return 0;
}

double timespec_double(const struct timespec * src)
{
   return src->tv_sec + ((double)(src->tv_nsec))/1000000000.0;
}

int timespec_set_zero(struct timespec * t)
{
   t->tv_sec = 0;
   t->tv_nsec = 0;
   return 0;
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

float maxz = 4.0;
std::vector<float> get_color_based_on_depth(float z)
{
    float zd = (z/maxz) * 255.0;
    if (zd <= 40)
        return std::vector<float>({0,0,0});
    if (zd>40 && zd <= 120)
        zd = ((zd-40)/80) * 127; // scale to 0->127
    else if (zd > 120 && zd <= 200)
        zd = 127 + ((zd-120) / 80)*127;
    else
        zd = 255; // Fixed color
    return colormap[(int)zd];
}

//////////////// == PANGOLIN GUI THREAD == /////////////////////
/// \brief run_pangolin - Function that runs the pangolin GUI till terminate is called
/// \param data - Class instance containing all data copied from LUA
///

int glWidth = 640;
int glHeight = 480;

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

// Saving files to disk
std::string save_frames_dir;
bool save_frames = false;
int framectr = 0;
bool frame_saved = true;

// Data copy
boost::shared_ptr<PySimData> datac;
pangolin::DataLog log_1, log_2;
pangolin::Plotter *plotter1, *plotter2;
int prevctr = 0;

void run_pangolin(const boost::shared_ptr<PySimData> data)
{
    /// ===== Set up a DART tracker with the baxter model

    // Setup OpenGL/CUDA/Pangolin stuff - Has to happen before DART tracker initialization
    //cudaGLSetGLDevice(0);
    //cudaDeviceReset();
    int ncols = data->posenets ? 4 : 3; // Number of cols in the viewer -> RGB | Pts | Poses | Masks (Poses are optional)
    const float totalwidth = 320*ncols + panelWidth;
    const float totalheight = 240*4;
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
    log_1.SetLabels(jterr_labels);

    // OpenGL 'view' of data. We might have many views of the same data.
    plotter1 = new pangolin::Plotter(&log_1,0.0f,200.0f,-75.0f,75.0f,50.0f,25.0f);
    if (data->posenets)
        plotter1->SetBounds(0.0f, 1.0f, 0.0f, 0.5f); // Both pose & jt angle errors
    else
        plotter1->SetBounds(0.0f, 1.0f, 0.0f, 1.0f); // No pose errors
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
//    std::vector<std::string> poserr_labels;
//    poserr_labels.push_back(std::string("Log_10(Pose error)"));
//    poserr_labels.push_back(std::string("Pose 1"));
//    poserr_labels.push_back(std::string("Pose 2"));
//    poserr_labels.push_back(std::string("Pose 3"));
//    poserr_labels.push_back(std::string("Pose 4"));
//    poserr_labels.push_back(std::string("Pose 5"));
//    poserr_labels.push_back(std::string("Pose 6"));
//    poserr_labels.push_back(std::string("Pose 7"));
//    poserr_labels.push_back(std::string("Pose 8"));
//    log_2.SetLabels(poserr_labels);

    // OpenGL 'view' of data. We might have many views of the same data.
    if (data->posenets)
    {
        plotter2 = new pangolin::Plotter(&log_2,0.0f,200.0f,-50.0f,1500.0f,50.0f,1.0f);
        //plotter2 = new pangolin::Plotter(&log_2,0.0f,500.0f,0.0f,400.0f,50.0f,10.0f*M_E);
        plotter2->SetBounds(0.0f, 1.0f, 0.5f, 1.0f);
        //plotter2.Track("$i");

        // Add series plots
        plotter2->ClearSeries();
        plotter2->AddSeries("$i", "$0", pangolin::DrawingMode::DrawingModeLine,
                            pangolin::Colour(colors[0].x, colors[0].y, colors[0].z), "Pose-1");
        plotter2->AddSeries("$i", "$1", pangolin::DrawingMode::DrawingModeLine,
                            pangolin::Colour(colors[1].x, colors[1].y, colors[1].z), "Pose-2");
        plotter2->AddSeries("$i", "$2", pangolin::DrawingMode::DrawingModeLine,
                            pangolin::Colour(colors[2].x, colors[2].y, colors[2].z), "Pose-3");
        plotter2->AddSeries("$i", "$3", pangolin::DrawingMode::DrawingModeLine,
                            pangolin::Colour(colors[3].x, colors[3].y, colors[3].z), "Pose-4");
        plotter2->AddSeries("$i", "$4", pangolin::DrawingMode::DrawingModeLine,
                            pangolin::Colour(colors[4].x, colors[4].y, colors[4].z), "Pose-5");
        plotter2->AddSeries("$i", "$5", pangolin::DrawingMode::DrawingModeLine,
                            pangolin::Colour(colors[5].x, colors[5].y, colors[5].z), "Pose-6");
        plotter2->AddSeries("$i", "$6", pangolin::DrawingMode::DrawingModeLine,
                            pangolin::Colour(colors[6].x, colors[6].y, colors[6].z), "Pose-7");
        plotter2->AddSeries("$i", "$7", pangolin::DrawingMode::DrawingModeLine,
                            pangolin::Colour(colors[7].x, colors[7].y, colors[7].z), "Pose-8");
    }

//    // Add some sample annotations to the plot
//    plotter2.AddMarker(pangolin::Marker::Horizontal, -1000, pangolin::Marker::LessThan, pangolin::Colour::White().WithAlpha(1.0f) );

    // =====
    // Create a display renderer
    pangolin::OpenGlRenderState camState(glK_pangolin);
    pangolin::OpenGlRenderState camStatePose(glK_pangolin);
    pangolin::View & rgbDisp = pangolin::Display("rgb").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & pcDisp  = pangolin::Display("pointcloud").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & maskDisp = pangolin::Display("mask").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & poseDisp = pangolin::Display("pose").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camStatePose));
    pangolin::View & rgbDispT = pangolin::Display("rgbT").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & pcDispT  = pangolin::Display("pointcloudT").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & maskDispT = pangolin::Display("maskT").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & poseDispT = pangolin::Display("poseT").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camStatePose));
    pangolin::View disp3d, disp3dT, camDisp;
    pangolin::View plotDisp;
    if (data->posenets)
    {
        disp3d = pangolin::Display("3d")
                .SetBounds(pangolin::Attach::Pix(360), pangolin::Attach::Pix(600), 0.0, 1.0)
                .SetLayout(pangolin::LayoutEqual)
                .AddDisplay(rgbDisp)
                .AddDisplay(pcDisp)
                .AddDisplay(poseDisp)
                .AddDisplay(maskDisp);
        disp3dT = pangolin::Display("3dT")
                .SetBounds(pangolin::Attach::Pix(120), pangolin::Attach::Pix(360), 0.0, 1.0)
                .SetLayout(pangolin::LayoutEqual)
                .AddDisplay(rgbDispT)
                .AddDisplay(pcDispT)
                .AddDisplay(poseDispT)
                .AddDisplay(maskDispT);
        plotDisp = pangolin::Display("plot")
                 .SetBounds(pangolin::Attach::Pix(600), pangolin::Attach::Pix(840), 0.0, 1.0)
                 .AddDisplay(*plotter1)
                 .AddDisplay(*plotter2); // Both jt angle & pose errors
    }
    else
    {
        disp3d = pangolin::Display("3d")
                .SetBounds(pangolin::Attach::Pix(360), pangolin::Attach::Pix(600), 0.0, 1.0)
                .SetLayout(pangolin::LayoutEqual)
                .AddDisplay(rgbDisp)
                .AddDisplay(pcDisp)
                .AddDisplay(maskDisp);
        disp3dT = pangolin::Display("3dT")
                .SetBounds(pangolin::Attach::Pix(120), pangolin::Attach::Pix(360), 0.0, 1.0)
                .SetLayout(pangolin::LayoutEqual)
                .AddDisplay(rgbDispT)
                .AddDisplay(pcDispT)
                .AddDisplay(maskDispT);
        plotDisp = pangolin::Display("plot")
                 .SetBounds(pangolin::Attach::Pix(600), pangolin::Attach::Pix(840), 0.0, 1.0)
                 .AddDisplay(*plotter1); // only jt angle errors
    }

    // Setup overall display
    pangolin::View &allDisp = pangolin::Display("multi")
            .SetBounds(0.0, 1.0f, pangolin::Attach::Pix(panelWidth), 1.0)
            .AddDisplay(plotDisp)
            .AddDisplay(disp3d)
            .AddDisplay(disp3dT);

    /// ===== Pangolin options

    // Options for color map
    pangolin::Var<bool> printModelView("se3gui.showModelView",false,false,true);
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

//    // Set cam state for pose
    Eigen::Matrix4f modelViewPose;
//    modelViewPose << -0.0309999, 0.999487, -0.00799519,	0.18, //0.117511,
//                      0.24277,	-0.000230283, -0.970084, 0.3, //0.193581,
//                     -0.969588,	-0.0320134,	-0.242639,	2, //0.29869,
//                      0,	0,	0,	1;
//    modelViewPose << -0.831689,	-0.154571,	0.533293,	-0.148585,
//                 0.0857293,	0.913204,	0.398383,	0.0341757,
//                 -0.548584,	0.377049,	-0.746251,	0.690717,
//                 0,	0,	0,	1;
//    modelViewPose << -0.902913,	-0.100653,	0.417871,	-0.0470576,
//                     0.181079,	0.792633,	0.582188,	0.00421101,
//                     -0.389818,	0.601333,	-0.697453,	0.4468,
//                     0,	0,	0,	1	;
    modelViewPose << -0.533587,	-0.8101,	-0.242944,	0.116422,
                     -0.641272,	0.200247,	0.740723,	0.184668,
                     -0.551411,	0.551033,	-0.626345,	1.2363,
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
//    Eigen::Matrix4f modelViewPoseInv = modelViewPose.inverse();

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
    datac = boost::shared_ptr<PySimData>(new PySimData(data->imgHeight, data->imgWidth,
                                                         data->imgScale, data->nSE3,
                                                         data->fx, data->fy,
                                                         data->cx, data->cy, data->savedir, data->posenets));

    // Run till asked to terminate
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

        //////////////////// ======= DRAW CURRENT POINTS ========= ///////////////////////

        //// DISPLAY PT CLOUDS /////

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        pcDisp.ActivateScissorAndClear(camState);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        //glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        //glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor4ub(0xff,0xff,0xff,0xff);

        // Apply inverse of modelview to get pts in model frame
        glMultMatrixf(modelViewInv.data());

        // Params
        int npts = datac->imgHeight * datac->imgWidth;

        // Compute normals
        compute_normals(datac->curr_cloud, datac->currnorm_cloud, datac->imgWidth, datac->imgHeight);

        glPointSize(3.0);
        glBegin(GL_POINTS);
        for (int r = 0; r < datac->imgHeight; r++) // copy over from the flow matrix
        {
           for (int c = 0; c < datac->imgWidth; c++)
           {
               // Get pt index
               int id = r * datac->imgWidth + c;

               // ==== Current (blue)
               std::vector<float> colorc = get_color_based_on_depth(datac->curr_cloud[id + 2*npts]);
               glColor3f(colorc[0],colorc[1],colorc[2]); // flip R & G so that right arm is red & left is green

//               // Color based on 3D point
//               float3 ptc = make_float3(datac->curr_cloud[id + 0*npts],
//                                       datac->curr_cloud[id + 1*npts],
//                                       datac->curr_cloud[id + 2*npts]);
//               float3 colorc = get_color(ptc, cube_min, cube_max);
//               glColor3f(colorc.x,colorc.y,colorc.z); // flip R & G so that right arm is red & left is green

//               // Render pt and normals
//               glNormal3f(datac->currnorm_cloud[id + 0*npts],
//                          datac->currnorm_cloud[id + 1*npts],
//                          datac->currnorm_cloud[id + 2*npts]);
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
        //glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        //glDisable(GL_LIGHTING);

        //////////////////// ======= DRAW TARGET POINTS ========= ///////////////////////

        //// DISPLAY PT CLOUDS /////

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        pcDispT.ActivateScissorAndClear(camState);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        //glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        //glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor4ub(0xff,0xff,0xff,0xff);

        // Apply inverse of modelview to get pts in model frame
        glMultMatrixf(modelViewInv.data());

        glPointSize(3.0);
        glBegin(GL_POINTS);
        for (int r = 0; r < datac->imgHeight; r++) // copy over from the flow matrix
        {
           for (int c = 0; c < datac->imgWidth; c++)
           {
               // Get pt index
               int id = r * datac->imgWidth + c;

               // ==== Target (green)
               // Color based on 3D point
//               float3 ptf = make_float3(datac->final_cloud[id + 0*npts],
//                                        datac->final_cloud[id + 1*npts],
//                                        datac->final_cloud[id + 2*npts]);
//               float3 colorf = get_color(ptf, cube_min, cube_max);
               std::vector<float> colorf = get_color_based_on_depth(datac->final_cloud[id + 2*npts]);
               glColor3f(colorf[0],colorf[1],colorf[2]); // flip R & G so that right arm is red & left is green

//               // Render pt and normals
//               glNormal3f(datac->finalnorm_cloud[id + 0*npts],
//                          datac->finalnorm_cloud[id + 1*npts],
//                          datac->finalnorm_cloud[id + 2*npts]);
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
       // glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        //glDisable(GL_LIGHTING);

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

        //////////////////// ======= SHOW MASKED CURRENT PT CLOUD ========= ///////////////////////

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        maskDisp.ActivateScissorAndClear(camState);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        //glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        //glEnable(GL_LIGHTING);
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

//               // Render pt and normals
//               if (useNormalsForMaskColor)
//               {
//                   glNormal3f(datac->currnorm_cloud[id + 0*npts],
//                              datac->currnorm_cloud[id + 1*npts],
//                              datac->currnorm_cloud[id + 2*npts]);
//               }

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
        //glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        //glDisable(GL_LIGHTING);

        //////////////////// ======= SHOW MASKED TARGET PT CLOUD ========= ///////////////////////

        // Clear the display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        maskDispT.ActivateScissorAndClear(camState);

        // Enable flags
        glEnable(GL_DEPTH_TEST);
        //glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        //glEnable(GL_LIGHTING);
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
                   if (datac->final_masks[id + k*npts] > maxval)
                   {
                       maxval = datac->final_masks[id + k*npts];
                       maxid = k;
                   }
               }
               float3 color = colors[maxid];
               glColor3f(color.x,color.y,color.z);

//               // Render pt and normals
//               if (useNormalsForMaskColor)
//               {
//                   glNormal3f(datac->finalnorm_cloud[id + 0*npts],
//                              datac->finalnorm_cloud[id + 1*npts],
//                              datac->finalnorm_cloud[id + 2*npts]);
//               }

               // Plot point
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
        //glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        //glDisable(GL_LIGHTING);


        //////////////////// ======= SHOW Frames ========= ///////////////////////

        if (data->posenets)
        {
            ////// ======= SHOW Init Frames ========= ///////

            if (showAllFrames)
                for(int i = 0; i < datac->nSE3; i++) *frameStatus[i] = true; // Reset options

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
            for(int i = 0; i < datac->nSE3; i++)
            {
                // Display or not
                if(*frameStatus[i])
                {
                    // Create the different SE3s
                    //dart::SE3 init = createSE3FromRt(&datac->init_poses[i*12]);
                    dart::SE3 tar  = createSE3FromRt(&datac->final_poses[i*12]);
                    dart::SE3 curr = createSE3FromRt(&datac->curr_poses[i*12]);
                    //if (showInitPoses) drawFrame(init, 0.5, frameLength, lineWidth);
                    if (showTarPoses)  drawDashedFrame(tar,  1.0, frameLength, lineWidth, colors[i]);
                    if (showCurrPoses) drawFrame(curr, 1.0, frameLength, lineWidth, colors[i]);
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

            //if (pangolin::Pushed(printModelView))
//            cout << camStatePose.GetModelViewMatrix() << endl;

            ////// ======= SHOW Final Frames ========= ///////

            // Clear the display
            glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
            poseDispT.ActivateScissorAndClear(camStatePose);

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
            for(int i = 0; i < datac->nSE3; i++)
            {
                // Display or not
                if(*frameStatus[i])
                {
                    // Create the different SE3s
                    dart::SE3 tar  = createSE3FromRt(&datac->final_poses[i*12]);
                    if (showTarPoses) drawDashedFrame(tar,  1.0, frameLength, lineWidth, colors[i]);
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
        }

        //////////////////// ======= Plot errors ========= ///////////////////////

        // Get current counter
        boost::mutex::scoped_lock update_lock1(data->dataMutex);
        int ctr = data->deg_errors.size();
        if (ctr != prevctr && ctr > 0)
        {
            for (int i = prevctr; i < ctr; i++)
            {
                // Joint angle error (show all latest values since last time)
                std::vector<float> d = data->deg_errors[i];
                log_1.Log(d[0], d[1], d[2], d[3], d[4], d[5], d[6]);

                // Pose error (show all latest values since last time)
                if (data->posenets)
                {
                    std::vector<float> e = data->pose_errors_indiv[i];
                    log_2.Log(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]);
                    //log_2.Log(log10(e[0]), log10(e[1]), log10(e[2]), log10(e[3]), log10(e[4]), log10(e[5]), log10(e[6]), log10(e[7]));
                }
            }

            // Prev ctr update
            prevctr = ctr;
        }
        update_lock1.unlock();

        //////////////////// ======= Render the arms (RGB) ========= ///////////////////////

        float4 lightPosition = make_float4(normalize(make_float3(-0.4405,-0.5357,-0.619)),0);
        GLfloat light_ambient[] = { 0.3, 0.3, 0.3, 1.0 };
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
        glLightfv(GL_LIGHT0, GL_POSITION, (float*)&lightPosition);

        ////////// == Current config == ///////////
        // Show arm @ current config in first display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        rgbDisp.ActivateScissorAndClear(camState);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);
        glPushMatrix();
        glColor4ub(0xff,0xff,0xff,0xff);
        glEnable(GL_COLOR_MATERIAL);

        // Update pose and render
        tracker.updatePose(baxterID_in);
        tracker.getModel(baxterID_in).render();

        glPopMatrix();
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        glDisable(GL_LIGHTING);
        glColor4ub(255,255,255,255);

        ////////// == Target config == ///////////
        // Show arm @ current config in first display
        glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
        rgbDispT.ActivateScissorAndClear(camState);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);
        glPushMatrix();
        glColor4ub(0xff,0xff,0xff,0xff);
        glEnable(GL_COLOR_MATERIAL);

        // Update pose and render
        tracker.updatePose(baxterID_tar);
        tracker.getModel(baxterID_tar).render();

        glPopMatrix();
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHT0);
        glDisable(GL_NORMALIZE);
        glDisable(GL_LIGHTING);
        glColor4ub(255,255,255,255);

//        //////////////////// ==== Text ==== ////

//        allDisp.Activate();
//        glColor3f(1,1,1); // Whitei
//        std::string text1 = "Joint angle errors (vs) Iterations";
//        std::string text2 = "Pose error (vs) Iterations";
//        pangolin::GlText txt1 = glFont.Text(text1.c_str());
//        txt1.DrawWindow(300, 240*3-30, 0.1);
//        pangolin::GlText txt2 = glFont.Text(text2.c_str());
//        txt2.DrawWindow(300+640, 240*3-30, 0.1);

//        //////////////////// ==== SHOW RGB ==== /////////////////////

//        if (!datac->use_simulator)
//        {
//            // Display RGB image
//            glColor3f(1,1,1);
//            rgbDisp.ActivateScissorAndClear();
//            texRGB.Upload(datac->curr_rgb,GL_BGR,GL_UNSIGNED_BYTE);
//            texRGB.RenderToViewportFlipY();

//            rgbDispT.ActivateScissorAndClear();
//            texRGBTar.Upload(datac->final_rgb,GL_BGR,GL_UNSIGNED_BYTE);
//            texRGBTar.RenderToViewportFlipY();
//        }

        //////////////////// ======= Finish ========= ///////////////////////

        // Save frames to disk now
        if (save_frames && !frame_saved)
        {
            std::string filename = save_frames_dir + "/render" + std::to_string(framectr);
            allDisp.SaveOnRender(filename);
            framectr++; // Increment frame counter
            frame_saved = true; // Since we have already saved the image for this particular mask update!
        }

        /// Finish frame
        pangolin::FinishFrame();
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
SimCtrlViz::SimCtrlViz(int imgHeight, int imgWidth, float imgScale, int nSE3,
                       float fx, float fy, float cx, float cy,
                       const std::string savedir, const int posenets)
{
    printf("==> [SIM-CTRL VIZ] Initializing data for visualizer \n");
    data = boost::shared_ptr<PySimData>(new PySimData(imgHeight, imgWidth, imgScale, nSE3,
                                                        fx, fy, cx, cy, savedir, posenets));

    /// ===== PANGOLIN viewer
    printf("==> [SIM-CTRL VIZ] Starting pangolin in a separate thread \n");
    pangolin_gui_thread.reset(new boost::thread(run_pangolin, data));

    while(!data->init_done) { usleep(100); }
    printf("==> [SIM-CTRL VIZ] Finished initializing pangolin visualizer \n");
    return;
}

//////////////////////
///
/// \brief terminate_viz - Kill the visualizer
///
SimCtrlViz::~SimCtrlViz()
{
    // Terminate visualizer (data will be deleted automatically as it is a shared ptr)
    terminate_pangolin = true;
    pangolin_gui_thread->join(); // Wait for thread to join
    printf("==> [SIM-CTRL VIZ] Terminated pangolin visualizer");
}

////////////////////////////////////////////////////////////////////////////////////////
/// === HELPER FUNCTIONS FOR RENDERING DATA

void SimCtrlViz::render_arm(const float *config,
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

void SimCtrlViz::render_pose(const float *config,
                             float *poses,
                             int *nposes)
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

    // Update curr jt angles
    memcpy(data->curr_jts, config, 7*sizeof(float));

    // Copy to output
    nposes[0] = data->mesh_transforms.size(); // Save num poses
    for(int k = 0; k < data->mesh_transforms.size(); k++)
    {
        // Get current pose
        float *currpose = &(poses[k*12]); // 12 values per pose
        const dart::SE3 se3 = data->mesh_transforms[k];

        // === Save the SE3
        // Save r0
        currpose[k*12 + 0] = se3.r0.w;
        currpose[k*12 + 1] = se3.r0.x;
        currpose[k*12 + 2] = se3.r0.y;
        currpose[k*12 + 3] = se3.r0.z;

        // Save r1
        currpose[k*12 + 4] = se3.r1.w;
        currpose[k*12 + 5] = se3.r1.x;
        currpose[k*12 + 6] = se3.r1.y;
        currpose[k*12 + 7] = se3.r1.z;

        // Save r2
        currpose[k*12 + 8]  = se3.r2.w;
        currpose[k*12 + 9]  = se3.r2.x;
        currpose[k*12 + 10] = se3.r2.y;
        currpose[k*12 + 11] = se3.r2.z;
    }

    // Unlock
    update_lock.unlock();
}

////////////////////////////////////////////////////////////////////////////////////////
/// === SHOW PREDICTIONS VS GRADIENTS

//////////////////////////////
///
/// \brief Updated predicted masks & Curr poses for SE3-Control
///
void SimCtrlViz::update_curr(const float *curr_angles,
                             const float *curr_ptcloud,
                             const float *curr_poses,
                             const float *curr_masks,
                             const float curr_pose_error,
                             const float* curr_pose_errors_indiv,
                             const float *curr_deg_errors,
                             int save_frame)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy vars
    memcpy(data->curr_jts,   curr_angles,  7 * sizeof(float));
    memcpy(data->curr_cloud, curr_ptcloud, data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->curr_masks, curr_masks,   data->imgHeight * data->imgWidth * data->nSE3 * sizeof(float));
    memcpy(data->curr_poses, curr_poses,   data->nSE3 * 12 * sizeof(float));

    // Append to the errors
    data->pose_errors.push_back(curr_pose_error);
    data->pose_errors_indiv.push_back(std::vector<float>(curr_pose_errors_indiv, curr_pose_errors_indiv+data->nSE3));
    data->deg_errors.push_back(std::vector<float>(curr_deg_errors, curr_deg_errors+7));

    // Frame saved set to false
    frame_saved = false;

    // Unlock mutex so that display happens
    update_lock.unlock();

    // Wait till frame is saved before returning
    if (save_frame)
    {
        while(!frame_saved)
        {
            usleep(1000);
        }
    }
}

//////////////////////////////
///
/// \brief Updated predicted masks & Curr poses for SE3-Control
///
void SimCtrlViz::update_init(const float *start_angles, const float *start_ptcloud,
                             const float *start_poses, const float *start_masks,
                             const float *goal_angles, const float *goal_ptcloud,
                             const float *goal_poses, const float *goal_masks,
                             const float start_pose_error,  const float *start_pose_errors_indiv,
                             const float *start_deg_errors)
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

    // Find max deg errors
    float max_deg_error = -HUGE_VALF, min_deg_error = HUGE_VALF;
    for (int k = 0; k < 7; k++)
    {
        max_deg_error = (start_deg_errors[k] > max_deg_error) ? start_deg_errors[k] : max_deg_error;
        min_deg_error = (start_deg_errors[k] < min_deg_error) ? start_deg_errors[k] : min_deg_error;
    }

    // Reset the loggers, dynamically change the max values of the loggers here
    this->reset();
    plotter1->SetDefaultView(pangolin::XYRangef(pangolin::Rangef(0.0f, 200.0f), pangolin::Rangef(min_deg_error, max_deg_error)));
    if (data->posenets)
        plotter2->SetDefaultView(pangolin::XYRangef(pangolin::Rangef(0.0f, 200.0f), pangolin::Rangef(0.0f, start_pose_error)));

    // Append to the errors
    data->pose_errors.push_back(start_pose_error);
    data->pose_errors_indiv.push_back(std::vector<float>(start_pose_errors_indiv, start_pose_errors_indiv+data->nSE3));
    data->deg_errors.push_back(std::vector<float>(start_deg_errors, start_deg_errors+7));

    // Compute normals for init & final clouds
    compute_normals(data->init_cloud, data->initnorm_cloud, data->imgWidth, data->imgHeight);
    compute_normals(data->final_cloud, data->finalnorm_cloud, data->imgWidth, data->imgHeight);

    // Reset frame counter
    framectr = 0;

    // Unlock mutex so that display happens
    update_lock.unlock();
}

//////////////////////////////
///
/// \brief Updated predicted masks & Curr poses for SE3-Control
///
void SimCtrlViz::reset()
{
    data->deg_errors.clear();
    data->pose_errors.clear();
    data->pose_errors_indiv.clear();
    log_1.Clear();
    log_2.Clear();
    prevctr = 0;
}

//////////////////////////////
///
/// \brief Start saving rendered frames to disk
///
void SimCtrlViz::start_saving_frames(const std::string framesavedir)
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
void SimCtrlViz::stop_saving_frames()
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Update flag
    save_frames = false;

    // Unlock mutex so that display happens
    update_lock.unlock();
}

//////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief Compute GT data associations between two configurations and render them
///
void SimCtrlViz::compute_gt_da(const float *input_jts,
                                const float *target_jts,
                                const int winsize,
                                const float thresh,
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
            numass++; // Successful association
        }
    }
    printf("Total: %d, Non-BG: %d, Num Ass: %d, Flow > 0.005m: %d/%d = %f, Err > 0.005m: %d. Max Err: %f \n", npts, nonbg,
           numass, bigflow_ass, bigflow_gt, bigflow_ass * (1.0/bigflow_gt), bigerr, max_flowerr);

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
}

void SimCtrlViz::initialize_problem(const float *start_jts, const float *goal_jts,
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

    // Copy to output
    memcpy(start_pts, data->init_cloud, 3*npts*sizeof(float)); // Copy start pt cloud
    memcpy(da_goal_pts, data->final_cloud, 3*npts*sizeof(float)); // Copy DA-Final pt cloud

    /// Unlock
    update_lock.unlock();
}
