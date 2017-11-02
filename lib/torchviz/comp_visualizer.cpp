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
#include "comp_visualizer.hpp"

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


// ------------------------------------------
void drawColoredPoints(const float * cloud, const int imgHeight, const int imgWidth,
                       const pangolin::View &camDisp, const pangolin::OpenGlRenderState &camState,
                       const Eigen::Matrix4f modelViewInv)
{
    //// DISPLAY PT CLOUDS /////

    // Clear the display
    glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
    camDisp.ActivateScissorAndClear(camState);

    // Enable flags
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    glColor4ub(0xff,0xff,0xff,0xff);

    // Apply inverse of modelview to get pts in model frame
    glMultMatrixf(modelViewInv.data());

    // Params
    int npts = imgHeight * imgWidth;

    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (int r = 0; r < imgHeight; r++) // copy over from the flow matrix
    {
       for (int c = 0; c < imgWidth; c++)
       {
           // Get pt index
           int id = r * imgWidth + c;

           // Color based on depth
           std::vector<float> colorc = get_color_based_on_depth(cloud[id + 2*npts]);
           glColor3f(colorc[0],colorc[1],colorc[2]); // flip R & G so that right arm is red & left is green
           glVertex3f(cloud[id + 0*npts],
                      cloud[id + 1*npts],
                      cloud[id + 2*npts]);
       }
    }
    glEnd();

    glPopMatrix(); // Remove inverse transform
    glColor4ub(255,255,255,255);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_NORMALIZE);
}

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

// ------------------------------------------
void drawMaskColoredPoints(const float * cloud, const float *mask, const int nSE3, const int imgHeight, const int imgWidth,
                       const pangolin::View &camDisp, const pangolin::OpenGlRenderState &camState,
                       const Eigen::Matrix4f modelViewInv)
{
    //// DISPLAY PT CLOUDS /////

    // Clear the display
    glClearColor(0.0,0.0,0.0,1.0); // This sets the value that all points that do not have a geometry render to (it was set to 1m before which messed up my rendering)
    camDisp.ActivateScissorAndClear(camState);

    // Enable flags
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    glColor4ub(0xff,0xff,0xff,0xff);

    // Apply inverse of modelview to get pts in model frame
    glMultMatrixf(modelViewInv.data());

    // Params
    int npts = imgHeight * imgWidth;

    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (int r = 0; r < imgHeight; r++) // copy over from the flow matrix
    {
       for (int c = 0; c < imgWidth; c++)
       {
           // Get pt index
           int id = r * imgWidth + c;

           // Color based on mask
           int maxid = -1; float maxval = -HUGE_VAL;
           for (int k = 0; k < nSE3; k++)
           {
               if (mask[id + k*npts] > maxval)
               {
                   maxval = mask[id + k*npts];
                   maxid = k;
               }
           }
           float3 color = colors[maxid];
           glColor3f(color.x,color.y,color.z);
           glVertex3f(cloud[id + 0*npts],
                      cloud[id + 1*npts],
                      cloud[id + 2*npts]);
       }
    }
    glEnd();

    glPopMatrix(); // Remove inverse transform
    glColor4ub(255,255,255,255);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_NORMALIZE);
}

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
bool frame_saved = true;

// Data copy
boost::shared_ptr<PyCompData> datac;
int prevctr = 0;

void run_pangolin(const boost::shared_ptr<PyCompData> data)
{
    /// ===== Set up a DART tracker with the baxter model

    // Setup OpenGL/CUDA/Pangolin stuff - Has to happen before DART tracker initialization
    //cudaGLSetGLDevice(0);
    //cudaDeviceReset();
    int ncols = (data->showposepts) ? 6 : 5;
    const float totalwidth = 320*ncols;
    const float totalheight = 240*2;
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

    // =====
    // Create a display renderer
    pangolin::OpenGlRenderState camState(glK_pangolin);
    pangolin::View & inpDisp     = pangolin::Display("inp").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & gtDisp      = pangolin::Display("gt").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & se3poseDisp = pangolin::Display("se3pose").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & se3Disp     = pangolin::Display("se3").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & flowDisp    = pangolin::Display("flow").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));

    // Setup overall display
    pangolin::View allDisp;
    pangolin::View se3poseptsDisp;
    if (data->showposepts)
        allDisp = pangolin::Display("multi")
                .SetBounds(pangolin::Attach::Pix(240), pangolin::Attach::Pix(480), 0.0, 1.0)
                .SetLayout(pangolin::LayoutEqual)
                .AddDisplay(inpDisp)
                .AddDisplay(gtDisp)
                .AddDisplay(se3poseDisp)
                .AddDisplay(se3poseptsDisp)
                .AddDisplay(se3Disp)
                .AddDisplay(flowDisp);
    else
    {
        se3poseptsDisp = pangolin::Display("se3posepts").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
        allDisp = pangolin::Display("multi")
                .SetBounds(pangolin::Attach::Pix(240), pangolin::Attach::Pix(480), 0.0, 1.0)
                .SetLayout(pangolin::LayoutEqual)
                .AddDisplay(inpDisp)
                .AddDisplay(gtDisp)
                .AddDisplay(se3poseDisp)
                .AddDisplay(se3Disp)
                .AddDisplay(flowDisp);
    }

    // Masked 3D points
    pangolin::View & inpDispM     = pangolin::Display("inpM").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & gtDispM      = pangolin::Display("gtM").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & se3poseDispM = pangolin::Display("se3poseM").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & se3DispM     = pangolin::Display("se3M").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & flowDispM    = pangolin::Display("flowM").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));

    // Setup overall display
    pangolin::View allDispM;
    pangolin::View se3poseptsDispM;
    if (data->showposepts)
        allDispM = pangolin::Display("multi-mask")
                .SetBounds(pangolin::Attach::Pix(0), pangolin::Attach::Pix(240), 0.0, 1.0)
                .SetLayout(pangolin::LayoutEqual)
                .AddDisplay(inpDispM)
                .AddDisplay(gtDispM)
                .AddDisplay(se3poseDispM)
                .AddDisplay(se3poseptsDispM)
                .AddDisplay(se3DispM)
                .AddDisplay(flowDispM);
    else
    {
        se3poseptsDispM = pangolin::Display("se3poseptsM").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
        allDispM = pangolin::Display("multi-mask")
                .SetBounds(pangolin::Attach::Pix(0), pangolin::Attach::Pix(240), 0.0, 1.0)
                .SetLayout(pangolin::LayoutEqual)
                .AddDisplay(inpDispM)
                .AddDisplay(gtDispM)
                .AddDisplay(se3poseDispM)
                .AddDisplay(se3DispM)
                .AddDisplay(flowDispM);
    }

    // Setup overall display
    pangolin::View &fullDisp = pangolin::Display("full")
            .SetBounds(0.0, 1.0f, 0.0, 1.0f)
            .AddDisplay(allDisp)
            .AddDisplay(allDispM);

    /// ====== Setup model view matrix from disk
    // Model view matrix
    Eigen::Matrix4f modelView = Eigen::Matrix4f::Identity();
    Eigen::read_binary("/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/cameramodelview.dat", modelView);

    // Set cam state
    camState.SetModelViewMatrix(modelView);
    inpDisp.SetHandler(new pangolin::Handler3D(camState));
    gtDisp.SetHandler(new pangolin::Handler3D(camState));
    se3poseDisp.SetHandler(new pangolin::Handler3D(camState));
    se3poseptsDisp.SetHandler(new pangolin::Handler3D(camState));
    se3Disp.SetHandler(new pangolin::Handler3D(camState));
    flowDisp.SetHandler(new pangolin::Handler3D(camState));

    // Create a renderer
    pangolin::OpenGlMatrixSpec glK = pangolin::ProjectionMatrixRDF_BottomLeft(glWidth,glHeight,glFLx,glFLy,glPPx,glPPy,0.01,1000);
    l2s::Renderer<TYPELIST2(l2s::IntToType<l2s::RenderVertMapWMeshID>, l2s::IntToType<l2s::RenderDepth>)> renderer(glWidth, glHeight, glK);
    renderer.setModelViewMatrix(modelView);

    // Save it
    Eigen::Matrix4f modelViewInv = modelView.inverse();

    // Enable blending once
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    data->init_done = true;

    // Keep copies of all data
    datac = boost::shared_ptr<PyCompData>(new PyCompData(data->imgHeight, data->imgWidth,
                                                        data->imgScale,
                                                        data->fx, data->fy,
                                                        data->cx, data->cy, data->showposepts));

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

        //////////////////// ======= DISPLAY PT CLOUDS ========= ///////////////////////

        // Inp | GT | SE3-Pose | SE3 | Flow
        drawColoredPoints(datac->inp_cloud, datac->imgHeight, datac->imgWidth,
                          inpDisp, camState, modelViewInv);
        drawColoredPoints(datac->gt_cloud, datac->imgHeight, datac->imgWidth,
                          gtDisp, camState, modelViewInv);
        drawColoredPoints(datac->se3pose_cloud, datac->imgHeight, datac->imgWidth,
                          se3poseDisp, camState, modelViewInv);
        if (datac->showposepts)
            drawColoredPoints(datac->se3posepts_cloud, datac->imgHeight, datac->imgWidth,
                              se3poseptsDisp, camState, modelViewInv);
        drawColoredPoints(datac->se3_cloud, datac->imgHeight, datac->imgWidth,
                          se3Disp, camState, modelViewInv);
        drawColoredPoints(datac->flow_cloud, datac->imgHeight, datac->imgWidth,
                          flowDisp, camState, modelViewInv);

        //////////////////// ======= SHOW MASKED CURRENT PT CLOUD ========= ///////////////////////

        // xxx | GT | SE3-Pose | SE3 | xxx
        drawMaskColoredPoints(datac->gt_cloud, datac->gt_mask, datac->nSE3, datac->imgHeight, datac->imgWidth,
                              gtDispM, camState, modelViewInv);
        drawMaskColoredPoints(datac->se3pose_cloud, datac->se3pose_mask, datac->nSE3, datac->imgHeight, datac->imgWidth,
                              se3poseDispM, camState, modelViewInv);
        if (datac->showposepts)
            drawMaskColoredPoints(datac->se3posepts_cloud, datac->se3posepts_mask, datac->nSE3, datac->imgHeight, datac->imgWidth,
                                  se3poseptsDispM, camState, modelViewInv);
        drawMaskColoredPoints(datac->se3_cloud, datac->se3_mask, datac->nSE3, datac->imgHeight, datac->imgWidth,
                              se3DispM, camState, modelViewInv);

        //////////////////// ======= Finish ========= ///////////////////////

        // Save frames to disk now
        if (save_frames && !frame_saved)
        {
            std::string filename = save_frames_dir + "/render" + std::to_string(framectr);
            fullDisp.SaveOnRender(filename);
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
CompViz::CompViz(int imgHeight, int imgWidth, float imgScale,
                 float fx, float fy, float cx, float cy, int showposepts)
{
    printf("==> [COMP VIZ] Initializing data for visualizer \n");
    data = boost::shared_ptr<PyCompData>(new PyCompData(imgHeight, imgWidth, imgScale,
                                                        fx, fy, cx, cy, showposepts));

    /// ===== PANGOLIN viewer
    printf("==> [COMP VIZ] Starting pangolin in a separate thread \n");
    pangolin_gui_thread.reset(new boost::thread(run_pangolin, data));

    while(!data->init_done) { usleep(100); }
    printf("==> [COMP VIZ] Finished initializing pangolin visualizer \n");
    return;
}

//////////////////////
///
/// \brief terminate_viz - Kill the visualizer
///
CompViz::~CompViz()
{
    // Terminate visualizer (data will be deleted automatically as it is a shared ptr)
    terminate_pangolin = true;
    pangolin_gui_thread->join(); // Wait for thread to join
    printf("==> [COMP VIZ] Terminated pangolin visualizer");
}

//////////////////////////////
///
/// \brief Updated predicted masks & Curr poses for SE3-Control
///
void CompViz::update(const float *inp_cloud,
                     const float *gt_cloud,
                     const float *se3pose_cloud,
                     const float *se3posepts_cloud,
                     const float *se3_cloud,
                     const float *flow_cloud,
                     const float *gt_mask,
                     const float *se3pose_mask,
                     const float *se3posepts_mask,
                     const float *se3_mask,
                     int save_frame)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Memcpy vars
    memcpy(data->inp_cloud,     inp_cloud,     data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->gt_cloud,      gt_cloud,      data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->se3pose_cloud, se3pose_cloud, data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->se3posepts_cloud, se3posepts_cloud, data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->se3_cloud,     se3_cloud,     data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->flow_cloud,    flow_cloud,    data->imgHeight * data->imgWidth * 3 * sizeof(float));
    memcpy(data->gt_mask,       gt_mask,       data->imgHeight * data->imgWidth * data->nSE3 * sizeof(float));
    memcpy(data->se3pose_mask,  se3pose_mask,  data->imgHeight * data->imgWidth * data->nSE3 * sizeof(float));
    memcpy(data->se3posepts_mask,  se3posepts_mask,  data->imgHeight * data->imgWidth * data->nSE3 * sizeof(float));
    memcpy(data->se3_mask,      se3_mask,      data->imgHeight * data->imgWidth * data->nSE3 * sizeof(float));

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
/// \brief Start saving rendered frames to disk
///
void CompViz::start_saving_frames(const std::string framesavedir)
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Update flag & the save directory
    if (save_frames_dir.compare(framesavedir) != 0)
        framectr = 0; // Reset frame counter to zero
    save_frames_dir = framesavedir;
    save_frames = true;

    // Unlock mutex so that display happens
    update_lock.unlock();
}


//////////////////////////////
///
/// \brief Stop saving rendered frames to disk
///
void CompViz::stop_saving_frames()
{
    // Update PCL viewer
    boost::mutex::scoped_lock update_lock(data->dataMutex);

    // Update flag
    save_frames = false;

    // Unlock mutex so that display happens
    update_lock.unlock();
}
