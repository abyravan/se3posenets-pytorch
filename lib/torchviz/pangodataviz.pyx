# --------------------------------------------------------
# SE3-POSE-NETS
# Copyright (c) 2017
# Licensed under The MIT License [see LICENSE for details]
# Written by Arunkumar Byravan
# --------------------------------------------------------

from libcpp.string cimport string
import numpy as np
cimport numpy as np
import ctypes

##################### Pose visualizer
cdef extern from "pangodataviz.hpp":
    cdef cppclass PangolinDataViz:
        PangolinDataViz(string data_path, int step_len, int seq_len, int nSE3, int ht, int wd,
                        int nstate, int nctrl, int ntracker, float fx, float fy, float cx, float cy,
                        const float *initconfig);

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

cdef class PyPangolinDataViz:
    cdef PangolinDataViz *pangodataviz     # hold a C++ instance which we're wrapping

    def __cinit__(self, string data_path, int step_len, int seq_len, int nSE3, int ht, int wd,
                        int nstate, int nctrl, int ntracker, float  fx, float fy, float cx, float cy,
                        np.ndarray[np.float32_t, ndim=1] initconfig):
        self.pangodataviz = new PangolinDataViz(data_path, step_len, seq_len, nSE3, ht, wd,
                                                nstate, nctrl, ntracker, fx, fy, cx, cy, &initconfig[0])

    def __dealloc__(self):
        del self.pangodataviz

    def update_viz(self, np.ndarray[np.float32_t, ndim=4] ptclouds,
                         np.ndarray[np.float32_t, ndim=4] fwdflows,
                         np.ndarray[np.float32_t, ndim=4] bwdflows,
                         np.ndarray[np.uint8_t, ndim=4] fwdvis,
                         np.ndarray[np.uint8_t, ndim=4] bwdvis,
                         np.ndarray[np.uint8_t, ndim=4] labels,
                         np.ndarray[np.uint8_t, ndim=4] rgbs,
                         np.ndarray[np.uint8_t, ndim=4] masks,
                         np.ndarray[np.float32_t, ndim=4] poses,
                         np.ndarray[np.float32_t, ndim=4] camposes,
                         np.ndarray[np.float32_t, ndim=2] modelview,
                         np.ndarray[np.float32_t, ndim=2] actconfigs,
                         np.ndarray[np.float32_t, ndim=2] actvels,
                         np.ndarray[np.float32_t, ndim=2] comconfigs,
                         np.ndarray[np.float32_t, ndim=2] comvels,
                         np.ndarray[np.float32_t, ndim=2] trackerconfigs,
                         np.ndarray[np.float32_t, ndim=2] controls,
                         np.ndarray[np.float32_t, ndim=1] id):
        # Run CPP code
        self.pangodataviz.update_viz(&ptclouds[0,0,0,0],
                                     &fwdflows[0,0,0,0],
                                     &bwdflows[0,0,0,0],
                                     &fwdvis[0,0,0,0],
                                     &bwdvis[0,0,0,0],
                                     &labels[0,0,0,0],
                                     &rgbs[0,0,0,0],
                                     &masks[0,0,0,0],
                                     &poses[0,0,0,0],
                                     &camposes[0,0,0,0],
                                     &modelview[0,0],
                                     &actconfigs[0,0],
                                     &actvels[0,0],
                                     &comconfigs[0,0],
                                     &comvels[0,0],
                                     &trackerconfigs[0,0],
                                     &controls[0,0],
                                     &id[0])
