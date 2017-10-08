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
cdef extern from "pangoposeviz.hpp":
    cdef cppclass PangolinPoseViz:
        PangolinPoseViz();

        void update_viz(const float *gtpose, const float *predpose, const float *predmask,
                        float *config, float *ptcloud);

cdef class PyPangolinPoseViz:
    cdef PangolinPoseViz *pangoposeviz     # hold a C++ instance which we're wrapping

    def __cinit__(self):
        self.pangoposeviz = new PangolinPoseViz()

    def __dealloc__(self):
        del self.pangoposeviz

    def update_viz(self, np.ndarray[np.float32_t, ndim=3] poses,
                         np.ndarray[np.float32_t, ndim=3] predposes,
                         np.ndarray[np.float32_t, ndim=3] predmasks,
                         np.ndarray[np.float32_t, ndim=1] config,
                         np.ndarray[np.float32_t, ndim=3] ptcloud):
        # Run CPP code
        self.pangoposeviz.update_viz(&poses[0,0,0],
                                     &predposes[0,0,0],
                                     &predmasks[0,0,0],
                                     &config[0],
                                     &ptcloud[0,0,0])
