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

cdef extern from "compviz.hpp":
    cdef cppclass CompViz:
        CompViz(int imgHeight, int imgWidth, float imgScale,
                float fx, float fy, float cx, float cy,
                int showposepts);

        void update(const float *inp_cloud,
                             const float *gt_cloud,
                             const float *se3pose_cloud,
                             const float *se3posepts_cloud,
                             const float *se3_cloud,
                             const float *flow_cloud,
                             const float *gt_mask,
                             const float *se3pose_mask,
                             const float *se3posepts_mask,
                             const float *se3_mask,
                             int save_frame);

        void start_saving_frames(const string framesavedir);

        void stop_saving_frames();

def assert_contiguous(arrays):
    for arr in arrays:
        assert arr.flags['C_CONTIGUOUS']

cdef class PyCompViz:
    cdef CompViz *compviz     # hold a C++ instance which we're wrapping
    cdef int img_ht
    cdef int img_wd

    def __cinit__(self, int img_ht, int img_wd, float img_scale,
                    float fx, float fy, float cx, float cy, int showposepts):
        self.img_ht   = img_ht
        self.img_wd   = img_wd
        self.compviz = new CompViz(img_ht, img_wd, img_scale,
                                   fx, fy, cx, cy, showposepts)

    def __dealloc__(self):
        del self.compviz

    def update(self, np.ndarray[np.float32_t, ndim=3] inp_ptcloud,
                     np.ndarray[np.float32_t, ndim=3] gt_ptcloud,
                     np.ndarray[np.float32_t, ndim=3] se3pose_ptcloud,
                     np.ndarray[np.float32_t, ndim=3] se3posepts_ptcloud,
                     np.ndarray[np.float32_t, ndim=3] se3_ptcloud,
                     np.ndarray[np.float32_t, ndim=3] flow_ptcloud,
                     np.ndarray[np.float32_t, ndim=3] gt_mask,
                     np.ndarray[np.float32_t, ndim=3] se3pose_mask,
                     np.ndarray[np.float32_t, ndim=3] se3posepts_mask,
                     np.ndarray[np.float32_t, ndim=3] se3_mask,
                     int save_frame):
        # Run CPP code
        self.compviz.update(&inp_ptcloud[0,0,0],
                            &gt_ptcloud[0,0,0],
                            &se3pose_ptcloud[0,0,0],
                            &se3posepts_ptcloud[0,0,0],
                            &se3_ptcloud[0,0,0],
                            &flow_ptcloud[0,0,0],
                            &gt_mask[0,0,0],
                            &se3pose_mask[0,0,0],
                            &se3posepts_mask[0,0,0],
                            &se3_mask[0,0,0],
                            save_frame)

    def start_saving_frames(self, string framesavedir):
        # Run CPP code
        self.compviz.start_saving_frames(framesavedir)

    def stop_saving_frames(self):
        # Run CPP code
        self.compviz.stop_saving_frames()
