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

cdef extern from "realctrlviz.hpp":
    cdef cppclass RealCtrlViz:
        RealCtrlViz(int imgHeight, int imgWidth, float imgScale, int nSE3,
                    float fx, float fy, float cx, float cy,
                    const string savedir);

        void render_arm(const float *config,
                        float *rendered_ptcloud,
                        float *rendered_labels);

        void update_real_curr(const float *curr_angles, const float *curr_ptcloud,
                              const float *curr_poses, const float *curr_masks,
                              const float curr_pose_error, const float *curr_deg_error);

        void update_real_init(const float *start_angles, const float *start_ptcloud,
                              const float *start_poses, const float *start_masks,
                              const float *goal_angles, const float *goal_ptcloud,
                              const float *goal_poses, const float *goal_masks,
                              const float init_pose_error, const float *init_deg_error);
        void reset();

        void start_saving_frames(const string framesavedir);

        void stop_saving_frames();

def assert_contiguous(arrays):
    for arr in arrays:
        assert arr.flags['C_CONTIGUOUS']

cdef class PyRealCtrlViz:
    cdef RealCtrlViz *realctrlviz     # hold a C++ instance which we're wrapping
    cdef int img_ht
    cdef int img_wd

    def __cinit__(self, int img_ht, int img_wd, float img_scale, int num_se3,
                    float fx, float fy, float cx, float cy,
                    string savedir):
        self.img_ht   = img_ht
        self.img_wd   = img_wd
        self.realctrlviz = new RealCtrlViz(img_ht, img_wd, img_scale, num_se3,
                                       fx, fy, cx, cy, savedir)

    def __dealloc__(self):
        del self.realctrlviz

    def render_arm(self, np.ndarray[np.float32_t, ndim=1] config,
                         np.ndarray[np.float32_t, ndim=3] ptcloud,
                         np.ndarray[np.float32_t, ndim=3] labels):
        # Run CPP code
        self.realctrlviz.render_arm(&config[0],
                                 &ptcloud[0,0,0],
                                 &labels[0,0,0])

    def update_real_curr(self, np.ndarray[np.float32_t, ndim=1] curr_angles,
                               np.ndarray[np.float32_t, ndim=3] curr_ptcloud,
                               np.ndarray[np.float32_t, ndim=3] curr_poses,
                               np.ndarray[np.float32_t, ndim=3] curr_masks,
                               float curr_pose_error,
                               np.ndarray[np.float32_t, ndim=1] curr_deg_error):
        # Run CPP code
        self.realctrlviz.update_real_curr(&curr_angles[0],
                                       &curr_ptcloud[0,0,0],
                                       &curr_poses[0,0,0],
                                       &curr_masks[0,0,0],
                                       curr_pose_error,
                                       &curr_deg_error[0])

    def update_real_init(self, np.ndarray[np.float32_t, ndim=1] start_angles,
                               np.ndarray[np.float32_t, ndim=3] start_ptcloud,
                               np.ndarray[np.float32_t, ndim=3] start_poses,
                               np.ndarray[np.float32_t, ndim=3] start_masks,
                               np.ndarray[np.float32_t, ndim=1] goal_angles,
                               np.ndarray[np.float32_t, ndim=3] goal_ptcloud,
                               np.ndarray[np.float32_t, ndim=3] goal_poses,
                               np.ndarray[np.float32_t, ndim=3] goal_masks,
                               float init_pose_error,
                               np.ndarray[np.float32_t, ndim=1] init_deg_error):
        # Run CPP code
        self.realctrlviz.update_real_init(&start_angles[0],
                                       &start_ptcloud[0,0,0],
                                       &start_poses[0,0,0],
                                       &start_masks[0,0,0],
                                       &goal_angles[0],
                                       &goal_ptcloud[0,0,0],
                                       &goal_poses[0,0,0],
                                       &goal_masks[0,0,0],
                                       init_pose_error,
                                       &init_deg_error[0])
    def reset(self):
        # Run CPP code
        self.realctrlviz.reset()

    def start_saving_frames(self, string framesavedir):
        # Run CPP code
        self.realctrlviz.start_saving_frames(framesavedir)

    def stop_saving_frames(self):
        # Run CPP code
        self.realctrlviz.stop_saving_frames()
