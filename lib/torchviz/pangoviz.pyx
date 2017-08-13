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

cdef extern from "pangoviz.hpp":
    cdef cppclass PangolinViz:
        PangolinViz(int seqLength, int imgHeight, int imgWidth, float imgScale, int nSE3,
                    float fx, float fy, float cx, float cy, float dt, int oldgrippermodel,
                    const string savedir);

        void initialize_problem(const float *start_jts,
                                const float *goal_jts,
                                float *start_pts,
                                float *da_goal_pts);

        void update_viz(const float *inputpts,
                        const float *outputpts_gt,
                        const float *outputpts_pred,
                        const float *jtangles_gt,
                        const float *jtangles_pred);

        void render_arm(const float *config,
                        float *rendered_ptcloud);

        void update_da(const float *init_pts,
                       const float *current_pts,
                       const float *current_da_ids,
                       const float * warpedcurrent_pts,
                       const float *final_pts);

        void compute_gt_da(const float *input_jts,
                           const float *target_jts,
                           const int winsize,
                           const float thresh,
                           const float *final_pts,
                           float *gtwarped_out,
                           float *gtda_ids);

        void update_pred_pts(const float *net_preds,
                             const float *net_grads);

        void update_pred_pts_unwarped(const float *net_preds,
                                      const float *net_grads);

        void initialize_poses(const float *init_poses,
                              const float *tar_poses);

        void update_masklabels_and_poses(const float *curr_masks,
                                         const float *curr_poses);

        void start_saving_frames(const string framesavedir);

        void stop_saving_frames();

def assert_contiguous(arrays):
    for arr in arrays:
        assert arr.flags['C_CONTIGUOUS']

cdef class PyPangolinViz:
    cdef PangolinViz *pangoviz     # hold a C++ instance which we're wrapping
    cdef int img_ht
    cdef int img_wd

    def __cinit__(self, int seq_len, int img_ht, int img_wd, float img_scale, int num_se3,
                    float fx, float fy, float cx, float cy, float dt, int oldgrippermodel,
                    string savedir):
        self.img_ht   = img_ht
        self.img_wd   = img_wd
        self.pangoviz = new PangolinViz(seq_len, img_ht, img_wd, img_scale, num_se3,
                                        fx, fy, cx, cy, dt, oldgrippermodel, savedir)

    def __dealloc__(self):
        del self.pangoviz

    def init_problem(self, np.ndarray[np.float32_t, ndim=1] start_jts,
                           np.ndarray[np.float32_t, ndim=1] goal_jts,
                           np.ndarray[np.float32_t, ndim=3] start_pts,
                           np.ndarray[np.float32_t, ndim=3] da_goal_pts):
        # Check input sanity
        #assert start_pts.flags['C_CONTIGUOUS'] and start_pts.shape == (3, self.img_ht, self.img_wd)
        #assert da_goal_pts.flags['C_CONTIGUOUS'] and da_goal_pts.shape == (3, self.img_ht, self.img_wd)
        # Run CPP code
        self.pangoviz.initialize_problem(&start_jts[0],
                                         &goal_jts[0],
                                         &start_pts[0,0,0],
                                         &da_goal_pts[0,0,0])

    def update_viz(self, np.ndarray[np.float32_t, ndim=3] input_pts,
                         np.ndarray[np.float32_t, ndim=3] output_pts_gt,
                         np.ndarray[np.float32_t, ndim=3] output_pts_pred,
                         np.ndarray[np.float32_t, ndim=1] jtangles_gt,
                         np.ndarray[np.float32_t, ndim=1] jtangles_pred):
        # Check input sanity
        #assert input_pts.flags['C_CONTIGUOUS'] and input_pts.shape == (3, self.img_ht, self.img_wd)
        #assert output_pts_gt.flags['C_CONTIGUOUS'] and output_pts_gt.shape == (3, self.img_ht, self.img_wd)
        #assert output_pts_pred.flags['C_CONTIGUOUS'] and output_pts_pred.shape == (3, self.img_ht, self.img_wd)
        # Run CPP code
        self.pangoviz.update_viz(&input_pts[0,0,0],
                                 &output_pts_gt[0,0,0],
                                 &output_pts_pred[0,0,0],
                                 &jtangles_gt[0],
                                 &jtangles_pred[0])

    def render_arm(self, np.ndarray[np.float32_t, ndim=1] config,
                         np.ndarray[np.float32_t, ndim=3] ptcloud):
        # Run CPP code
        self.pangoviz.render_arm(&config[0],
                                 &ptcloud[0,0,0])

    def update_da(self, np.ndarray[np.float32_t, ndim=3] init_pts,
                        np.ndarray[np.float32_t, ndim=3] current_pts,
                        np.ndarray[np.float32_t, ndim=3] current_da_ids,
                        np.ndarray[np.float32_t, ndim=3] warpedcurrent_pts,
                        np.ndarray[np.float32_t, ndim=3] final_pts):
        # Run CPP code
        self.pangoviz.update_da(&init_pts[0,0,0],
                                &current_pts[0,0,0],
                                &current_da_ids[0,0,0],
                                &warpedcurrent_pts[0,0,0],
                                &final_pts[0,0,0])

    def compute_gt_da(self, np.ndarray[np.float32_t, ndim=1] input_jts,
                            np.ndarray[np.float32_t, ndim=1] target_jts,
                            np.int32_t winsize,
                            np.float32_t thresh,
                            np.ndarray[np.float32_t, ndim=3] final_pts,
                            np.ndarray[np.float32_t, ndim=3] gtwarped_out,
                            np.ndarray[np.float32_t, ndim=3] gtda_ids):
        # Run CPP code
        self.pangoviz.compute_gt_da(&input_jts[0],
                                    &target_jts[0],
                                    winsize,
                                    thresh,
                                    &final_pts[0,0,0],
                                    &gtwarped_out[0,0,0],
                                    &gtda_ids[0,0,0])

    def update_pred_pts(self, np.ndarray[np.float32_t, ndim=3] net_preds,
                              np.ndarray[np.float32_t, ndim=3] net_grads):
        # Run CPP code
        self.pangoviz.update_pred_pts(&net_preds[0,0,0],
                                      &net_grads[0,0,0])

    def update_pred_pts_unwarped(self, np.ndarray[np.float32_t, ndim=3] net_preds,
                                       np.ndarray[np.float32_t, ndim=3] net_grads):
        # Run CPP code
        self.pangoviz.update_pred_pts_unwarped(&net_preds[0,0,0],
                                               &net_grads[0,0,0])

    def initialize_poses(self, np.ndarray[np.float32_t, ndim=3] init_poses,
                               np.ndarray[np.float32_t, ndim=3] tar_poses):
        # Run CPP code
        self.pangoviz.initialize_poses(&init_poses[0,0,0],
                                       &tar_poses[0,0,0])

    def update_masklabels_and_poses(self, np.ndarray[np.float32_t, ndim=3] curr_masks,
                                          np.ndarray[np.float32_t, ndim=3] curr_poses):
        # Run CPP code
        self.pangoviz.update_masklabels_and_poses(&curr_masks[0,0,0],
                                                  &curr_poses[0,0,0])

    def start_saving_frames(self, string framesavedir):
        # Run CPP code
        self.pangoviz.start_saving_frames(framesavedir)

    def stop_saving_frames(self):
        # Run CPP code
        self.pangoviz.stop_saving_frames()
