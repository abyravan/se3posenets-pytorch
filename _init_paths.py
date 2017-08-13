# --------------------------------------------------------
# SE3-POSE-NETS
# Copyright (c) 2017 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Arunkumar Byravan
# --------------------------------------------------------

"""Set up paths for SE3-Pose-Nets."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)