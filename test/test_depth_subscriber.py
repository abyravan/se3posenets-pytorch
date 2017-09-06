#!/usr/bin/env python
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import torch
import data

from __future__ import print_function

class DepthImageSubscriber:
    global img
    def __init__(self, ht, wd, scale, intrinsics):
        self.subscriber = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.callback)
        self.bridge     = CvBridge()
        self.ht, self.wd, self.scale = ht, wd, scale
        self.intrinsics = intrinsics

    def callback(self,data):
        try:
            self.imgf = self.bridge.imgmsg_to_cv2(data, "16UC1").astype(np.int16) * self.scale
        except CvBridgeError as e:
            print(e)

    def get_ptcloud(self):
        ptcloud = torch.zeros(1,3,self.ht,self.wd)
        if (self.imgf.shape[0] != int(self.ht) or self.imgf.shape[1] != int(self.wd)):
            depth = cv2.resize(self.imgf, (int(self.wd), int(self.ht)), interpolation=cv2.INTER_NEAREST)  # Resize image with no interpolation (NN lookup)
        else:
            depth = self.imgf
        ptcloud[0, 2].copy_(torch.FloatTensor(depth))  # Copy depth

        # Compute x & y values for the 3D points (= xygrid * depths)
        xy = ptcloud[:, 0:2]
        xy.copy_(self.intrinsics['xygrid'].expand_as(xy))  # = xygrid
        xy.mul_(ptcloud[0, 2])  # = xygrid * depths
        return ptcloud

def main(args):
    ht, wd, scale = 240, 320, 1e-3
    intrinsics = {'fx': 589.3664541825391 / 2,
                  'fy': 589.3664541825391 / 2,
                  'cx': 320.5 / 2,
                  'cy': 240.5 / 2}
    intrinsics['xygrid'] = data.compute_camera_xygrid_from_intrinsics(ht, wd, intrinsics)

    DI = DepthImageSubscriber(ht, wd, scale, intrinsics)
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
