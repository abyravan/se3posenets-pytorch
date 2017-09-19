#import run_baxter_se3_ctrl as ctrl
import sys
import time
import numpy as np
import torch
import rospy
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import threading
import multiprocessing as mp

class DepthImageSubscriber:
    def __init__(self, ht, wd, scale, intrinsics):
        self.subscriber = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.callback)
        self.bridge     = CvBridge()
        self.ht, self.wd, self.scale = ht, wd, scale
        self.xygrid = intrinsics['xygrid'].cuda()
        self.ptcloud = torch.zeros(1,3,self.ht,self.wd).cuda()
        self.imgf = None
        self.mutex = threading.Lock()
        self.q = mp.Queue() # there should only ever be the most uptodate image in the queue
        self.p = mp.Process(target=self._update_ptcloud, args=(self,))
        self.p.start()

    def callback(self,data):
        try:
            self.imgf = self.bridge.imgmsg_to_cv2(data, "16UC1").astype(np.int16) * self.scale
        except CvBridgeError as e:
            print(e)

    def _update_ptcloud(self, src):
        while True:
            #assert self.imgf is not None, "Error: Haven't seen a single depth image yet!"
            while src.imgf is None: # wait until we get an image
                pass

            if (src.imgf.shape[0] != int(src.ht) or src.imgf.shape[1] != int(src.wd)):
                depth = cv2.resize(src.imgf, (int(src.wd), int(src.ht)),
                                   interpolation=cv2.INTER_NEAREST)  # Resize image with no interpolation (NN lookup)
            else:
                depth = self.imgf

            ### Setup point cloud
            src.ptcloud[0, 2].copy_(torch.FloatTensor(depth).cuda())  # Copy depth

            # Compute x & y values for the 3D points (= xygrid * depths)
            xy = src.ptcloud[:, 0:2]
            xy.copy_(src.xygrid.expand_as(xy))  # = xygrid
            xy.mul_(src.ptcloud[0, 2])  # = xygrid * depths

            # save to queue
            #l.acquire()
            if len(src.q):
                src.q.empty() # remove old image
            src.q.put(self.ptcloud.clone())
            #l.release()

    def get_ptcloud(self):
        # Return clone
        print 'Getting'
        #self.mutex.acquire()
        img = self.q.get(block=True) # wait if necessary to get image
        #self.mutex.release()
        print 'Got'
        return img


def image_publisher(topicname):
    pub = rospy.Publisher(topicname,Image, queue_size=10)
    while True:
        pub.publish()
        time.sleep(0.02)

def main():

    rospy.init_node('image_tester')

    #topicname = '/test/images'
    #ipub = mp.Process(target=image_publisher, args=(topicname,))

    print 'Node initialized'

    depth_src = DepthImageSubscriber(100,
                                          100,
                                          1000,
                                          {'xygrid':torch.Tensor([1])})

    print 'Source initialized'

    try:
        while True:
            img = depth_src.get_ptcloud()
            #img = np.random.random((100,100))
            print 'Got image'
            plt.imshow(img)
            #plt.show(block=True)
            plt.draw()
            plt.pause(2)
            #time.sleep(2)
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()