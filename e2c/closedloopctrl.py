#!/usr/bin/env python
# To run:
# sourceblocks && python e2c/closedloopctrl.py -c <yaml-file>

# Global imports
import h5py
import numpy as np
import os
import sys

# ROS imports
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gazebo_learning_planning.srv import Configure
from gazebo_learning_planning.srv import ConfigureRequest
from gazebo_learning_planning.msg import ConfigureObject

# Chris's simulator
from simulator.yumi_simulation import YumiSimulation

arm_l_idx = [0, 2, 4, 6, 8, 10, 12]
arm_r_idx = [1, 3, 5, 7, 9, 11, 13]
gripper_r_idx = 14
gripper_l_idx = 15

######### Interface to YUMI robot
class YUMIInterface(object):
    #### Callbacks defined before constructor initialization
    # Joint state callback
    def _js_cb(self, msg):
        self.q     = np.array(msg.position)
        self.dq    = np.array(msg.velocity)
        for name, q in zip(msg.name, msg.position):
            self.qdict[name] = q

    # RGB image callback
    def _rgb_cb(self, msg):
        try:
            frame    = self.bridge.imgmsg_to_cv2(msg)
            self.rgb = np.array(frame, dtype=np.uint8)
        except CvBridgeError as e:
            print(e)

    # Depth image callback
    def _depth_cb(self, msg):
        try:
            frame      = self.bridge.imgmsg_to_cv2(msg)
            print(frame.dtype)
            self.depth = np.array(frame, dtype=np.float32)
        except CvBridgeError as e:
            print(e)

    #### Constructor
    def __init__(self, hz=30.):
        # Setup service proxy for initializing simulator state from h5
        self.configure = rospy.ServiceProxy("simulation/configure", Configure)

        # Setup callbacks and services for JointState
        self.js_sub = rospy.Subscriber("/robot/joint_states", JointState, self._js_cb)
        self.js_cmd = rospy.Publisher(YumiSimulation.listen_topic, JointState, queue_size=1000)
        self.q      = None
        self.dq     = None
        self.qdict  = {}
        self.hz     = hz

        # Setup callback for RGB/D images
        self.bridge    = CvBridge()
        self.rgb_sub   = rospy.Subscriber("/robot/image", Image, self._rgb_cb)
        self.depth_sub = rospy.Subscriber("/robot/depth_image", Image, self._depth_cb)
        self.rgb       = None
        self.depth     = None

    # Send a full command to the
    def commandJts(self, rg, lg, ra, la):
        '''
        Publish the provided position commands.
        '''
        data = {}
        for k, v in zip(YumiSimulation.robot_left_gripper, [lg]):
            data[k] = v
        for k, v in zip(YumiSimulation.robot_right_gripper, [rg]):
            data[k] = v
        for k, v in zip(YumiSimulation.yumi_left_arm, la):
            data[k] = v
        for k, v in zip(YumiSimulation.yumi_right_arm, ra):
            data[k] = v
        msg          = JointState()
        msg.name     = data.keys()
        msg.position = data.values()
        self.js_cmd.publish(msg)

    # Command joint velocities to the robot (internally integrates to get positions and sends those)
    def commandJtVelocities(self, cmd, dt=None):
        if dt is None:
            dt = 1./self.hz
        msg          = JointState()
        msg.name     = YumiSimulation.yumi_joint_names
        msg.position = self.q + cmd*dt
        self.js_cmd.publish(msg)

    # Replay the commands from a h5 data file
    def replayH5(self, h5, start=None, goal=None):
        # Get the commands from the h5 data
        rg = np.array(h5['right_gripper_cmd'])
        lg = np.array(h5['left_gripper_cmd'])
        ra = np.array(h5['right_arm_cmd'])
        la = np.array(h5['left_arm_cmd'])

        # Get start and goal ids
        start = 0 if start is None else max(start, 0) # +ve start id
        if goal is not None:
            assert (goal >= start), "Goal id: {} is < Start id: {}".format(goal, start)
        else:
            goal = rg.shape[0]
        print('Replaying the commands from the H5 file. Start: {}, Goal: {}'.format(start, goal))

        # Initialize with the first command
        rospy.sleep(0.5)
        self.commandJts(rg[start], lg[start], ra[start], la[start])
        rospy.sleep(1.)

        # Send all commands from [start, goal)
        rate = rospy.Rate(self.hz)
        for i in range(start, goal):
            # publish a single command to the robot
            self.commandJts(rg[i], lg[i], ra[i], la[i])
            rate.sleep()

    # Initialize the state of the simulator from a H5 data file
    def configureH5(self, h5):
        # Setup object configuration. Max of 20 objects we might actually care about
        msg = ConfigureRequest()
        print('Setting up the simulator state from the initial config of the H5 file')
        for i in range(20):
            name = "pose%d" % i
            if name in h5:
                poses = np.array(h5[name])
                if poses.shape[0] > 0:
                    # Setup object pose
                    pose = poses[0]
                    obj = ConfigureObject()
                    obj.id.data = i
                    obj.pose.position.x = pose[0]
                    obj.pose.position.y = pose[1]
                    obj.pose.position.z = pose[2]
                    obj.pose.orientation.x = pose[3]
                    obj.pose.orientation.y = pose[4]
                    obj.pose.orientation.z = pose[5]
                    obj.pose.orientation.w = pose[6]
                    # Add object pose to msg
                    msg.object_poses.append(obj)

        # Setup robot configuration
        q = np.array(h5["robot_positions"])[0]
        msg.joint_state.position = q

        # Send configuration to simulator
        self.configure(msg)

if __name__ == "__main__":
    # TODO(@cpaxton):
    if len(sys.argv) < 2:
        print("usage: %s [filename]" % str(sys.argv[0]))

    rospy.init_node("control_robot")
    filename = sys.argv[1]
    h5 = h5py.File(os.path.expanduser(filename))
    iface = YUMIInterface(hz=30.)
    iface.configureH5(h5)
    iface.replayH5(h5)
