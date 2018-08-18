#!/usr/bin/env python

# Global imports
import h5py
import numpy as np
import os
import sys

# ROS imports
import rospy
from sensor_msgs.msg import JointState
from gazebo_learning_planning.srv import Configure
from gazebo_learning_planning.srv import ConfigureRequest
from gazebo_learning_planning.msg import ConfigureObject

# Chris's simulator
from simulator.yumi_simulation import YumiSimulation

arm_l_idx = [0, 2, 4, 6, 8, 10, 12]
arm_r_idx = [1, 3, 5, 7, 9, 11, 13]
gripper_r_idx = 14
gripper_l_idx = 15

def configureH5(h5):
    print(h5.keys())

    msg = ConfigureRequest()

    # Max of 20 objects we might actually care about
    for i in range(20):
        name = "pose%d" % i
        if name in h5:
            poses = np.array(h5[name])
            if poses.shape[0] > 0:
                pose = poses[0]
                print("Found object pose:", name, poses.shape)

                obj = ConfigureObject()
                obj.id.data = i
                obj.pose.position.x = pose[0]
                obj.pose.position.y = pose[1]
                obj.pose.position.z = pose[2]
                obj.pose.orientation.x = pose[3]
                obj.pose.orientation.y = pose[4]
                obj.pose.orientation.z = pose[5]
                obj.pose.orientation.w = pose[6]

                msg.object_poses.append(obj)

    q = np.array(h5["robot_positions"])[0]
    msg.joint_state.position = q

    configure = rospy.ServiceProxy("simulation/configure", Configure)
    configure(msg)


class Interface(object):

    def _js_cb(self, msg):
        self.q = msg.position
        self.dq = msg.velocity
        self.names = msg.name

    def __init__(self, hz=30):
        self.js_sub = rospy.Subscriber("/robot/joint_states", JointState,
                self._js_cb)
        self.js_cmd = rospy.Publisher(YumiSimulation.listen_topic, JointState,
                queue_size=1000)
        self.q = None
        self.dq = None
        self.names = None
        self.hz = hz

    def send(self, rg, lg, ra, la):
        '''
        Publish the provided position commands.
        '''
        data = {}
        for k, v in zip(YumiSimulation.robot_left_gripper, [lg]):
            print(k,"=",v)
            data[k] = v
        for k, v in zip(YumiSimulation.robot_right_gripper, [rg]):
            print(k,"=",v)
            data[k] = v
        for k, v in zip(YumiSimulation.yumi_left_arm, la):
            print(k,"=",v)
            data[k] = v
        for k, v in zip(YumiSimulation.yumi_right_arm, ra):
            print(k,"=",v)
            data[k] = v
        msg = JointState()
        msg.name = data.keys()
        msg.position = data.values()
        self.js_cmd.publish(msg)

    def sendDiff(self, cmd, dt=None):
        if dt is None:
            dt = 1./self.hz
        msg = JointState()
        msg.name = YumiSimulation.yumi_joint_names
        msg.position = self.q + cmd*dt
        self._js_cb.publish(msg)

    def replay(self, h5):
        q = np.array(h5['robot_positions'])
        rg = np.array(h5['right_gripper_cmd'])
        lg = np.array(h5['left_gripper_cmd'])
        ra = np.array(h5['right_arm_cmd'])
        la = np.array(h5['left_arm_cmd'])

        rospy.sleep(0.5)
        self.send(rg[0], lg[0], ra[0], la[0])
        rospy.sleep(1.)

        i = 0
        rate = rospy.Rate(self.hz)
        for i in range(rg.shape[0]):
            # publish a single command to the robot
            #send(pub, rg[i], lg[i], ra[i], la[i])
            #cmd = getCmd(q[i], rg[i], lg[i], ra[i], la[i])
            self.send(rg[i], lg[i], ra[i], la[i])
            rate.sleep()

if __name__ == "__main__":
    # TODO(@cpaxton):
    if len(sys.argv) < 2:
        print("usage: %s [filename]" % str(sys.argv[0]))

    rospy.init_node("control_robot")
    filename = sys.argv[1]
    h5 = h5py.File(os.path.expanduser(filename))
    configureH5(h5)
    iface = Interface(hz=30.)
    iface.replay(h5)
