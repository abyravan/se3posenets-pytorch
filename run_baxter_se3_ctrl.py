#!/usr/bin/env python

"""
Joint Position Control in Position Mode
"""

import numpy as np
import argparse
import rospy

from dynamic_reconfigure.server import (
    Server,
)
from std_msgs.msg import (
    Empty,
)

import baxter_interface

from baxter_interface import CHECK_VERSION


class SE3ControlPositionMode(object):
    """

    @param limb: limb on which to run joint control in position mode

    """

    def __init__(self, limb):

        # control parameters
        self._rate = 100.0  # Hz
        self._missed_cmds = 5.0  # Missed cycles before triggering timeout

        # create our limb instance
        self._limb = baxter_interface.Limb(limb)

        self._start_angles = dict()

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + limb + '/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")

    def move_to_neutral(self):
        """
        Moves the limb to neutral location.
        """
        self._limb.move_to_neutral()

    def move_to_pos(self, joint_angles):
        """
        Moves the limb to joint_angles.
        """
        self._limb.move_to_joint_positions(joint_angles)

    def cur_pos(self):
        return self._limb.joint_angles()

    def control_loop(self):
        # given start and end configuration this should loop over
        # optimizing controls with the se3net, updating the se3net
        # with current joint angles and point cloud and moving to new
        # joint angles
        return

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        print("\nExiting example...")
        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()


def main():
    """RSDK Joint Torque Example: Joint Springs

    Moves the specified limb to a neutral location and enters
    torque control mode
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        '-l', '--limb', dest='limb', required=True, choices=['left', 'right'],
        help='limb on which to attach joint springs'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("baxter_se3_control_%s" % (args.limb,))
    print(args.limb)

    jc = SE3ControlPositionMode(args.limb)
    # register shutdown callback
    rospy.on_shutdown(jc.clean_shutdown)
    # jc.move_to_neutral()
    pos_1 = {'right_s0': -0.459, 'right_s1': -0.202, 'right_e0':
             1.807, 'right_e1': 1.714, 'right_w0': -0.906,
             'right_w1': -1.545, 'right_w2': -0.276}

    pos_2 = {'right_s0': -0.395, 'right_s1': -0.202, 'right_e0':
             1.831, 'right_e1': 1.981, 'right_w0': -1.979,
             'right_w1': -1.100, 'right_w2': -0.448}

    for i in range(10):
        jc.move_to_pos(pos_1)
        jc.move_to_pos(pos_2)


if __name__ == "__main__":
    main()
