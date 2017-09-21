
import sys, os
import numpy as np
import rospy
import baxter_interface
import time
import threading


class Joint_Commander(threading.Thread):

	def __init__(self, limb, threshold=0.008726646):

		threading.Thread.__init__(self)
		self.threshold = threshold
		self.mutex = threading.Lock()
		self.valid = False
		self.limb = limb
		self.target = self.limb.joint_angles()
		self.error = lambda: np.array([abs(self.target[joint] - current) for joint, current in self.limb.joint_angles().items()])

	def run(self):
		while not rospy.is_shutdown():
			
			if self.valid and (self.error() > self.threshold).any():
				self.mutex.acquire()
				self.limb.set_joint_positions({joint:(0.012488*self.target[joint] + 0.98751*current) for joint, current in self.limb.joint_angles().items()})
				self.mutex.release()
			time.sleep(0.01) # 100 Hz


	def set_target(self, target=None):
		if target is None:
			target = self.limb.joint_angles()
		self.mutex.acquire()
		self.target = target
		self.mutex.release()

	def set_valid(self, valid=None):
		if valid is None:
			valid = not self.valid
		self.valid = valid


if __name__ == '__main__':
	# test joint commander
	
	rospy.init_node('Bx_Joint_Commander')
	limb = baxter_interface.Limb('right')
	jc = Joint_Commander(limb)

	jc.start()

	targets = [{'right_s0': 0.11159710231866385, 'right_s1': 0.5242379342598401, 'right_w0': -1.9562089997508738, 'right_w1': -0.756252528427509, 'right_w2': -0.5579855115933192, 'right_e0': 1.2429079333841564, 'right_e1': 0.6074563920026238},
			{'right_s0': 0.3673883986985566, 'right_s1': -0.8271991398672094, 'right_w0': -1.700417703370981, 'right_w1': 0.7796457354427615, 'right_w2': 0.8574952604279463, 'right_e0': 1.6137477888554552, 'right_e1': 0.48473792897179074},
			]

	jc.set_valid()

	for i in range(1000):
		jc.set_target(targets[i % len(targets)])
		time.sleep(1)

	

	
