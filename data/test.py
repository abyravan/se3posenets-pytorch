import pybullet as p
import pybullet_data
#physicsClient = p.connect(p.DIRECT, options='--opengl2')#or p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath('/home/barun/Projects/se3nets-pytorch/data/models/') #used by loadURDF
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
p.setPhysicsEngineParameter(fixedTimeStep=1.0/60.) #, numSolverIterations=5, numSubSteps=2)
#p.setRealTimeSimulation(True)
planeId = p.loadURDF("plane.urdf")
baxterStartPos = [0,0,0.93]
baxterStartOrientation = p.getQuaternionFromEuler([0,0,0])
#baxterId = p.loadURDF("baxter/model.urdf", baxterStartPos, baxterStartOrientation)
baxterId = p.loadURDF("/home/barun/Projects/se3nets-pytorch/data/models/baxter/model.urdf", baxterStartPos, baxterStartOrientation, useFixedBase=True)
p.setJointMotorControlArray(baxterId, [36,37], p.POSITION_CONTROL, targetPositions=[-1.57,-1.57])
import time; st = time.time();
import matplotlib.pyplot as plt
plt.ion()
plt.figure(100)
plt.show()
prev = st
while((time.time()-st) < 100.0):
	st1 = time.time()
	p.stepSimulation()
	print('Sim time: {}'.format(time.time()-st1))
	if (time.time()-prev) > 1.0:
		st2 = time.time()
		wd, ht, rgb, depth, seg = p.getCameraImage(width=320, height=240, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
		print('Render time: {}'.format(time.time()-st2))
		plt.imshow(rgb)
		plt.pause(0.01)
		prev = time.time()
baxterPos, baxterOrn = p.getBasePositionAndOrientation(baxterId)
print(baxterPos, baxterOrn)
#p.disconnect()

