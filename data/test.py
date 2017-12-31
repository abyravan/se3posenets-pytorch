import pybullet as p
import pybullet_data
physicsClient = p.connect(p.GUI, options='--opengl2')#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath('/home/barun/Projects/se3nets-pytorch/data/models/') #used by loadURDF
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
p.setPhysicsEngineParameter(fixedTimeStep=1.0/50.) #, numSolverIterations=5, numSubSteps=2)
planeId = p.loadURDF("plane.urdf")
baxterStartPos = [0,0,0.93]
baxterStartOrientation = p.getQuaternionFromEuler([0,0,0])
#baxterId = p.loadURDF("baxter/model.urdf", baxterStartPos, baxterStartOrientation)
baxterId = p.loadURDF("/home/barun/Projects/se3nets-pytorch/data/models/baxter/model.urdf", baxterStartPos, baxterStartOrientation, useFixedBase=True)
p.setJointMotorControlArray(baxterId, [36,37], p.POSITION_CONTROL, targetPositions=[-1.57,-1.57])
dt = 0.01
import time; st = time.time();
while((time.time()-st) < 100.0):
	p.stepSimulation()
baxterPos, baxterOrn = p.getBasePositionAndOrientation(baxterId)
print(baxterPos, baxterOrn)
#p.disconnect()

