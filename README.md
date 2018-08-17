
# Intro

Setting up training:
1) Install the following packages:
     PyTorch: conda install pytorch=0.4.0 cuda90 torchvision -c pytorch (0.4.1 has some compile issues) 
     Tensorflow (only the CPU version - for tensor board visualisation), 
     OpenCV (condo install opencv -c menpo)
     configargparse
2) Compile the code with: sh make.sh (in the main se3nets-pytorch folder)
3) To train (npc-pivot branch): python train_se3posenets.py -c <config_file>
     For an example config file, look at config/icra18final/simdata/se3pose/def_rmsprop.yaml
     At the least, you might have to change the “data” path inside the config file to be your path to the dataset

Setting up ROS compatibility:
1) After sourcing the correct conda version (2.7 default), run the following install command:
    pip install -U rospkg catkin_pkg
  This should install the necessary bridge between ROS and conda.
2) To check if this works and in general to run a ROS workspace you need to first source the correct conda version (sourceconda2) and then source the ROS workspace (such as for baxter it could be . baxter.sh sim or .baxter.sh). Then to test, go to the terminal and type: import rospy. If this succeeds, you should be set.

Setting up control:
1) You need to install Pangolin (https://github.com/stevenlovegrove/Pangolin)
2) Next, you need to compile the code in lib/:
     Type (in lib/): cd torchviz && mkdir -p build && cd build && cmake ../ && make -j7 && cd ../../ && make
3) To test the control on the simulated baxter data (se3compose branch), do:
     python run_control.py --checkpoint <path-to-pre-trained-se3-pose-net> --optimization gn --max-iter 200 --gn-perturb 1e-3 --only-top6-jts --ctrl-mag-decay 1.0 --loss-thresh 2e-3 --num-configs -1 --save-dir temp
     This will create a window so it can’t be run over ssh
4) To test the open loop controller (conjugate gradient) on the simulated baxter data (se3compose branch) do:
     python run_control_openloop.py --checkpoint <path-to-pre-trained-se3-pose-net> --only-top4-jts --loss-thresh 2e-3 --num-configs 5 --save-dir temp --ctrl-init zero --optimizer xalglib_cg --max-iter 200 --horizon 10 --goal-horizon 5 --loss-scale 100

Setting up Chris's pybullet interface:
1) Setup ROS workspace:
    sourceconda2 # Source your conda version
    sourceindigo # Source Indigo/Kinetic ROS
    pip install -U rospkg catkin_pkg
    pip install catkin_tools empy
    mkdir workspace && cd workspace
    catkin init
    mkdir src && cd src
    git clone https://github.com/cpaxton/gazebo-learning-planning.git gazebo_learning_planning
    git clone https://github.com/clemense/yumi.git
    git clone https://github.com/orocos/orocos_kinematics_dynamics.git
    touch yumi/yumi_hw/CATKIN_IGNORE
    touch yumi/yumi_launch/CATKIN_IGNORE
    catkin build
    pip install -U pybullet
2) To test:
    source ../devel/setup.bash
    ./gazebo_learning_planning/posetest

    In a separate terminal, do:
    sourceconda2 && source ../devel/setup.bash
    ./gazebo_learning_planning/nodes/control.py <path-to-saved-h5-data>

# Details

```
conda install opencv -c menpo
conda install configargparse
# Install tensorflow
```

  - layers are in `layers/`
  - data loading is in `data.py`
  - network structures are in `ctrlnets.py`
  - `main_ctrlnets_multi_f.py` contrains control nets code

Cleanest branch is `ngc-pivot`

# Build

## Prerequisites

```
sudo apt-get install libglew-dev libmatheval-dev
```

Install Pangolin from source:
```
# Change to some reasonable path for Pangolin
cd $HOME/src

# Clone
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

# Build with CMake
mkdir build
cd build
cmake ..
sudo make install
```


## SE3 Nets
```
sh make.sh
```

## Control Experiments
```
mkdir -p build && cd build && cmake ../ && make -j7 && cd ../../ && python setup.py install
```

# Run Experiments
Run:
```
train_flownets.py # baseline models
train_se3posenets.py
se3flownets.py

# open-loop control (experimental)
run_control_openloop.py
```

