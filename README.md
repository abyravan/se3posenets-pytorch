
# Intro

Setting up training:
1) Install the following packages: 
     Tensorflow (only the CPU version - for tensor board visualisation), 
     OpenCV (condo install opencv -c menpo)
     configargparse
2) Compile the code with: sh make.sh (in the main se3nets-pytorch folder)
3) To train (npc-pivot branch): python train_se3posenets.py -c <config_file>
     For an example config file, look at config/icra18final/simdata/se3pose/def_rmsprop.yaml
     At the least, you might have to change the “data” path inside the config file to be your path to the dataset

Setting up control:
1) You need to install Pangolin (https://github.com/stevenlovegrove/Pangolin)
2) Next, you need to compile the code in lib/:
     Type (in lib/): cd torchviz && mkdir -p build && cd build && cmake ../ && make -j7 && cd ../../ && make
3) To test the control on the simulated baxter data (se3compose branch), do:
     python run_control.py --checkpoint <path-to-pre-trained-se3-pose-net> --optimization gn --max-iter 200 --gn-perturb 1e-3 --only-top6-jts --ctrl-mag-decay 1.0 --loss-thresh 2e-3 --num-configs -1 --save-dir temp
     This will create a window so it can’t be run over ssh
4) To test the open loop controller (conjugate gradient) on the simulated baxter data (se3compose branch) do:
     python run_control_openloop.py --checkpoint <path-to-pre-trained-se3-pose-net> --only-top4-jts --loss-thresh 2e-3 --num-configs 5 --save-dir temp --ctrl-init zero --optimizer xalglib_cg --max-iter 200 --horizon 10 --goal-horizon 5 --loss-scale 100

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
# Dependencies

Pangolin:
```
git clone https://github.com/stevenlovegrove/Pangolin.git
```

