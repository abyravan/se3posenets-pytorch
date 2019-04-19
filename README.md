Setting up training:
1) Install the Anaconda package manager (https://repo.continuum.io/archive/). Code works with Python 2 & 3.
2) Install the following packages:
     PyTorch: conda install pytorch=0.4.0 cuda90 torchvision -c pytorch (0.4.1 has some compile issues) 
     Tensorflow (only the CPU version - for tensor board visualisation, pip install tensorflow),
     OpenCV (conda install opencv -c menpo)
     configargparse (conda install configargparse)
3) Compile the code with: sh make.sh (in the main se3nets-pytorch folder)
4) To train the se3 pose nets: python train_se3posenets.py -c <config_file>
     For an example config file, look at config/icra18final/simdata/se3pose/def_rmsprop.yaml
     You *need* to change the “data” path inside the config file to be your path to the dataset.
5) To train the baseline se3/flow networks: python train_flow_se3_nets.py -c <config_file>
     For the flow network, the config file is at config/icra18final/simdata/flow/def.yaml
     For the se3 net, the config file is at config/icra18final/simdata/se3/def.yaml
6) You can visualize the errors and predictions using tensorboard:
     tensorboard --logdir=<log_dir> --port=8000
   where log_dir can be the path to the directory specified in the config file as save-dir.
