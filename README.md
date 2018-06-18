
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
