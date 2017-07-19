import csv
import torch
import numpy as np
import cv2
import os

############
### Helper functions for reading baxter data

# Read baxter state files
def read_baxter_state_file(filename):
    ret = {}
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        ret['actjtpos'] = torch.Tensor(spamreader.next()[0:-1]) # Last element is a string due to the way the file is created
        ret['actjtvel'] = torch.Tensor(spamreader.next()[0:-1])
        ret['actjteff'] = torch.Tensor(spamreader.next()[0:-1])
        ret['comjtpos'] = torch.Tensor(spamreader.next()[0:-1])
        ret['comjtvel'] = torch.Tensor(spamreader.next()[0:-1])
        ret['comjtacc'] = torch.Tensor(spamreader.next()[0:-1])
        ret['tarendeffpos'] = torch.Tensor(spamreader.next()[0:-1])
    return ret

# Read baxter SE3 state file for all the joints
def read_baxter_se3state_file(filename):
    # Read all the lines in the SE3-state file
    lines = []
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for row in spamreader:
            if len(row)  == 0:
                continue
            if type(row[-1]) == str: # In case we have a string at the end of the list
                row = row[0:-1]
            lines.append(torch.Tensor(row))
    # Parse the SE3-states
    ret, ctr = {}, 0
    while (ctr < len(lines)):
        id      = int(lines[ctr][0])            # ID of mesh
        data    = lines[ctr+1].view(3,4)   # Transform data
        T       = torch.cat([data[0:3,1:4], data[0:3,0]], 1) # [R | t]
        ret[id] = T # Add to list of transforms
        ctr += 2    # Increment counter
    return ret

# Read baxter joint labels and their corresponding mesh index value
def read_baxter_labels_file(filename):
    ret = {}
    with open(filename, 'rb') as csvfile:
        spamreader     = csv.reader(csvfile, delimiter=' ')
        ret['frames']  = spamreader.next()[0:-1]
        ret['meshids'] = torch.IntTensor( [int(x) for x in spamreader.next()[0:-1]] )
    return ret

# Read baxter camera data file
def read_cameradata_file(filename):
    # Read lines in the file
    lines = []
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            lines.append([x for x in row if x != ''])
    # Compute modelview and camera parameter matrix
    ret = {}
    ret['modelview'] = torch.Tensor([float(x) for x in lines[1] + lines[2] + lines[3] + lines[4]]).view(4,4).clone()
    ret['camparam']  = torch.Tensor([float(x) for x in lines[6] + lines[7] + lines[8] + lines[9]]).view(4,4).clone()
    return ret

############
### Helper functions for reading image data

# Read depth image from disk
def read_depth_image(filename, ht = 240, wd = 320, scale = 1e-4):
    imgf = cv2.imread(filename, -1).astype(np.int16) * scale # Read image (unsigned short), convert to short & scale to get float
    if (imgf.shape[0] != int(ht) or imgf.shape[1] != int(wd)):
        imgscale = cv2.resize(imgf, (int(wd), int(ht)), interpolation=cv2.INTER_NEAREST) # Resize image with no interpolation (NN lookup)
    else:
        imgscale = imgf
    return torch.Tensor(imgscale).unsqueeze(0) # Add extra dimension

# Read flow image from disk
def read_flow_image_xyz(filename, ht = 240, wd = 320, scale = 1e-4):
    imgf = cv2.imread(filename, -1).astype(np.int16) * scale # Read image (unsigned short), convert to short & scale to get float
    if (imgf.shape[0] != int(ht) or imgf.shape[1] != int(wd)):
        imgscale = cv2.resize(imgf, (int(wd), int(ht)), interpolation=cv2.INTER_NEAREST) # Resize image with no interpolation (NN lookup)
    else:
        imgscale = imgf
    return torch.Tensor(imgscale.transpose((2,0,1))) # NOTE: OpenCV reads BGR so it's already xyz when it is read

############
###  SETUP DATASETS: RECURRENT VERSIONS FOR BAXTER DATA - FROM NATHAN'S BAG FILE

### Helper functions for reading the data directories & loading train/test files
def read_recurrent_baxter_dataset(load_dirs, img_suffix, train_per, step_len, seq_len):
	# Get all the load directories
	load_dir_splits = load_dirs.split(',,') # Get all the load directories

	# Iterate over each load directory to find the datasets
	datasets = {'train': [], 'test': [], 'ntrain': [], 'ntest': []}
	for load_dir in load_dir_splits:
		# Get folder names & data statistics for a single load-directory
		dirs = os.listdir(load_dir)
		for dir in dirs:
			path = os.path.join(load_dir, dir)
			if (os.path.isdir(path)):
				# Get number of images in the folder
				statsfile = os.path.join(path, 'postprocessstats.txt')
				assert(os.path.exists(statsfile))
				reader  = csv.reader(statsfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
				nimages = reader.next()[0]
				print('Found {} images in the dataset: {}'.format(nimages, path))

				# Setup training and test splits in the dataset
				ntrain = int(train_per * nimages) # Use first train_per images for training
				train = {'path'   : path,
						 'suffix' : img_suffix,
						 'start'  : 0,
						 'end'    : ntrain-1,
						 'step'   : step_len,
						 'seq'    : seq_len} # start & end inclusive
				test  = {'path'   : path,
						 'suffix' : img_suffix,
						 'start'  : ntrain,
						 'end'    : nimages-1,
						 'step'   : step_len,
						 'seq'    : seq_len} # start & end inclusive
				datasets['train'].append(train)
				datasets['test'].append(test)
				datasets['ntrain'].append(ntrain)
				datasets['ntest'].append(nimages-ntrain)
	return datasets

### Generate the data files (with all the depth, flow etc.) for each sequence
def generate_baxter_sequence(dataset, id):
    # Get stuff from the dataset
    path, step, seq, suffix = dataset['path'], dataset['step'], dataset['seq'], dataset['suffix']
    assert(id >= dataset['start'] and id < dataset['end']);
    # Setup start/end IDs of the sequence
    start, end = id, id + (step * seq)
    sequence, ct, stepid = {}, 0, step
    for k in xrange(start, end+1, step):
        sequence[ct] = {'depth'     : path + 'depth'  + suffix + str(k) + '.png',
                        'label'     : path + 'labels' + suffix + str(k) + '.png',
                        'state1'    : path + 'state' + str(k)   + '.txt',
                        'state2'    : path + 'state' + str(k+1) + '.txt',
                        'se3state1' : path + 'se3state' + str(k) + '.txt',
                        'se3state2' : path + 'se3state' + str(k+1) + '.txt',
                        'flow'      : path + 'flow_' + str(stepid)
                                      + '/flow' + suffix + str(start) + '.png'}
        stepid += step # Get flow from start image to the next step
        ct += 1        # Increment counter
    return sequence


'''

'''