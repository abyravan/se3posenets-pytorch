import os
import torch

################# HELPER FUNCTIONS

# ### Check if torch variable is of type autograd.Variable
# def is_var(x):
#     return (type(x) == torch.autograd.Variable)

### Convert variable to numpy array
def to_np(x):
    return x.detach().cpu().numpy()

### Set gradient flag
def req_grad(x, req=False):
    x.requires_grad = req
    return x

# ### Convert torch tensor to autograd.variable
# def to_var(x, to_cuda=False, requires_grad=False, volatile=False):
#     if torch.cuda.is_available() and to_cuda:
#         x = x.cuda()
#     return torch.autograd.Variable(x, requires_grad=requires_grad, volatile=volatile)

### Create a directory. From: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

################# HELPER CLASSES

### Computes sum/avg stats
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

### Write to stdout and log file
class Tee(object):
    def __init__(self, stdout, logfile):
        self.stdout = stdout
        self.logfile = logfile

    def write(self, obj):
        self.stdout.write(obj)
        self.logfile.write(obj)

    def flush(self):
        self.stdout.flush()

    def __del__(self):
        self.logfile.close()

### Enumerate over data
class DataEnumerator(object):
    """Allows iterating over a data loader easily"""
    def __init__(self, data):
        self.data   = data # Store the data
        self.len    = len(self.data) # Number of samples in the entire data
        self.niters = 0    # Num iterations in current run
        self.nruns  = 0    # Num rounds over the entire data
        self.enumerator = enumerate(self.data) # Keeps an iterator around
        self.ids = [] # Used in case we want to keep track of the ids in the data
    def next(self):
        try:
            sample = next(self.enumerator) #self.enumerator.next() # Get next sample
        except StopIteration:
            self.enumerator = enumerate(self.data) # Reset enumerator once it reaches the end
            self.nruns += 1 # Done with one complete run of the data
            self.niters = 0 # Num iters in current run
            sample = next(self.enumerator) #self.enumerator.next() # Get next sample
            #print('Completed a run over the data. Num total runs: {}, Num total iters: {}'.format(
            #    self.nruns, self.niters+1))
        self.niters += 1 # Increment iteration count
        # try:
        #     self.ids.append(sample[1]['id']) # Append if it exists
        # except:
        #     pass
        return sample

    def __len__(self):
        return len(self.data)

    def iteration_count(self):
        return (self.nruns * self.len) + self.niters
