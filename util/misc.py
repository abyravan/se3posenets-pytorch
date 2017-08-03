import os
import torch

################# HELPER FUNCTIONS

### Check if torch variable is of type autograd.Variable
def is_var(x):
    return (type(x) == torch.autograd.variable.Variable)

### Convert variable to numpy array
def to_np(x):
    if is_var(x):
        return x.data.cpu().numpy()
    else:
        return x.cpu().numpy()

### Convert torch tensor to autograd.variable
def to_var(x, to_cuda=False, requires_grad=False):
    if torch.cuda.is_available() and to_cuda:
        x = x.cuda()
    return torch.autograd.Variable(x, requires_grad=requires_grad)

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
