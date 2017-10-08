import multiprocessing
import signal
import sys
import traceback
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import collections

# Call this to initialize the loading processes.
def start_pool(nproc):
    global proc_pool
    proc_pool = MultiProcMapper(nproc=nproc, use_pool=True)

#######################################
########## Default collate function from torch data loader
if sys.version_info[0] == 2:
    string_classes = basestring
else:
    string_classes = (str, bytes)

_use_shared_memory = False
"""Whether to use shared memory in default_collate"""

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

#######################################
#### Custom data loader, wraps around load_func
class CustomDataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, the ``shuffle`` argument is ignored.
        collate_fn (callable, optional)
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If False and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 collate_fn=default_collate, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.use_pool = True

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        elif not shuffle:
            self.sampler = SequentialSampler(dataset)

    def __next__(self):
        if self.drop_last and self.samples_remaining < self.batch_size:
            raise StopIteration
        if self.samples_remaining == 0:
            raise StopIteration
        indices = self._next_indices()
        if self.use_pool:
            global proc_pool
            batch = proc_pool.map(self.dataset, indices)
        else:
            batch = [self.dataset[element] for element in indices]
        return self.collate_fn(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        self.samples_remaining = len(self.sampler) # Reset sampler!
        self.sample_iter = iter(self.sampler) # Reset sampler!
        return self

    def _next_indices(self):
        batch_size = min(self.samples_remaining, self.batch_size)
        batch = [next(self.sample_iter) for _ in range(batch_size)]
        self.samples_remaining -= len(batch)
        return batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


################## INTERNAL FUNCTIONS ###########################

import dill

def multiproc_obj_dill(args):
    try:
        obj, id = dill.loads(args)
        ret = obj[id]
        return ret
    except:
        raise Exception("\n===================\n" + "".join(traceback.format_exception(*sys.exc_info())))

# def apply_async(pool, obj, args):
#     for arg in args:
#         payload = dill.dumps((obj, arg))
#         return pool.apply_async(multiproc_obj_dill, (payload,))

def combine_obj_and_args(obj, args):
    ret = []
    for arg in args:
        ret.append(dill.dumps((obj, arg)))
    return ret
#
# def multiproc_obj(args):
#     try:
#         obj  = args[0]
#         id   = args[1]
#         print('B: ', id); sys.stdout.flush();
#         ret = obj[id]
#         return ret
#     except:
#         raise Exception("\n===================\n" + "".join(traceback.format_exception(*sys.exc_info())))

class MultiProcMapper:
    def __init__(self, nproc=multiprocessing.cpu_count(), init_fcn=None, use_pool=True, maxtasksperchild=None):
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        if use_pool:
            self.pool = multiprocessing.Pool(nproc, initializer=init_fcn, maxtasksperchild=maxtasksperchild)
        else:
            self.pool = multiprocessing.pool.ThreadPool(nproc, initializer=init_fcn, maxtasksperchild=maxtasksperchild)
        signal.signal(signal.SIGINT, original_sigint_handler)

    def __del__(self):
        self.pool.close()

    def map(self, obj, args):
        res = None
        while True:
            try:
                if res is None:
                    res = self.pool.map_async(multiproc_obj_dill,
                        combine_obj_and_args(obj, args))
                ret = res.get(0.1) # Without the timeout this blocking call ignores all signals.
                break
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                self.pool.terminate()
                raise
            except multiprocessing.TimeoutError:
                pass
        return ret