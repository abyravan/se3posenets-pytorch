import os
import torch
from torch.utils.ffi import create_extension

# Declare CPU sources
sources = ['src/ntfm3d_cpu.c',
           'src/ntfm3ddelta_cpu.c',
           'src/project3dpts_cpu.c',
           'src/wt3dtfmloss_cpu.c',
           'src/wt3dtfmnormloss_cpu.c']
headers = ['src/ntfm3d_cpu.h',
           'src/ntfm3ddelta_cpu.h',
		   'src/project3dpts_cpu.h',
           'src/wt3dtfmloss_cpu.h',
           'src/wt3dtfmnormloss_cpu.h']
defines = []
with_cuda = False

# Declare GPU sources
extra_objects = None
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/ntfm3d_cuda.c',
                'src/ntfm3ddelta_cuda.c',
				'src/project3dpts_cuda.c',
                'src/wt3dtfmloss_cuda.c']
    headers += ['src/ntfm3d_cuda.h',
                'src/ntfm3ddelta_cuda.h',
                'src/project3dpts_cuda.h',
                'src/wt3dtfmloss_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

	# Get the pre-compiled CUDA kernels (currently all CUDA code has to be compiled apriori)
    this_file = os.path.dirname(os.path.realpath(__file__))
    extra_objects = ['src/cuda/lib/ntfm3d_kernel.cu.o',
                     'src/cuda/lib/ntfm3ddelta_kernel.cu.o',
					 'src/cuda/lib/project3dpts_kernel.cu.o',
                     'src/cuda/lib/wt3dtfmloss_kernel.cu.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# Setup the overall compilation
ffi = create_extension(
    '_ext.se3layers',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

# Compile
if __name__ == '__main__':
    ffi.build()
