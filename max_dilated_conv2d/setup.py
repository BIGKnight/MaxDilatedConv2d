from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
setup(name='max_dilated_conv2d',
      ext_modules=[CUDAExtension('max_dilated_conv2d_gpu', ['max_dilated_conv2d.cpp', 'max_dilated_conv2d_cuda.cu']),],
      cmdclass={'build_ext': BuildExtension})
