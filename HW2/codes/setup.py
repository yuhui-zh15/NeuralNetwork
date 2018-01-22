from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
import numpy

extensions = [
  Extension('imgutil', ['imgutil.pyx'],
    include_dirs = [numpy.get_include()]),
]

setup(
    ext_modules = cythonize(extensions),
)
