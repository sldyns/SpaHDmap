r'''
SpaHDmap
'''

# Prevent BLAS multi-threading deadlock issues
# This is especially important for NMF computation which uses scipy/numpy BLAS operations
# The deadlock can occur in both Jupyter and terminal environments
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from SpaHDmap import data, utils, model
from .train import Mapper
from .data import STData, prepare_stdata, select_svgs

name = "SpaHDmap"
__version__ = '0.1.5'
__author__ = "Kun Qian, Junjie Tang"
