r'''
SpaHDmap
'''

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from SpaHDmap import data, utils, model
from .train import Mapper
from .data import STData, prepare_stdata, select_svgs

name = "SpaHDmap"
__version__ = version(name)
__author__ = "Kun Qian, Junjie Tang"
