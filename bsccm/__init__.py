"""Top-level package for bsccm."""
name = "bsccm"

__author__ = """Henry Pinkard"""
__email__ = 'henry.pinkard@gmail.com'

from bsccm.bsccm import BSCCM
from bsccm.phase.util import *
from bsccm.phase.functional_dpc import *
from ._version import __version__, version_info