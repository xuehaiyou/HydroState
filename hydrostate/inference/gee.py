"""Compatibility import for the structured acquisition workflow."""
import sys
from hydrostate.data.gee.workflows import region as _implementation
sys.modules[__name__] = _implementation
