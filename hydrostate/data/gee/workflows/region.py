"""Compatibility imports for callers predating the three regional backends."""
import shutil as shutil

from hydrostate.data.gee.client import retry as retry
from hydrostate.data.gee.products import normalize_pixels as normalize_pixels
from hydrostate.data.gee.selection import epoch as epoch
from hydrostate.data.gee.selection import select_window as select_window
from hydrostate.data.regional.common import assemble_inputs as assemble_inputs
from hydrostate.data.regional.gee_drive import prepare_region_drive as prepare_region_drive
from hydrostate.data.regional.gee_pixels import SceneCache as SceneCache
from hydrostate.data.regional.gee_pixels import prepare_date as prepare_date
