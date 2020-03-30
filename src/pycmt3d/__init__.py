from __future__ import (absolute_import, print_function, division)
import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

# setup the logger
logger = logging.getLogger("pycmt3d")
logger.setLevel(logging.INFO)
logger.propagate = 0

ch = logging.StreamHandler()
# Add formatter
# FORMAT = "%(name)s - %(levelname)s: %(message)s"
# formatter = logging.Formatter(FORMAT)

#Add custom formatter for color in the terminal
ch.setFormatter(ch.setFormatter(CustomFormatter()))
logger.addHandler(ch)

from .source import CMTSource  # NOQA
from .data_container import DataContainer  # NOQA
from .config import WeightConfig, DefaultWeightConfig, Config  # NOQA
from .cmt3d import Cmt3D  # NOQA
from .grid3d import Grid3dConfig, Grid3d  # NOQA
from .inversion import Inversion  # NOQA
