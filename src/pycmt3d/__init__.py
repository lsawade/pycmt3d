from __future__ import (absolute_import, print_function, division)
import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    # Setting up the different ANSI color escape sequences for color terminals:
    # Codes are 3/4-bit codes from here:
    # https://en.wikipedia.org/wiki/ANSI_escape_code
    # For 8-colors: use "\x1b[38;5;<n>m" where <n> is the number of the color.
    # See lightblue
    grey = "\x1b[38;21m"
    lightblue = "\x1b[38;5;81m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Formats The spaces accommodate the different length of the words and
    # amount of detail wanted in the message:
    format_inf = "[%(asctime)s] %(name)s | %(levelname)s     | %(message)s"
    format_war = "[%(asctime)s] %(name)s | %(levelname)s  | %(message)s"
    format_dbg = "[%(asctime)s] %(name)s | %(levelname)s    | %(message)s (%(filename)s:%(lineno)d)"
    format_err = "[%(asctime)s] %(name)s | %(levelname)s    | %(message)s (%(filename)s:%(lineno)d)"
    format_cri = "[%(asctime)s] %(name)s | %(levelname)s | %(message)s (%(filename)s:%(lineno)d)"

    # Create format dictionary
    FORMATS = {
        logging.INFO: format_inf,
        logging.DEBUG: format_dbg,
        logging.WARNING: yellow + format_war + reset,
        logging.ERROR: red + format_err + reset,
        logging.CRITICAL: bold_red + format_cri + reset
    }

    # Initialize with a default logging.Formatter
    def __init__(self):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')

    def format(self, record):

        # Use the logging.LEVEL to get the right formatting
        log_fmt = self.FORMATS.get(record.levelno)

        # Create new formatter with modified timestamp formatting.
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")

        # Return
        return formatter.format(record)

# setup the logger
logger = logging.getLogger("pycmt3d")
logger.setLevel(logging.DEBUG)
logger.propagate = 0

ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

from .source import CMTSource  # NOQA
from .data_container import DataContainer  # NOQA
from .config import WeightConfig, DefaultWeightConfig, Config  # NOQA
from .cmt3d import Cmt3D  # NOQA
from .grid3d import Grid3dConfig, Grid3d  # NOQA
from .inversion import Inversion  # NOQA
