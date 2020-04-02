from __future__ import (absolute_import, print_function, division)
import logging


class CustomFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors

    This class organizes the customization of the logging output.
    The formatter as of now outputs the logs in the following manner in
    order of Loglevel:

    .. code-block:: python

        [2020-03-30 21:00:17] matpy -- [  DEBUG   ] \
                        Test Verbose Level (matrixmultiplication.py:53)
        [2020-03-30 21:00:17] matpy -- [ VERBOSE  ] \
                        Test Verbose Level
        [2020-03-30 21:00:17] matpy -- [  INFO    ] \
                        Initializing matrices...
        [2020-03-30 21:00:17] matpy -- [ WARNING  ] \
                        Matrix size exceeds 4 elements.
        [2020-03-30 21:00:17] matpy -- [  ERROR   ] \
                        Test Error Level (matrixmultiplication.py:54)
        [2020-03-30 21:00:17] matpy -- [ CRITICAL ] \
                        Test Critical Level (matrixmultiplication.py:55)


    These outputs are colored in the actual output but the formatting is just
    as shown above. VERBOSE is an extra added LogLevel formatting. More can be
    added below the comment `EXTRA LEVELS` in the same way the VERBOSE
    is added.

    The variable VERBOSE is given at the top of the module. That way it can be
    changed for all depending function

    """

    # Setting up the different ANSI color escape sequences for color terminals:
    # Codes are 3/4-bit codes from here:
    # https://en.wikipedia.org/wiki/ANSI_escape_code
    # For 8-colors: use "\x1b[38;5;<n>m" where <n> is the number of the color.
    grey = "\x1b[38;21m"
    green = "\x1b[38;5;64m"
    light_grey = "\x1b[38;5;240m"
    dark_blue = "\x1b[38;5;25m"
    light_blue = "\x1b[38;5;69m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    dark_red = "\x1b[38;5;97m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Formats The spaces accommodate the different length of the words and
    # amount of detail wanted in the message:
    time_fmt = light_grey + "[%(asctime)s]" + reset
    name_fmt = "-- %(name)s -"
    pre_fmt = time_fmt + " " + name_fmt

    debug_fmt = "--- [" + light_blue + "%(levelname)s" + reset + "]:" \
                + light_blue + " %(message)s (%(filename)s:%(lineno)d)" + reset
    info_fmt = "---- [%(levelname)s]: %(message)s"
    warning_fmt = "- [" + yellow + "%(levelname)s" + reset + "]:" \
                  + yellow + " %(message)s" + reset
    error_fmt = "--- [" + red + "%(levelname)s" + reset + "]:" \
                + red + " %(message)s (%(filename)s:%(lineno)d)" + reset
    critical_fmt = " [" + bold_red + "%(levelname)s" + reset + "]:" \
                   + bold_red + " %(message)s (%(filename)s:%(lineno)d)" \
                   + reset

    # Create format dictionary
    FORMATS = {
        logging.DEBUG: pre_fmt + debug_fmt,
        logging.INFO: pre_fmt + info_fmt,
        logging.WARNING: pre_fmt + warning_fmt,
        logging.ERROR: pre_fmt + error_fmt,
        logging.CRITICAL: pre_fmt + critical_fmt
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
logger.setLevel(logging.INFO)
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
