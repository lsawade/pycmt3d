#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for source inversion and grid search. Combines cmt3d and grid3d

:copyright:
    Lucas Sawade (lsawade@princeton.edu), 2020
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

Last Update: January 2020

"""

import copy
import os

from . import logger
from .cmt3d import Cmt3D
from .source import CMTSource
from .grid3d import Grid3d
from .data_container import DataContainer
from .plot_util import PlotInvSummary
import matplotlib.pyplot as plt


class Inversion(object):

    def __init__(self, cmtsource: CMTSource,
                 data_container: DataContainer,
                 cmt3d_config: Cmt3D,
                 grid3d_config: Grid3d):
        """Class to run full inversion and generate output.

        :param cmt3d: cmt3d inversion class for the 6 moment tensor parameters
                      and the location (lat, lon, depth)
        :param grid3d: grid3d grid search class for the grid search for
                       scalar moment M0 and timeshift in centroid moment time.

        """

        self.cmtsource = cmtsource
        self.data_container = data_container
        self.cmt3d_config = cmt3d_config
        self.grid3d_config = grid3d_config

        # Things to be computed
        self.new_cmtsource = None

    def source_inversion(self):
        """Uses the different classes to both invert for the parameters and
        grid_search."""

        # Invert for parameters
        self.cmt3d = Cmt3D(self.cmtsource, self.data_container,
                           self.cmt3d_config)
        self.cmt3d.source_inversion()
        self.cmt3d.compute_new_syn()

        # Grid search for CMT shift and
        self.grid3d = Grid3d(self.cmt3d.new_cmtsource, self.data_container,
                             self.grid3d_config)
        self.grid3d.grid_search()
        self.grid3d.prepare_new_cmtsource()

        self.new_cmtsource = copy.deepcopy(self.grid3d.new_cmtsource)

    def plot_summary(self, outputdir=".", figure_format="pdf",
                     mode="global"):
        """
        Plot inversion summary, including source parameter change,
        station distribution, and beach ball change.

        :param outputdir: output directory
        :return:
        """
        eventname = self.cmtsource.eventname
        npar = self.cmt3d_config.npar
        if self.cmt3d_config.double_couple:
            suffix = "ZT_DC"
        elif self.cmt3d_config.zero_trace:
            suffix = "ZT"
        else:
            suffix = "no_constraint"
        outputfn = "%s.%dp_%s.inv" % (eventname, npar, suffix)
        outputfn = os.path.join(outputdir, outputfn)
        figurename = outputfn + "." + figure_format

        logger.info("Source inversion summary figure: %s" % figurename)

        # Fix time shift stats
        mean = copy.deepcopy(self.cmt3d.par_mean)
        mean[9] = self.grid3d.t00_mean
        std = copy.deepcopy(self.cmt3d.par_std)
        std[9] = self.grid3d.t00_std

        M0_stats = [self.grid3d.m00_mean, self.grid3d.m00_std]

        plot_util = PlotInvSummary(
            data_container=self.data_container, config=self.cmt3d_config,
            cmtsource=self.cmtsource,
            nregions=self.cmt3d_config.weight_config.azi_bins,
            new_cmtsource=self.new_cmtsource,
            bootstrap_mean=mean,
            bootstrap_std=std,
            M0_stats=M0_stats,
            var_reduction=self.grid3d.var_reduction,
            mode=mode, grid3d=self.grid3d)
        plot_util.plot_inversion_summary(figurename=figurename)
