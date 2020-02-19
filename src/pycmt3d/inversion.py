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
from .config import Config
from .cmt3d import Cmt3D
from .source import CMTSource
from .grid3d import Grid3d, Grid3dConfig
from .gradient3d import Gradient3d, Gradient3dConfig
from .data_container import DataContainer
from .plot_util import PlotInvSummary, plot_seismograms
import matplotlib.pyplot as plt


class Inversion(object):

    def __init__(self, cmtsource: CMTSource,
                 data_container: DataContainer,
                 cmt3d_config: Config,
                 mt_config=None):
        """Class to run full inversion and generate output.

        :param cmt3d: cmt3d inversion class for the 6 moment tensor parameters
                      and the location (lat, lon, depth)
        :param mt_config: config for method to find timeshift and scalar moment

        """

        self.cmtsource = cmtsource
        self.data_container = data_container
        self.cmt3d_config = cmt3d_config
        self.mt_config = mt_config

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
        #self.cmt3d.plot_new_synt_seismograms(
        # "/home/lsawade/pycmt3d/seis/cmt3d")

        if type(self.mt_config) == Grid3dConfig:
            # Grid search for CMT shift and
            self.grid3d = Grid3d(self.cmt3d.new_cmtsource, self.data_container,
                                self.grid3d_config)
            self.grid3d.grid_search()

            self.new_cmtsource = copy.deepcopy(self.grid3d.new_cmtsource)

        elif type(self.mt_config) == Gradient3dConfig:
            
            self.G = Gradient3d(self.cmtsource, self.data_container, 
                                self.mt_config)
            self.G.search()
            self.new_cmtsource = copy.deepcopy(self.G.new_cmtsource)

    def plot_new_synt_seismograms(self, outputdir, figure_format="pdf"):
        """
        Plot the new synthetic waveform
        """
        plot_seismograms(self.data_container, outputdir,
                         self.cmtsource, figure_format=figure_format)

    def write_new_cmtfile(self, outputdir="."):
        """
        Write new_cmtsource into a file
        """
        if self.cmt3d_config.double_couple:
            suffix = "ZT_DC"
        elif self.cmt3d_config.zero_trace:
            suffix = "ZT"
        else:
            suffix = "no_constraint"
        outputfn = "%s.%dp_%s.inv" % (
            self.cmtsource.eventname, self.cmt3d_config.npar, suffix)
        cmtfile = os.path.join(outputdir, outputfn)
        logger.info("New cmt file: %s" % cmtfile)

        self.new_cmtsource.write_CMTSOLUTION_file(cmtfile)

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
        mean[9] = 1 #self.grid3d.t00_mean
        std = copy.deepcopy(self.cmt3d.par_std)
        std[9] = 1#self.grid3d.t00_std

        M0_stats = [1, 1] #[self.grid3d.m00_mean, self.grid3d.m00_std]

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


    def write_summary_json(self, outputdir="."):
        """This function uses all computed statistics and outputs a json
        file. Content will include the statistics table.
        cost reduction in. """
        
        eventname = self.cmtsource.eventname
        npar = self.cmt3d_config.npar
        if self.cmt3d_config.double_couple:
            suffix = "ZT_DC"
        elif self.cmt3d_config.zero_trace:
            suffix = "ZT"
        else:
            suffix = "no_constraint"
        outputfn = "%s.%dp_%s.stats.json" % (eventname, npar, suffix)
        outputfn = os.path.join(outputdir, outputfn)
        filename = outputfn

        logger.info("Source inversion summary file: %s" % figurename)

        outdict = dict()

        outdict["oldcmt"] = self.cmtsource.__dict__
        outdict["newcmt"] = self.new_cmtsource.__dict__

        outdict["stations"] = self.data_container[]

        