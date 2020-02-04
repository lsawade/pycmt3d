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
import numpy as np

from . import logger
from .config import Config
from .cmt3d import Cmt3D
from .source import CMTSource
from .grid3d import Grid3d
from .grid3d import Grid3dConfig
from .gradient3d_mpi import Gradient3d
from .gradient3d_mpi import Gradient3dConfig
from .data_container import DataContainer
from .plot_util import PlotInvSummary
from .plot_util import plot_seismograms
import matplotlib.pyplot as plt
from .util import dump_json


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
            self.G = Grid3d(self.cmt3d.new_cmtsource, self.data_container,
                                self.mt_config)
            self.G.grid_search()


            self.new_cmtsource = copy.deepcopy(self.G.new_cmtsource)

        elif type(self.mt_config) == Gradient3dConfig:
            self.grid3d = None
            self.G = Gradient3d(self.cmt3d.new_cmtsource, self.data_container, 
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
        Plot inversion summary, including source parameter change,if self.config.origin_time_inv:
                
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

        plot_util = PlotInvSummary(
            data_container=self.data_container, config=self.cmt3d_config,
            cmtsource=self.cmtsource,
            nregions=self.cmt3d_config.weight_config.azi_bins,
            new_cmtsource=self.new_cmtsource,
            bootstrap_mean=self.cmt3d.par_mean,
            bootstrap_std=self.cmt3d.par_std,
            var_reduction=self.G.var_reduction,
            mode=mode, G=self.G)
        plot_util.plot_inversion_summary(figurename=figurename)


    def write_summary_json(self, outputdir=".", mode="global"):
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

        logger.info("Source inversion summary file: %s" % filename)

        outdict = dict()

        outdict["oldcmt"] = self.cmtsource.__dict__
        outdict["newcmt"] = self.new_cmtsource.__dict__
        print(type([window.latitude for window in self.data_container.trwins]))
        outdict["sta_lat"] = np.array([window.latitude for window in self.data_container.trwins]).tolist()
        outdict["sta_lon"] = np.array([window.longitude for window in self.data_container.trwins]).tolist()
        outdict["nwindows"] = self.data_container.nwindows
        outdict["nwin_on_trace"] = np.array([window.nwindows for window in self.data_container.trwins]).tolist()

        outdict["bootstrap_mean"] = self.cmt3d.par_mean.tolist()
        outdict["bootstrap_std"] = self.cmt3d.par_std.tolist()

        outdict["nregions"] = self.cmt3d_config.weight_config.azi_bins

        outdict["var_reduction"] = self.G.var_reduction
    
        outdict["G"] = {"method": self.G.config.method,
                        "tshift": self.G.t00_best,
                        "ascale": self.G.m00_best,
                        "bootstrap_mean": self.G.bootstrap_mean.tolist(),
                        "bootstrap_std": self.G.bootstrap_std.tolist(),
                        "chi_list": self.G.chi_list,
                        "meancost_array": self.G.meancost_array.tolist(),
                        "stdcost_array": self.G.stdcost_array.tolist(),
                        "maxcost_array": self.G.maxcost_array.tolist(),
                        "mincost_array": self.G.mincost_array.tolist()}
        outdict["config"] = {"envelope_coef": self.cmt3d_config.envelope_coef,
                             "npar": self.cmt3d_config.npar, 
                             "zero_trace": self.cmt3d_config.zero_trace,
                             "double_couple": self.cmt3d_config.double_couple,
                             "station_correction": self.cmt3d_config.station_correction,
                             "damping": self.cmt3d.config.damping,
                             "weight_config": 
                             {"normalize_by_energy": self.cmt3d_config.weight_config.normalize_by_energy,
                             "normalize_by_category": self.cmt3d_config.weight_config.normalize_by_category}}
        outdict["mode"] = mode

        dump_json(outdict, filename)
