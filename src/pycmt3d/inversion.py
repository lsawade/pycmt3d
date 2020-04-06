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
from .gradient3d import Gradient3dConfig
from .gradient3d import Gradient3d
from .data_container import DataContainer
from .plot_util import PlotInvSummary
from .plot_util import plot_seismograms
from .util import dump_json
from .util import get_trwin_tag
from collections import defaultdict


class Inversion(object):

    def __init__(self, cmtsource: CMTSource,
                 data_container: DataContainer,
                 cmt3d_config: Config,
                 mt_config=None):

        """Class to run full inversion and generate output.

        :param cmt3d: cmt3d inversion class for the 6 moment tensor parameters
                      and the location (lat, lon, depth)
        :param mt_config: config for method to find timeshift and
                          scalar moment.
                          If set to None no improved scalar moment and
                          gradient is found. Default None.

        """

        self.cmtsource = cmtsource
        self.data_container = data_container
        self.cmt3d_config = cmt3d_config
        self.mt_config = mt_config

        # Things to be computed
        self.new_cmtsource = None

        # Variance Reduction
        self.var_reduction = None

        # Statistics
        self.stats = None

    def source_inversion(self):
        """Uses the different classes to both invert for the parameters and
        grid_search."""

        # Invert for parameters
        self.cmt3d = Cmt3D(self.cmtsource, self.data_container,
                           self.cmt3d_config)
        self.cmt3d.source_inversion()
        self.cmt3d.compute_new_syn()

        if type(self.mt_config) == Grid3dConfig:
            # Grid search for CMT shift and
            self.G = Grid3d(self.cmt3d.new_cmtsource, self.data_container,
                            self.mt_config)
            self.G.grid_search()

            self.new_cmtsource = copy.deepcopy(self.G.new_cmtsource)

            self.var_reduction = self.G.var_reduction

        elif type(self.mt_config) == Gradient3dConfig:
            self.grid3d = None

            self.G = Gradient3d(
                self.cmt3d.new_cmtsource,
                self.data_container,
                self.mt_config)

            self.G.search()
            self.new_cmtsource = copy.deepcopy(self.G.new_cmtsource)

            self.var_reduction = self.G.var_reduction

        else:
            self.G = None
            self.new_cmtsource = copy.deepcopy(self.cmt3d.new_cmtsource)
            self.var_reduction = self.cmt3d.var_reduction

        # Extract stats to save to json
        self.extract_stats()  # Creates self.stats [dict]

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
        if self.config.origin_time_inv:

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
        mean[9] = 1  # self.grid3d.t00_mean
        std = copy.deepcopy(self.cmt3d.par_std)
        std[9] = 1  # self.grid3d.t00_std

        plot_util = PlotInvSummary(
            data_container=self.data_container, config=self.cmt3d_config,
            cmtsource=self.cmtsource,
            nregions=self.cmt3d_config.weight_config.azi_bins,
            new_cmtsource=self.new_cmtsource,
            bootstrap_mean=self.cmt3d.par_mean,
            bootstrap_std=self.cmt3d.par_std,
            var_reduction=self.var_reduction,
            mode=mode, G=self.G)
        plot_util.plot_inversion_summary(figurename=figurename)

    def extract_metadata(self, cat_name, meta_varname):
        data_old = []
        data_new = []
        cat_data = self.metas_sort[cat_name]
        for meta in cat_data:
            data_old.extend(meta.prov["synt"][meta_varname])
            data_new.extend(meta.prov["new_synt"][meta_varname])

        return data_old, data_new

    def extract_stats(self):
        """This function uses the info save in the metas to compute
        the component means, stds, mins, and maxs."""

        # Define list of statistics to get.
        vtype_list = ["tshift", "cc", "power_l1", "power_l2", "cc_amp", "chi"]

        # Sort the metadata into categories (040_100.obsd.BHZ is a category eg)
        self.sort_metas()

        # Create empty dictionary for the statistics.
        self.stats = dict()

        # Get category names from the sorted meta dictionary
        cat_names = sorted(self.metas_sort.keys())

        # Loop over categories and compute statistics for the
        for irow, cat in enumerate(cat_names):

            self.stats[cat] = self.get_stats_one_cat(cat, vtype_list)

    def get_stats_one_cat(self, cat_name, vtype_list):
        cat_dict = dict()
        for var_idx, varname in enumerate(vtype_list):

            # Collect the raw data from new and old synthetics.
            data_before, data_after = \
                self.extract_metadata(cat_name, varname)

            # Compute before/after dictionary for the the varname (cc eg.)
            cat_dict[varname] = self.get_stats_one_entry(data_before,
                                                         data_after)

        return cat_dict

    @staticmethod
    def get_stats_one_entry(data_before, data_after):
        # Stats
        a_mean = np.mean(data_after)
        a_std = np.std(data_after)
        b_mean = np.mean(data_before)
        b_std = np.std(data_before)

        return {"before": {"mean": b_mean, "std": b_std},
                "after": {"mean": a_mean, "std": a_std}}

    def sort_metas(self):
        """sort metas into different categories for future plotting """
        metas_sort = defaultdict(list)
        key_map = defaultdict(set)

        # Get the computed meta information.
        if self.G is None:
            metas = self.cmt3d.metas
        else:
            metas = self.G.metas

        for trwin, meta in zip(self.data_container, metas):
            comp = trwin.channel
            cat_name = get_trwin_tag(trwin)
            key_map[comp].add(cat_name)
            metas_sort[cat_name].append(meta)

        self.metas_sort = metas_sort
        self.key_map = key_map

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
        outdict["sta_lat"] = np.array([window.latitude
                                       for window
                                       in self.data_container.trwins]).tolist()
        outdict["sta_lon"] = np.array([window.longitude
                                       for window
                                       in self.data_container.trwins]).tolist()
        outdict["nwindows"] = self.data_container.nwindows
        outdict["nwin_on_trace"] = np.array([window.nwindows
                                             for window
                                             in self.data_container.trwins]
                                            ).tolist()

        outdict["bootstrap_mean"] = self.cmt3d.par_mean.tolist()
        outdict["bootstrap_std"] = self.cmt3d.par_std.tolist()

        outdict["nregions"] = self.cmt3d_config.weight_config.azi_bins

        outdict["stats"] = self.stats

        if self.G is not None:
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
        else:
            outdict["var_reduction"] = self.cmt3d.var_reduction
            outdict["G"] = None

        outdict["config"] = {"envelope_coef": self.cmt3d_config.envelope_coef,
                             "npar": self.cmt3d_config.npar,
                             "zero_trace": self.cmt3d_config.zero_trace,
                             "double_couple": self.cmt3d_config.double_couple,
                             "station_correction":
                             self.cmt3d_config.station_correction,
                             "damping": self.cmt3d.config.damping,
                             "weight_config":
                             {"normalize_by_energy":
                              self.cmt3d_config.weight_config
                              .normalize_by_energy,
                              "normalize_by_category":
                              self.cmt3d_config.weight_config
                              .normalize_by_category}}
        outdict["mode"] = mode

        dump_json(outdict, filename)
