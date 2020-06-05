#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot utils

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

from __future__ import print_function, division, absolute_import
import os
from collections import defaultdict
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from obspy.geodetics import gps2dist_azimuth
from obspy.imaging.beachball import beach
import shapefile
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import argparse
import matplotlib.ticker as mticker
from cartopy.crs import PlateCarree
import cartopy

# From pycmt3d
from .source import CMTSource
from .geo_data_util import GeoMap
from .data_container import DataContainer, TraceWindow
from .grid3d import Grid3d
from .gradient3d import Gradient3d
from . import logger
from .util import get_cmt_par, get_trwin_tag
from .util import load_json
from .measure import _envelope

matplotlib.use('agg')

# earth half circle
EARTH_HC, _, _ = gps2dist_azimuth(0, 0, 0, 180)


class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)


def get_new_locations(lat1, lon1, lat2, lon2, padding):
    """ Get new plodding locations

    :param lat1: Old cmt lat
    :param lon1: Old cmt lon
    :param lat2: New cmt lat
    :param lon2: New cmt lon
    :param az: azimuth
    :param padding: padding
    :return: newlat1, newlon1, newlat2, newlon2
    """

    # Get aziumth
    _, az, _ = gps2dist_azimuth(lat1, lon1,
                                lat2, lon2)

    # length of corner placement:
    dx = 1.2 * padding

    if 0 <= az < 90:
        newlat1 = lat1 - dx
        newlon1 = lon1 - dx
        newlat2 = lat1 + dx
        newlon2 = lon1 + dx

    elif 90 <= az < 180:
        newlat1 = lat1 - dx
        newlon1 = lon1 + dx
        newlat2 = lat1 + dx
        newlon2 = lon1 - dx

    elif 180 <= az < 270:
        newlat1 = lat1 + dx
        newlon1 = lon1 + dx
        newlat2 = lat1 - dx
        newlon2 = lon1 - dx

    else:
        newlat1 = lat1 + dx
        newlon1 = lon1 - dx
        newlat2 = lat1 - dx
        newlon2 = lon1 + dx

    return newlat1, newlon1, newlat2, newlon2


def _plot_new_seismogram_sub(trwin: TraceWindow, outputdir,
                             cmtsource, figure_format, suffix=None):
    obsd = trwin.datalist['obsd']
    synt = trwin.datalist['synt']
    new_synt = trwin.datalist['new_synt']

    tag = trwin.tags['obsd']
    station = obsd.stats.station
    network = obsd.stats.network
    channel = obsd.stats.channel
    location = obsd.stats.location

    if suffix is None:
        outputfig = os.path.join(outputdir, "%s.%s.%s.%s.%s.%s" % (
            tag, network, station, location, channel, figure_format))
    else:
        outputfig = os.path.join(outputdir, "%s.%s.%s.%s.%s.%s.%s" % (
            tag, network, station, location, channel, suffix, figure_format))

    # Times and offsets computed individually, since the grid search applies
    # a timeshift which changes the times of the traces.
    if cmtsource is None:
        offset = 0
    else:
        offset = obsd.stats.starttime - cmtsource.cmt_time
        offset_synt = synt.stats.starttime - cmtsource.cmt_time
        offset_new = new_synt.stats.starttime - cmtsource.cmt_time
    times = [offset + obsd.stats.delta * i for i in range(obsd.stats.npts)]
    times_synt = [offset_synt + synt.stats.delta * i
                  for i in range(synt.stats.npts)]
    times_new = [offset_new + new_synt.stats.delta * i
                 for i in range(new_synt.stats.npts)]
    fig = plt.figure(figsize=(15, 5))

    plt.rcParams.update({'font.size': 13,
                         'lines.linewidth': 1.5})

    # plot seismogram
    ax1 = plt.subplot(211)
    ax1.plot(times, obsd.data, color="black", linewidth=0.8, alpha=0.6,
             label="obsd")
    ax1.plot(times_synt, synt.data, color="red", linewidth=1,
             label="synt")
    ax1.plot(times_new, new_synt.data, color="blue", linewidth=1,
             label="new synt")
    ax1.set_xlim(times[0], times[-1])
    ax1.legend(loc='upper right', frameon=False, ncol=3, prop={'size': 11})

    # Setting top left corner text manually
    fontsize = 11
    text = "Network: %2s    Station: %s\n" \
           "Location: %2s  Channel: %3s" \
           % (network, station, location, channel)
    if trwin.velocity:
        text += "\nV"
    ax1.text(0.005, 0.95, text,
             fontsize=fontsize,
             transform=ax1.transAxes,
             horizontalalignment="left", verticalalignment="top")

    for win in trwin.windows:
        left = win[0] + offset
        right = win[1] + offset
        re = Rectangle((left, plt.ylim()[0]), right - left,
                       plt.ylim()[1] - plt.ylim()[0], color="blue",
                       alpha=0.25)
        plt.gca().add_patch(re)

    # plot envelope
    ax2 = plt.subplot(212)
    ax2.plot(times, _envelope(obsd.data), color="black", linewidth=0.8,
             alpha=0.6, label="obsd")
    ax2.plot(times, _envelope(synt.data), color="red", linewidth=1,
             label="synt")
    ax2.plot(times, _envelope(new_synt.data), color="blue", linewidth=1,
             label="new synt")
    ax2.set_xlim(times[0], times[-1])

    ax2.set_xlabel("Time [s]", fontsize=13)

    for win in trwin.windows:
        left = win[0] + offset
        right = win[1] + offset
        re = Rectangle((left, plt.ylim()[0]), right - left,
                       plt.ylim()[1] - plt.ylim()[0], color="blue",
                       alpha=0.25)
        plt.gca().add_patch(re)

    logger.info("output figname: %s" % outputfig)
    ax2.legend(loc='upper right', frameon=False, ncol=3, prop={'size': 11})
    plt.savefig(outputfig)
    plt.close(fig)


def plot_seismograms(data_container: DataContainer, outputdir, cmtsource=None,
                     figure_format="png", suffix=None):
    """
    Plot the new synthetic and old synthetic data together with data.
    So we can see the how the seimogram changes after inversion.
    """
    # make a check
    if 'new_synt' not in data_container.trwins[0].datalist.keys():
        return "New synt not generated...Can't plot"

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    logger.info("Plotting observed, synthetics and windows to dir: %s"
                % outputdir)
    for trwin in data_container:
        _plot_new_seismogram_sub(trwin, outputdir, cmtsource,
                                 figure_format, suffix=suffix)


class PlotStats(object):
    """ plot histogram utils"""

    def __init__(self, data_container: DataContainer, metas, outputfn):
        self.data_container = data_container
        self.metas = metas
        self.outputfn = outputfn

        self.metas_sort = None
        self.key_map = None

    def sort_metas(self):
        metas_sort = defaultdict(list)
        key_map = defaultdict(set)
        """ sort metas into different categories for future plotting """
        for trwin, meta in zip(self.data_container, self.metas):
            comp = trwin.channel
            cat_name = get_trwin_tag(trwin)
            key_map[comp].add(cat_name)
            metas_sort[cat_name].append(meta)

        self.metas_sort = metas_sort
        self.key_map = key_map

    @staticmethod
    def plot_stats_histogram_one_entry(pos, cat, vtype, data_b, data_a,
                                       num_bin):
        plt.subplot(pos)
        plt.xlabel(vtype, fontsize=15)
        plt.ylabel(cat, fontsize=15)
        if vtype == "cc":
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
        elif vtype == "chi":
            ax_min = 0.0
            ax_max = max(max(data_b), max(data_a))
        else:
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
            abs_max = max(abs(ax_min), abs(ax_max))
            ax_min = -abs_max
            ax_max = abs_max
        binwidth = (ax_max - ax_min) / num_bin

        # Stats
        a_mean = np.mean(data_a)
        a_std = np.std(data_a)
        b_mean = np.mean(data_b)
        b_std = np.std(data_b)

        nb, _, _ = plt.hist(
            data_b, bins=np.arange(ax_min, ax_max + binwidth / 2., binwidth),
            facecolor='blue', alpha=0.3)
        nb_max = np.max(nb)
        plt.plot([b_mean, b_mean], [0, nb_max], "b--")
        plt.plot([b_mean - b_std, b_mean + b_std],
                 [nb_max / 2, nb_max / 2], "b--")
        na, _, _ = plt.hist(
            data_a, bins=np.arange(ax_min, ax_max + binwidth / 2., binwidth),
            facecolor='green', alpha=0.5)
        na_max = np.max(na)
        plt.plot([a_mean, a_mean], [0, na_max], "g-")
        plt.plot([a_mean - a_std, a_mean + a_std],
                 [na_max / 2, na_max / 2], "g-")

    def extract_metadata(self, cat_name, meta_varname):
        data_old = []
        data_new = []
        cat_data = self.metas_sort[cat_name]
        for meta in cat_data:
            data_old.extend(meta.prov["synt"][meta_varname])
            data_new.extend(meta.prov["new_synt"][meta_varname])

        return data_old, data_new

    def plot_stats_histogram_one_category(
            self, G, irow, cat_name, vtype_list, num_bins, vtype_dict):
        for var_idx, varname in enumerate(vtype_list):
            meta_varname = vtype_dict[varname]
            data_before, data_after = \
                self.extract_metadata(cat_name, meta_varname)
            self.plot_stats_histogram_one_entry(
                G[irow, var_idx], cat_name, varname, data_before, data_after,
                num_bins[var_idx])

    def plot_stats_histogram(self):
        """
        Plot histogram of tshift, cc, power, cc_amplitude_ratio,
        waveform misfit values before and after inversion inside
        windows.

        :return:
        """
        vtype_list = ['time shift', 'cc',
                      'power_l1_ratio(dB)', 'power_l2_ratio(dB)',
                      'CC amplitude ratio(dB)', 'chi']
        num_bins = [15, 15, 15, 15, 15, 15]
        vtype_dict = {'time shift': "tshift",
                      'cc': "cc",
                      "power_l1_ratio(dB)": "power_l1",
                      "power_l2_ratio(dB)": "power_l2",
                      "CC amplitude ratio(dB)": "cc_amp",
                      "chi": "chi"}

        self.sort_metas()
        nrows = len(self.metas_sort.keys())
        ncols = len(vtype_list)

        plt.figure(figsize=(4 * ncols, 4 * nrows))
        G = gridspec.GridSpec(nrows, ncols)

        cat_names = sorted(self.metas_sort.keys())
        for irow, cat in enumerate(cat_names):
            self.plot_stats_histogram_one_category(
                G, irow, cat, vtype_list, num_bins, vtype_dict)
        plt.tight_layout()
        plt.savefig(self.outputfn)
        plt.close()


class PlotInvSummary(object):

    def __init__(self, data_container=None, cmtsource=None, config=None,
                 nregions=12, new_cmtsource=None, bootstrap_mean=None,
                 bootstrap_std=None, var_reduction=0.0,
                 mode="regional", G=None):
        """ Plots inversion summary with all necessary information.

        Args:
            data_container (pycmt3d.data_container.DataContainer or dict):
            The datacontainer is the full fata container used in the inversion
            itself. Defaults to None.
            cmtsource (pycmt3d.source.CMTSource): original cmtsource.
                                                  Defaults to None.
            config (pycmt3d.config.Config, optional): Configuration class
                                                    used for the inversion.
                                                    Defaults to None.
            nregions (int, optional): Number of bins in the station and window
                                    histograms. Defaults to 12.
            new_cmtsource (pycmt3d.source.CMTSource): inverted cmtsolution.
                                                    Defaults to None.
            bootstrap_mean (numpy.ndarray, optional): Bootstrap mean.
                                                      Defaults to None
            bootstrap_std (numpy.ndarray): bootstrap std. Defaults to None.
            M0_stats (numpy.ndarray, optional): statistics on gradient method.
            var_reduction (float, optional): Variance reduction overwrite.
                                             Defaults to 0.0.
            mode (str, optional): Plot mode. Defaults to "regional".
            G (pycmt3d.grid3d.Grid3d or pycmt3d.gradient3d or dict, optional):
                input grid search or gradient data. Defaults to None.
        """

        if type(data_container) == DataContainer:
            self.data_container = data_container
            self.nwindows = data_container.nwindows
            self.sta_lat = [window.latitude
                            for window in self.data_container.trwins]
            self.sta_lon = [window.longitude
                            for window in self.data_container.trwins]
            self.nwin_on_trace = [window.nwindows
                                  for window in self.data_container.trwins]

        elif type(data_container) == dict:
            self.sta_lon = data_container["sta_lon"]
            self.sta_lat = data_container["sta_lat"]
            self.nwindows = data_container["nwindows"]
            self.nwin_on_trace = data_container["nwin_on_trace"]

        else:
            self.sta_lat = None
            self.sta_lon = None
            self.nwindows = None
            self.nwin_on_trace = None

        self.cmtsource = cmtsource
        self.config = config
        self.nregions = nregions

        self.new_cmtsource = new_cmtsource
        self.bootstrap_std = bootstrap_std
        self.bootstrap_mean = bootstrap_mean
        self.var_reduction = var_reduction

        if mode.lower() not in ["global", "regional"]:
            raise ValueError("Plot mode: 1) global; 2) regional")
        self.mode = mode.lower()

        self.sta_dist = []
        # azimuth in degree unit
        self.sta_azi = []
        # azimuth in radius unit
        self.sta_theta = []
        self.prepare_array()

        # Loading G from the grid search
        if type(G) in [Gradient3d, dict]:
            if type(G) == Gradient3d:
                self.G = G
            elif type(G) == dict:
                self.G = Struct(**G)
                for att in self.G.__dict__.keys():
                    if isinstance(getattr(self.G, att), list):
                        setattr(self.G, att,
                                np.array(getattr(self.G, att)))
            # Fix time
            if not np.isnan(self.G.bootstrap_mean[0]):
                self.bootstrap_mean[9] = self.G.bootstrap_mean[1]
                self.bootstrap_std[9] = self.G.bootstrap_std[1]

                # Fix moment
                self.bootstrap_mean = np.append(self.bootstrap_mean,
                                                self.G.bootstrap_mean[0]
                                                * self.cmtsource.M0)
                self.bootstrap_std = np.append(self.bootstrap_std,
                                               self.G.bootstrap_std[0]
                                               * self.cmtsource.M0)
            else:
                # Fix time
                self.bootstrap_mean[9] = np.nan
                self.bootstrap_std[9] = np.nan

                # Fix moment
                self.bootstrap_mean = np.append(self.bootstrap_mean, np.nan)
                self.bootstrap_std = np.append(self.bootstrap_std, np.nan)

        elif type(G) == Grid3d:

            self.G = G

            # Fix time shift
            self.bootstrap_mean[9] = np.nan
            self.bootstrap_std[9] = np.nan

            # Fix moment
            self.bootstrap_mean = np.append(self.bootstrap_mean, np.nan)
            self.bootstrap_std = np.append(self.bootstrap_std, np.nan)

        else:
            self.G = G
            # Fix time shift
            self.bootstrap_mean[9] = np.nan
            self.bootstrap_std[9] = np.nan

            # Fix moment
            self.bootstrap_mean = np.append(self.bootstrap_mean, np.nan)
            self.bootstrap_std = np.append(self.bootstrap_std, np.nan)

    @classmethod
    def from_JSON(cls, cmt3d_file, g3d_file=None):

        d = load_json(os.path.abspath(cmt3d_file))
        data_container = {"sta_lon": np.array(d["sta_lon"]),
                          "sta_lat": np.array(d["sta_lat"]),
                          "nwindows": d["nwindows"],
                          "nwin_on_trace": d["nwin_on_trace"]}
        cmtsource = CMTSource.from_dictionary(d["oldcmt"])
        new_cmtsource = CMTSource.from_dictionary(d["newcmt"])
        config = Struct(**d["config"])
        nregions = d["nregions"]
        bootstrap_mean = np.array(d["bootstrap_mean"])
        bootstrap_std = np.array(d["bootstrap_std"])
        var_reduction = d["var_reduction"]
        mode = d["mode"]

        if g3d_file is not None:
            g3d = load_json(os.path.abspath(g3d_file))
            G = g3d["G"]
            G["config"] = g3d["config"]
            new_cmtsource = CMTSource.from_dictionary(g3d["newcmt"])

        # final scaling and shift
        # misfit reduction
        # bootstrap misfit reduction min
        # bootstrap misfit reduction max
        # bootstrap misfit reduction std
        # bootstrap misfit reduction mean

        return cls(data_container=data_container, cmtsource=cmtsource,
                   config=config, nregions=nregions,
                   new_cmtsource=new_cmtsource,
                   bootstrap_mean=bootstrap_mean,
                   bootstrap_std=bootstrap_std,
                   var_reduction=var_reduction,
                   mode=mode,
                   G=G)

    def prepare_array(self):
        # station

        for sta_lat, sta_lon in zip(self.sta_lat, self.sta_lon):
            dist, az, baz = gps2dist_azimuth(self.cmtsource.latitude,
                                             self.cmtsource.longitude,
                                             sta_lat, sta_lon)
            self.sta_azi.append(az)
            self.sta_theta.append(az / 180.0 * np.pi)
            if self.mode == "regional":
                # if regional, then use original distance(in km)
                self.sta_dist.append(dist / 1000.0)
            elif self.mode == "global":
                # if global, then use degree as unit
                self.sta_dist.append(dist / EARTH_HC)

    def get_azimuth_bin_number(self, azimuth):
        """
        Calculate the bin number of a given azimuth

        :param azimuth: test test test
        :return:
        """
        # the azimth ranges from [0,360]
        # so a little modification here
        daz = 360.0 / self.nregions
        k = int(math.floor(azimuth / daz))
        if k < 0 or k > self.nregions:
            if azimuth - 360.0 < 0.0001:
                k = self.nregions - 1
            else:
                raise ValueError('Error binning azimuth')
        return k

    def calculate_azimuth_bin(self, azimuth_array):
        """
        Calculate the azimuth and sort them into bins

        :return:
        """
        delta = 2 * np.pi / self.nregions
        bins = [delta * i for i in range(self.nregions)]

        naz_wins = np.zeros(self.nregions)
        for azimuth in azimuth_array:
            bin_idx = self.get_azimuth_bin_number(azimuth[0])
            naz_wins[bin_idx] += azimuth[1]
        return bins, naz_wins

    @staticmethod
    def plot_si_bb(ax, cmt):
        # get moment tensor
        mt = [cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp, cmt.m_tp]
        # plot beach ball
        b = beach(mt, linewidth=1, xy=(0, 0.6), width=1, size=2,
                  facecolor='r')
        ax.add_collection(b)
        # set axis
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1.5])
        ax.set_aspect('equal')
        # magnitude
        text = "Mw=%4.3f" % cmt.moment_magnitude
        plt.text(-0.9, -0.3, text, fontsize=9)
        # lat and lon
        text = r"lat=%6.3f$^\circ$; lon=%6.3f$^\circ$" \
               % (cmt.latitude, cmt.longitude)
        plt.text(-0.9, -0.5, text, fontsize=9)
        # depth
        text = "dep=%6.3f km;" % (cmt.depth_in_m / 1000.0)
        plt.text(-0.9, -0.7, text, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        # title
        text = "C" + cmt.eventname + " - Init CMT"
        plt.text(-0.9, 1.3, text, fontsize=10)

    @staticmethod
    def plot_si_bb_comp(ax, cmt, cmt_init, tag):
        # get moment tensor
        mt = [cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp, cmt.m_tp]
        # plot beach ball
        b = beach(mt, linewidth=1, xy=(0, 0.6), width=1, size=2,
                  facecolor='r')
        ax.add_collection(b)
        # set axis
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1.5])
        ax.set_aspect('equal')
        # magnitude
        text = r"$\Delta$Mw=%4.3f" % (cmt.moment_magnitude
                                      - cmt_init.moment_magnitude)
        plt.text(-0.9, -0.3, text, fontsize=9)
        # lat and lon
        text = r"$\Delta$lat=%6.3f$^\circ$; $\Delta$lon=%6.3f$^\circ$" \
               % (cmt.latitude - cmt_init.latitude,
                  cmt.longitude - cmt_init.longitude)
        plt.text(-0.9, -0.5, text, fontsize=9)
        # depth
        text = r"$\Delta$dep=%6.3f km;" % ((cmt.depth_in_m
                                            - cmt_init.depth_in_m)
                                           / 1000.0)
        plt.text(-0.9, -0.7, text, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        text = tag
        plt.text(-0.9, 1.3, text, fontsize=10)

    def plot_cmt3d_table(self):
        par_mean = self.bootstrap_mean
        par_std = self.bootstrap_std
        std_over_mean = np.zeros(par_mean.shape)
        for _i in range(par_mean.shape[0]):
            if par_mean[_i] != 0:
                std_over_mean[_i] = par_std[_i] / np.abs(par_mean[_i])
            else:
                std_over_mean[_i] = 0.0
        fontsize = 9
        incre = 0.06
        pos = 0.94
        # Moment Tensors
        format1 = "%15.4e  %15.4e  %15.4e  %15.4e  %10.2f%%"
        # CMT/HDR
        # format2 = r"%10.3f s  %13.3f s  %13.3f s  %13.3f s  %13.2f%%"
        # Depth
        format3 = "%10.3f km  %12.3f km  %12.3f km  %12.3f km  %12.2f%%"
        # LatLon
        format4 = "%10.3f deg  %11.3f deg  %11.3f deg  %11.3f deg  %11.2f%%"

        text = "Number of stations: %5d    Number of windows: %4d" \
               % (len(self.sta_lat), self.nwindows) + \
               "    Envelope coef: %12.3f" % self.config.envelope_coef
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "Number of Parameter: %4d    Zero-Trace: %11s" \
               "    Double-couple: %12s " \
               % (self.config.npar, self.config.zero_trace,
                  self.config.double_couple)
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "Station Correction: %5s    Norm_by_energy: %7s" \
               "    Norm_by_category: %9s" \
               % (self.config.station_correction,
                  self.config.weight_config.normalize_by_energy,
                  self.config.weight_config.normalize_by_category)
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        energy_change = \
            (self.new_cmtsource.M0 - self.cmtsource.M0) / self.cmtsource.M0
        text = "Inversion Damping: %6.3f    Energy Change: %7.2f%%" \
               "    Variance Reduction: %6.2f%%" \
               % (self.config.damping, energy_change * 100,
                  self.var_reduction * 100)
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "-" * 32 + "   Summary Table   " + "-" * 32
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "PAR      Old_CMT          New_CMT          Bootstrap_Mean" \
               "   Bootstrap_STD  STD/Mean"
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "Mrr:" + format1 % (
            self.cmtsource.m_rr, self.new_cmtsource.m_rr,
            par_mean[0], par_std[0], std_over_mean[0] * 100)
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mtt:" + format1 % (
            self.cmtsource.m_tt, self.new_cmtsource.m_tt,
            par_mean[1], par_std[1], std_over_mean[1] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mpp:" + format1 % (
            self.cmtsource.m_pp, self.new_cmtsource.m_pp,
            par_mean[2], par_std[2], std_over_mean[2] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mrt:" + format1 % (
            self.cmtsource.m_rt, self.new_cmtsource.m_rt,
            par_mean[3], par_std[3], std_over_mean[3] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mrp:" + format1 % (
            self.cmtsource.m_rp, self.new_cmtsource.m_rp,
            par_mean[4], par_std[4], std_over_mean[4] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mtp:" + format1 % (
            self.cmtsource.m_tp, self.new_cmtsource.m_tp,
            par_mean[5], par_std[5], std_over_mean[5] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "DEP:" + format3 % (
            self.cmtsource.depth_in_m / 1000,
            self.new_cmtsource.depth_in_m / 1000,
            par_mean[6] / 1000, par_std[6] / 1000, std_over_mean[6] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "LAT:" + format4 % (
            self.cmtsource.latitude, self.new_cmtsource.latitude,
            par_mean[8], par_std[8], std_over_mean[8] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "LON:" + format4 % (
            self.cmtsource.longitude, self.new_cmtsource.longitude,
            par_mean[7], par_std[7], std_over_mean[7] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        plt.axis('off')

    def plot_full_table(self):
        """Plots the table that shows the CMT3D and the G3D stuff"""

        par_mean = self.bootstrap_mean
        par_std = self.bootstrap_std
        std_over_mean = np.zeros(par_mean.shape)
        for _i in range(par_mean.shape[0]):
            if par_mean[_i] != 0:
                std_over_mean[_i] = par_std[_i] / np.abs(par_mean[_i])
            else:
                std_over_mean[_i] = 0.0
        fontsize = 9
        incre = 0.06
        pos = 1.06
        # Moment Tensors
        format1 = "%15.4e  %15.4e  %15.4e  %15.4e  %10.2f%%"
        # CMT/HDR
        format2 = r"%10.3f s  %13.3f s  " \
                  "%13.3f s  %13.3f s  %13.2f%%"
        # Depth
        format3 = "%10.3f km  %12.3f km  %12.3f km  %12.3f km  %12.2f%%"
        # LatLon
        format4 = "%10.3f deg  %11.3f deg  %11.3f deg  %11.3f deg  %11.2f%%"

        text = "Number of stations: %5d    Number of windows: %4d" \
               % (len(self.sta_lat), self.nwindows) + \
               "    Envelope coef: %12.3f" % self.config.envelope_coef
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "Number of Parameter: %4d    Zero-Trace: %11s" \
               "    Double-couple: %12s " \
               % (self.config.npar, self.config.zero_trace,
                  self.config.double_couple)
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "Station Correction: %5s    Norm_by_energy: %7s" \
               "    Norm_by_category: %9s" \
               % (self.config.station_correction,
                  self.config.weight_config.normalize_by_energy,
                  self.config.weight_config.normalize_by_category)
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        energy_change = \
            (self.new_cmtsource.M0 - self.cmtsource.M0) / self.cmtsource.M0
        text = "Inversion Damping: %6.3f    Energy Change: %7.2f%%" \
               "    Variance Reduction: %6.2f%%" \
               % (self.config.damping, energy_change * 100,
                  self.var_reduction * 100)
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "-" * 32 + "   Summary Table   " + "-" * 32
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "PAR      Global CMT          CMT3D           GRID          " \
               "Bootstrap_Mean   Bootstrap_STD  STD/Mean"
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "PAR      Global CMT          CMT3D           GRID          " \
               "Bootstrap_Mean   Bootstrap_STD  STD/Mean"
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        pos -= incre
        text = "Mrr:" + format1 % (
            self.cmtsource.m_rr, self.new_cmtsource.m_rr,
            par_mean[0], par_std[0], std_over_mean[0] * 100)
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mtt:" + format1 % (
            self.cmtsource.m_tt, self.new_cmtsource.m_tt,
            par_mean[1], par_std[1], std_over_mean[1] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mpp:" + format1 % (
            self.cmtsource.m_pp, self.new_cmtsource.m_pp,
            par_mean[2], par_std[2], std_over_mean[2] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mrt:" + format1 % (
            self.cmtsource.m_rt, self.new_cmtsource.m_rt,
            par_mean[3], par_std[3], std_over_mean[3] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mrp:" + format1 % (
            self.cmtsource.m_rp, self.new_cmtsource.m_rp,
            par_mean[4], par_std[4], std_over_mean[4] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Mtp:" + format1 % (
            self.cmtsource.m_tp, self.new_cmtsource.m_tp,
            par_mean[5], par_std[5], std_over_mean[5] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "DEP:" + format3 % (
            self.cmtsource.depth_in_m / 1000,
            self.new_cmtsource.depth_in_m / 1000,
            par_mean[6] / 1000, par_std[6] / 1000, std_over_mean[6] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "LAT:" + format4 % (
            self.cmtsource.latitude, self.new_cmtsource.latitude,
            par_mean[8], par_std[8], std_over_mean[8] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "LON:" + format4 % (
            self.cmtsource.longitude, self.new_cmtsource.longitude,
            par_mean[7], par_std[7], std_over_mean[7] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "Grid Search Parameters:"
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        text = "CMT:" + format2 % (
            self.cmtsource.time_shift, self.new_cmtsource.time_shift,
            self.cmtsource.time_shift + par_mean[9],
            par_std[9], 0.0)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')

        text = "M0: " + format1 % (
            self.cmtsource.M0, self.new_cmtsource.M0,
            self.bootstrap_mean[-1], self.bootstrap_std[-1],
            self.bootstrap_std[-1] / self.bootstrap_mean[-1] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize, fontfamily='monospace')
        plt.axis('off')

    def plot_global_map(self):
        """
        Plot global map of event and stations
        :return:
        """
        ax = plt.gca()
        ax.frameon = True
        ax.outline_patch.set_linewidth(0.75)
        # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
        # function around 180deg
        gl = ax.gridlines(crs=PlateCarree(), draw_labels=False,
                          linewidth=1, color='lightgray', alpha=0.5,
                          linestyle='-')
        gl.top_labels = False
        gl.left_labels = False
        gl.xlines = True
        # ax.outline_patch.set_visible(False)

        # Change fontsize
        fontsize = 12
        font_dict = {"fontsize": fontsize,
                     "weight": "bold"}
        ax.set_xticklabels(ax.get_xticklabels(), fontdict=font_dict)
        ax.set_yticklabels(ax.get_yticklabels(), fontdict=font_dict)

        ax.add_feature(cartopy.feature.LAND, zorder=0,
                       edgecolor='black', facecolor=(0.85, 0.85, 0.85))

        # Plot stations
        ax.scatter(self.sta_lon, self.sta_lat, 30, color="r", marker="^",
                   edgecolor="k", linewidth='0.3', zorder=3)

        # Get CMT location
        cmt_lat = self.cmtsource.latitude
        cmt_lon = self.cmtsource.longitude
        focmecs = get_cmt_par(self.cmtsource)[:6]

        if self.mode == 'regional':
            minlon = np.min(self.sta_lon)
            maxlon = np.max(self.sta_lon)
            minlat = np.min(self.sta_lat)
            maxlat = np.max(self.sta_lat)
            padding = 1.0

            # Updated parallels..
            ax.set_extent([minlon - padding, maxlon + padding,
                           minlat - padding, maxlat + padding])
            width_beach = min((maxlon + 2 * padding - minlon) / (8 * padding),
                              (maxlat + 2 * padding - minlat) / (8 * padding))
        else:
            ax.set_global()
            width_beach = 20

        bb = beach(focmecs, xy=(cmt_lon, cmt_lat),
                   width=width_beach, linewidth=1, alpha=1.0)
        bb.set_zorder(10)
        ax.add_collection(bb)

    def plot_sta_dist_azi(self):
        plt.title("Station Dist and Azi", fontsize=10, y=1.15)
        ax = plt.gca()
        c = plt.scatter(self.sta_theta, self.sta_dist, marker=u'^', c='r',
                        s=20, edgecolor='k', linewidth='0.3')
        c.set_alpha(0.75)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        if self.mode == "regional":
            ax.set_rmax(1.10 * max(self.sta_dist))
        elif self.mode == "global":
            ax.set_rmax(1.0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_sta_azi(self):
        # set plt.subplot(***, polar=True)
        plt.title("Station Azimuth", fontsize=10, y=1.15)
        azimuth_array = []
        for azi in self.sta_azi:
            azimuth_array.append([azi, 1])
        bins, naz = self.calculate_azimuth_bin(azimuth_array)
        norm_factor = np.max(naz)

        bars = plt.bar(bins, naz, width=(bins[1] - bins[0]), bottom=0.0)
        for r, bar in zip(naz, bars):
            bar.set_facecolor(plt.cm.jet(r / norm_factor))
            bar.set_alpha(0.5)
            bar.set_linewidth(0.3)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        ax = plt.gca()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_win_azi(self):
        # set plt.subplot(***, polar=True)
        plt.title("Window Azimuth", fontsize=10, y=1.15)
        win_azi = []
        for azi, nwindows in zip(self.sta_azi, self.nwin_on_trace):
            win_azi.append([azi, nwindows])
        bins, naz = self.calculate_azimuth_bin(win_azi)
        norm_factor = np.max(naz)

        bars = plt.bar(bins, naz, width=(bins[1] - bins[0]), bottom=0.0)
        for r, bar in zip(naz, bars):
            bar.set_facecolor(plt.cm.jet(r / norm_factor))
            bar.set_alpha(0.5)
            bar.set_linewidth(0.3)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        ax = plt.gca()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_dataset(self, figurename=None):
        """
        Plot only the dataset, including global map, station and window
        distribution, and beach ball
        """
        plt.figure(figsize=(10, 7), facecolor='w', edgecolor='k')
        g = gridspec.GridSpec(2, 3)
        plt.subplot(g[0, :-1], projection=PlateCarree(0.0))
        self.plot_global_map()
        plt.subplot(g[1, 0], polar=True)
        self.plot_sta_dist_azi()
        plt.subplot(g[1, 1], polar=True)
        self.plot_sta_azi()
        plt.subplot(g[1, 2], polar=True)
        self.plot_win_azi()
        ax = plt.subplot(g[0, 2])
        self.plot_si_bb(ax, self.cmtsource)
        plt.tight_layout()
        if figurename is None:
            plt.show()
        else:
            plt.savefig(figurename)
            plt.close()

    def plot_faults(self):
        fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "faults",
                             "gem_active_faults_harmonized.shp")
        with shapefile.Reader(fname) as shp:
            coordinates = []
            for k, rec in enumerate(shp.shapes()):
                coordinates.append(
                    np.array(rec.__geo_interface__['coordinates']))

        faults = [[(lon, lat) for lon, lat in coords] for coords in
                  coordinates]

        ax = plt.gca()
        lc = LineCollection(faults, colors=(0, 0, 0), zorder=150,
                            lw=0.5)
        ax.add_collection(lc)

    def plot_mini_map(self):

        ax = plt.gca()

        # Get CMT location
        cmt_lat = self.cmtsource.latitude
        cmt_lon = self.cmtsource.longitude

        padding = 0.1
        minlon = cmt_lon - padding
        maxlon = cmt_lon + padding
        minlat = cmt_lat - padding
        maxlat = cmt_lat + padding

        # Updated parallels..
        ax.set_xlim([minlon - padding, maxlon + padding])
        ax.set_ylim([minlat - padding, maxlat + padding])
        ax.frameon = True

        lon_tick = np.arange(np.floor(minlon) - padding,
                             np.ceil(maxlon) + 1.51 * padding,
                             padding / 2)
        lat_tick = np.arange(np.floor(minlat) - padding,
                             np.ceil(maxlat) + 1.51 * padding,
                             padding / 2)

        # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
        # function around 180deg
        gl = ax.gridlines(crs=PlateCarree(), draw_labels=True,
                          linewidth=1, color='lightgray', alpha=0.5,
                          linestyle='-')
        gl.top_labels = False
        gl.left_labels = False
        gl.xlines = True
        gl.xlocator = mticker.FixedLocator(lon_tick)
        gl.ylocator = mticker.FixedLocator(lat_tick)
        gl.xlabel_style = {"rotation": 45., "ha": "right"}

        # ax.outline_patch.set_visible(False)

        # Change fontsize
        fontsize = 12
        font_dict = {"fontsize": fontsize,
                     "weight": "bold"}
        ax.set_xticklabels(ax.get_xticklabels(), fontdict=font_dict)
        ax.set_yticklabels(ax.get_yticklabels(), fontdict=font_dict)

        # ax.add_feature(cartopy.feature.COASTLINE, lw=2,
        ax.coastlines(color='black', lw=1, zorder=200)

        # Plot stations
        ax.scatter(self.sta_lon, self.sta_lat, 30, color="r", marker="^",
                   edgecolor="k", linewidth='0.4', zorder=201)

        # Plot beachballs
        self.plot_beachballs(minlon, maxlon, minlat, maxlat, padding)

    def plot_beachballs(self, minlon, maxlon, minlat, maxlat, padding):

        ax = plt.gca()

        # Beachball width
        width_beach = min((maxlon + 2 * padding - minlon) / (180 * padding),
                          (maxlat + 2 * padding - minlat) / (180 * padding))

        # Get CMT location
        cmt_lat = self.cmtsource.latitude
        cmt_lon = self.cmtsource.longitude

        # Get new location
        new_cmt_lat = self.new_cmtsource.latitude
        new_cmt_lon = self.new_cmtsource.longitude

        # Correct plotting locations
        oldlat, oldlon, newlat, newlon, = \
            get_new_locations(cmt_lat, cmt_lon,
                              new_cmt_lat, new_cmt_lon,
                              padding)
        # Plot points
        markersize = 7.5
        ax.plot(cmt_lon, cmt_lat, "k.", zorder=200,
                markersize=markersize)
        ax.plot(new_cmt_lon, new_cmt_lat, "k.", zorder=200,
                markersize=markersize)

        # Plot lines
        ax.plot([cmt_lon, oldlon], [cmt_lat, oldlat], "k", zorder=199)
        ax.plot([new_cmt_lon, newlon], [new_cmt_lat, newlat], "k", zorder=199)

        # Old CMT
        ax = plt.gca()
        focmecs = get_cmt_par(self.cmtsource)[:6]
        bb = beach(focmecs, xy=(oldlon, oldlat),
                   width=width_beach, linewidth=1, alpha=1.0, zorder=250)
        ax.add_collection(bb)

        # New CMT
        new_focmecs = get_cmt_par(self.new_cmtsource)[:6]
        new_bb = beach(new_focmecs, facecolor='r',
                       xy=(newlon, newlat), width=width_beach,
                       linewidth=1, alpha=1.0, zorder=250)
        ax.add_collection(new_bb)

    def plot_geology(self):
        """Reads json and plots geological features"""

        fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "geology", "geology.json")
        geomap = GeoMap.load_json(fname)
        ax = plt.gca()

        # Get discrete colormap
        self.geo_cmap = cm.get_cmap('gist_ncar', geomap.ndescr)

        # Translate the colors to a colormap
        self.colordict = dict()

        for k, descriptions in enumerate(sorted(list(geomap.descriptions))):
            self.colordict[descriptions] = k

        patches = []
        colors = []
        for member in geomap.members:

            for poly in member.coordinates:
                patches.append(Polygon(np.fliplr(poly), joinstyle="round",
                                       fill=True, edgecolor=None))
                colors.append(self.colordict[member.description])

        p = PatchCollection(patches, cmap=self.geo_cmap, alpha=0.4, zorder=100)
        p.set_array(np.array(colors))
        ax.add_collection(p)

    def plot_geology_legend(self):

        plt.figure()
        plt.figure(figsize=(15, 8), facecolor='w', edgecolor='k')
        ax = plt.axes()
        circs = []
        labels = []

        for key, value in self.colordict.items():
            circs.append(
                Line2D([0], [0], linestyle="none", marker="s", alpha=0.4,
                       markersize=10, markerfacecolor=self.geo_cmap(value)))
            labels.append(key)

        ax.legend(circs, labels, numpoints=1, loc="best",
                  ncol=2, fontsize=8, frameon=False)
        ax.axis('off')

    def plot_cost(self):

        ax = plt.gca()

        x = np.arange(0, self.G.mincost_array.shape[0], 1)
        ax.fill_between(x, self.G.mincost_array, self.G.maxcost_array,
                        color='lightgray', label="min/max")
        ax.fill_between(x, self.G.meancost_array - self.G.stdcost_array,
                        self.G.meancost_array + self.G.stdcost_array,
                        color='darkgray', label=r"$\bar{\chi}\pm\sigma$")
        ax.plot(self.G.meancost_array, 'k', label=r"$\bar{\chi}$")

        method = self.G.config.method

        if method == "gn":
            label = "Gauss-Newton"
        else:
            label = "Newton"

        ax.plot(self.G.chi_list, "r",
                label=r"%s ($\mathcal{C}_{min} = %.3f$)"
                % (label, self.G.chi_list[-1]))
        plt.legend(prop={'size': 6}, fancybox=False, framealpha=1)
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Misfit reduction")
        ax.set_ylim([0, 1])

    def plot_inversion_summary(self, figurename=None):
        """
        Plot the dataset and the inversion result.
        """
        if self.new_cmtsource is None:
            raise ValueError("No new cmtsource...Can't plot summary")

        fig = plt.figure(figsize=(10, 10.5), facecolor='w', edgecolor='k',
                         tight_layout=True)
        g = gridspec.GridSpec(3, 3)

        # Global map
        plt.subplot(g[0, :-1], projection=PlateCarree())
        self.plot_global_map()

        # Grid search or stations vz distance and azimuth
        if type(self.G) == Grid3d:
            plt.subplot(g[1, 0])
            self.G.plot_grid()
        elif type(self.G) in [Gradient3d, Struct]:
            plt.subplot(g[1, 0])
            self.plot_cost()
        else:
            plt.subplot(g[1, 0], polar=True)
            self.plot_sta_dist_azi()

        #  station azimuth histogram
        plt.subplot(g[1, 1], polar=True)
        self.plot_sta_azi()

        # Plot window azimuth histogram
        plt.subplot(g[1, 2], polar=True)
        self.plot_win_azi()
        ax = plt.subplot(g[0, 2], projection=PlateCarree())
        # ax.set_global()
        self.plot_mini_map()
        # self.plot_geology()
        self.plot_faults()
        # self.plot_si_bb(ax, self.cmtsource)
        ax = plt.subplot(g[2, 2])
        self.plot_si_bb_comp(ax, self.new_cmtsource, self.cmtsource,
                             "Inversion")
        plt.subplot(g[2, :-1])
        if type(self.G) is not None:
            self.plot_cmt3d_table()
        else:
            self.plot_full_table()
        fig.canvas.draw()
        plt.tight_layout()
        if figurename is None:
            plt.show()
        else:
            plt.savefig(figurename)
            plt.close('all')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', help='Path to cmt3d inversion summary JSON file',
                        type=str, dest='cmt3dfile', required=True)
    parser.add_argument('-g', type=str, dest='g3dfile', required=False,
                        help='Path to grid search inversion summary JSON file',
                        default=None)
    parser.add_argument('-o', help='Output filename', dest='outputfile',
                        required=True, type=str)
    args = parser.parse_args()

    # Run
    pl = PlotInvSummary.from_JSON(args.cmt3dfile, g3d_file=args.g3dfile)
    pl.plot_inversion_summary(figurename=args.outputfile)


if __name__ == "__main__":
    main()
