#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pycmt3d test suite.

Run with pytest.

:copyright:
    Wenjie Lei (lei@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import print_function, division
import inspect
import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # NOQA
from pycmt3d import CMTSource
from pycmt3d import DataContainer
from pycmt3d import DefaultWeightConfig, Config
from pycmt3d.constant import PARLIST
from pycmt3d import Cmt3D


# Most generic way to get the data geology path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")
OBSD_DIR = os.path.join(DATA_DIR, "data_T006_T030")
SYNT_DIR = os.path.join(DATA_DIR, "syn_T006_T030")
CMTFILE = os.path.join(DATA_DIR, "CMTSOLUTION")


@pytest.fixture
def cmtsource():
    return CMTSource.from_CMTSOLUTION_file(CMTFILE)


@pytest.fixture
def default_config():
    return DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 2.0, "R": 1.0, "T": 2.0},
        love_dist_weight=0.78, pnl_dist_weight=1.15,
        rayleigh_dist_weight=0.55, azi_exp_idx=0.5)


@pytest.fixture
def dcon_one():
    """
    Data container with only one station
    """
    dcon = DataContainer(parlist=PARLIST[:9])
    os.chdir(DATA_DIR)
    window_file = os.path.join(DATA_DIR,
                               "flexwin_T006_T030.output.one_station")
    dcon.add_measurements_from_sac(window_file, tag="T006_T030",
                                   file_format="txt")
    return dcon


def construct_dcon_two():
    """
    Data Container with two stations
    """
    dcon = DataContainer(parlist=PARLIST[:9])
    os.chdir(DATA_DIR)
    window_file = os.path.join(DATA_DIR,
                               "flexwin_T006_T030.output.two_stations")
    dcon.add_measurements_from_sac(window_file, tag="T006_T030",
                                   file_format="txt")
    return dcon


def construct_dcon_three():
    """
    Data Container with two stations and a velocity measurement
    """
    dcon = DataContainer(parlist=PARLIST[:9])
    os.chdir(DATA_DIR)
    window_file = os.path.join(DATA_DIR,
                               "flexwin_T006_T030.output.two_stations")
    dcon.add_measurements_from_sac(window_file, tag="T006_T030",
                                   wave_weight=0.5, velocity=True,
                                   file_format="txt")


    return dcon


def weight_sum(metas):
    sumw = 0
    for meta in metas:
        sumw += np.sum(meta.weights)
    return sumw


def test_weighting_two(tmpdir, cmtsource):
    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    config = Config(6, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                    zero_trace=True, weight_data=True,
                    station_correction=True,
                    weight_config=weight_config)

    srcinv = Cmt3D(cmtsource, dcon_two, config)
    srcinv.source_inversion()
    srcinv.plot_new_synt_seismograms(str(tmpdir))


def test_weighting_two_7par(tmpdir, cmtsource):
    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    config = Config(7, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                    zero_trace=True, weight_data=True,
                    station_correction=True,
                    weight_config=weight_config)

    srcinv = Cmt3D(cmtsource, dcon_two, config)
    srcinv.source_inversion()
    srcinv.plot_new_synt_seismograms(str(tmpdir))


def test_weighting_two_9par(tmpdir, cmtsource):
    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                    zero_trace=True, weight_data=True,
                    station_correction=True,
                    weight_config=weight_config)

    srcinv = Cmt3D(cmtsource, dcon_two, config)
    srcinv.source_inversion()
    srcinv.plot_new_synt_seismograms(str(tmpdir))


def setup_inversion(cmt):
    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    config = Config(6, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                    zero_trace=True, weight_data=True,
                    station_correction=True,
                    weight_config=weight_config)

    srcinv = Cmt3D(cmt, dcon_two, config)
    srcinv.source_inversion()
    return srcinv


def test_weighting_two_9par_with_wave_weight_and_velocity(tmpdir, cmtsource):
    dcon_two = construct_dcon_three()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                    zero_trace=True, weight_data=True, wave_weight=True,
                    station_correction=True, envelope_coef=0.5,
                    weight_config=weight_config)

    srcinv = Cmt3D(cmtsource, dcon_two, config)
    srcinv.source_inversion()
    srcinv.plot_new_synt_seismograms(str(tmpdir))


def test_cmt_bootstrap(cmtsource, tmpdir):
    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                    zero_trace=True, weight_data=True,
                    station_correction=True,
                    weight_config=weight_config,
                    bootstrap=True, bootstrap_repeat=20,
                    bootstrap_subset_ratio=0.4)

    srcinv = Cmt3D(cmtsource, dcon_two, config)
    srcinv.source_inversion()
    srcinv.plot_summary(str(tmpdir))
    assert False

def test_write_new_cmtfile(cmtsource, tmpdir):
    srcinv = setup_inversion(cmtsource)
    srcinv.write_new_cmtfile(outputdir=str(tmpdir))


def test_plot_summary(cmtsource, tmpdir):
    srcinv = setup_inversion(cmtsource)
    srcinv.plot_summary(outputdir=str(tmpdir), mode="global")


def test_plot_stats_histogram(cmtsource, tmpdir):
    srcinv = setup_inversion(cmtsource)
    srcinv.plot_stats_histogram(outputdir=str(tmpdir))


if __name__ == "__main__":
    cmt = CMTSource.from_CMTSOLUTION_file(CMTFILE)

    dcon = DataContainer(parlist=PARLIST[:9])
    os.chdir(DATA_DIR)
    window_file = os.path.join(DATA_DIR,
                               "flexwin_T006_T030."
                               "output.two_stations")
    dcon.add_measurements_from_sac(window_file, tag="T006_T030",
                                   velocity=True, wave_type="local",
                                   wave_weight=1.0,
                                   file_format="txt")

    weight_config = DefaultWeightConfig(
        normalize_by_energy=True, normalize_by_category=True,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    cmt3d_config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                          zero_trace=True, weight_data=True,
                          station_correction=True, wave_weight=True,
                          weight_config=weight_config,
                          bootstrap=True, bootstrap_repeat=20,
                          bootstrap_subset_ratio=0.4)

    outdir = "/Users/lucassawade/inversion_test"
    pre = "/Users/lucassawade/inversion_test/pregrid"

    inv = Cmt3D(cmt, dcon, cmt3d_config)
    inv.source_inversion()
    inv.write_summary_json(outdir)
    inv.plot_summary(outdir, figure_format='pdf')

    inv.plot_new_synt_seismograms(outdir, figure_format='pdf')
    inv.write_new_syn(outdir, suffix="short")
    inv.write_new_cmtfile(outdir)
