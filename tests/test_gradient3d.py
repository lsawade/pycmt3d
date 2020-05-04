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
from pycmt3d import Gradient3d, Gradient3dConfig


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
    dcon = DataContainer()
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
    dcon = DataContainer()
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
    dcon = DataContainer()
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

def test_simple_grad3d(tmpdir, cmtsource):

    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    grad3d_config = Gradient3dConfig(method="gn", weight_data=True,
                                     weight_config=weight_config,
                                     taper_type="tukey",
                                     c1=1e-4, c2=0.9,
                                     idt=0.0, ia=1.,
                                     nt=10, nls=20,
                                     crit=0.01,
                                     precond=False, reg=False,
                                     bootstrap=False, bootstrap_repeat=20,
                                     bootstrap_subset_ratio=0.4,
                                     mpi_env=False)

    G = Gradient3d(cmtsource, dcon_two, grad3d_config)
    G.search()
    G.plot_stats_histogram(str(tmpdir))
    G.plot_new_synt_seismograms(str(tmpdir))




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

    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    grad3d_config = Gradient3dConfig(method="gn", weight_data=True,
                                     weight_config=weight_config,
                                     taper_type="tukey",
                                     c1=1e-4, c2=0.9,
                                     idt=0.0, ia=1.,
                                     nt=10, nls=20,
                                     crit=0.01,
                                     precond=False, reg=False,
                                     bootstrap=False, bootstrap_repeat=20,
                                     bootstrap_subset_ratio=0.4,
                                     mpi_env=False)


    outdir = "/Users/lucassawade/inversion_test"

    G = Gradient3d(cmtsource, dcon_two, grad3d_config)
    G.search()
    G.write_summary_json(outdir)