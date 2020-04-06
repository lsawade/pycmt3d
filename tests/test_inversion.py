#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for combined source inversion and grid search.

:copyright:
    Lucas Sawade (lsawade@princeton.edu), 2020
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

Last Update: January 2020

"""

from __future__ import print_function, division
import inspect
import os
import pytest
from pycmt3d import DefaultWeightConfig, Config
from pycmt3d.constant import PARLIST
from pycmt3d import CMTSource
from pycmt3d import DataContainer
from pycmt3d import Grid3dConfig
from pycmt3d import Inversion
from pycmt3d.gradient3d import Gradient3dConfig
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


def test_inversion(cmtsource, tmpdir):

    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    cmt3d_config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                          zero_trace=True, weight_data=True,
                          station_correction=True,
                          weight_config=weight_config,
                          bootstrap=True, bootstrap_repeat=20,
                          bootstrap_subset_ratio=0.4)

    energy_keys = ["power_l1", "power_l2", "cc_amp", "chi"]
    grid3d_config = Grid3dConfig(origin_time_inv=True, time_start=-5,
                                 time_end=5,
                                 dt_over_delta=5, energy_inv=True,
                                 energy_start=0.5, energy_end=1.5,
                                 denergy=0.1,
                                 energy_keys=energy_keys,
                                 energy_misfit_coef=[0.25, 0.25, 0.25, 0.25],
                                 weight_data=True, weight_config=weight_config,
                                 use_new=True)

    inv = Inversion(cmtsource, dcon_two, cmt3d_config, grid3d_config)
    inv.source_inversion()
    inv.plot_summary(str(tmpdir))


def test_inversion_no_grid(cmtsource, tmpdir):

    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    cmt3d_config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                          zero_trace=True, weight_data=True,
                          station_correction=True,
                          weight_config=weight_config,
                          bootstrap=True, bootstrap_repeat=20,
                          bootstrap_subset_ratio=0.4)

    inv = Inversion(cmtsource, dcon_two, cmt3d_config, mt_config=None)
    inv.source_inversion()
    inv.plot_summary(str(tmpdir))


def test_inversion_grad(cmtsource, tmpdir):

    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    cmt3d_config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                          zero_trace=True, weight_data=True,
                          station_correction=True,
                          weight_config=weight_config,
                          bootstrap=True, bootstrap_repeat=20,
                          bootstrap_subset_ratio=0.4)

    grad3d_config = Gradient3dConfig(method="gn", weight_data=True,
                                     weight_config=weight_config,
                                     use_new=True,
                                     taper_type="tukey",
                                     c1=1e-4, c2=0.9,
                                     idt=0.0, ia=1.,
                                     nt=10, nls=20,
                                     crit=0.01,
                                     precond=False, reg=False,
                                     bootstrap=False, bootstrap_repeat=20,
                                     bootstrap_subset_ratio=0.4, mpi_env=True)

    inv = Inversion(cmtsource, dcon_two, cmt3d_config, grad3d_config)
    inv.source_inversion()
    inv.write_summary_json(str(tmpdir))
    inv.plot_summary(str(tmpdir))


def test_inversion_grad_no_mpi(cmtsource, tmpdir):

    dcon_two = construct_dcon_two()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    cmt3d_config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                          zero_trace=True, weight_data=True,
                          station_correction=True,
                          weight_config=weight_config,
                          bootstrap=True, bootstrap_repeat=20,
                          bootstrap_subset_ratio=0.4)

    grad3d_config = Gradient3dConfig(method="gn", weight_data=True,
                                     weight_config=weight_config,
                                     use_new=True,
                                     taper_type="tukey",
                                     c1=1e-4, c2=0.9,
                                     idt=0.0, ia=1.,
                                     nt=10, nls=20,
                                     crit=0.01,
                                     precond=False, reg=False,
                                     bootstrap=False, bootstrap_repeat=20,
                                     bootstrap_subset_ratio=0.4, mpi_env=False)

    inv = Inversion(cmtsource, dcon_two, cmt3d_config, grad3d_config)
    inv.source_inversion()
    inv.write_summary_json(str(tmpdir))
    inv.plot_summary(str(tmpdir))


if __name__ == "__main__":
    outdir = "."
    cmt = CMTSource.from_CMTSOLUTION_file(CMTFILE)

    dcon = DataContainer(parlist=PARLIST[:9])
    os.chdir(DATA_DIR)
    window_file = os.path.join(DATA_DIR,
                               "flexwin_T006_T030."
                               "output.two_stations")
    dcon.add_measurements_from_sac(window_file, tag="T006_T030",
                                   file_format="txt")

    weight_config = DefaultWeightConfig(
        normalize_by_energy=False, normalize_by_category=False,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    cmt3d_config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                          zero_trace=True, weight_data=True,
                          station_correction=True,
                          weight_config=weight_config,
                          bootstrap=True, bootstrap_repeat=20,
                          bootstrap_subset_ratio=0.4)

    energy_keys = ["power_l1", "power_l2", "cc_amp", "chi"]

    grid3d_config = Grid3dConfig(origin_time_inv=True, time_start=-1.5,
                                 time_end=1.5,
                                 dt_over_delta=1, energy_inv=True,
                                 energy_start=0.2, energy_end=1.8,
                                 denergy=0.025,
                                 energy_keys=energy_keys,
                                 energy_misfit_coef=[0.25, 0.25,
                                                     0.25, 0.25],
                                 weight_data=True,
                                 weight_config=weight_config,
                                 use_new=True)

    grad3d_config = Gradient3dConfig(method="gn", weight_data=True,
                                     weight_config=weight_config,
                                     use_new=True,
                                     taper_type="tukey",
                                     c1=1e-4, c2=0.9,
                                     idt=0.0, ia=1.,
                                     nt=10, nls=20,
                                     crit=0.01,
                                     precond=False, reg=False,
                                     bootstrap=True, bootstrap_repeat=20,
                                     bootstrap_subset_ratio=0.4)
    inv = Inversion(cmt, dcon, cmt3d_config, grad3d_config)
    inv.source_inversion()
    inv.write_summary_json("/Users/lucassawade")
    inv.plot_summary("/Users/lucassawade")

    inv.plot_new_synt_seismograms("/Users/lucassawade")
    inv.write_new_cmtfile("/Users/lucassawade")
