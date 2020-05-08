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
import os
import glob
import shutil
import inspect
import pytest
from pycmt3d import DefaultWeightConfig, Config
from pycmt3d.constant import PARLIST
from pycmt3d import CMTSource
from pycmt3d import DataContainer
from pycmt3d import Cmt3D, Config
from pycmt3d import Gradient3d, Gradient3dConfig
# Most generic way to get the data geology path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")
OBSD_DIR = os.path.join(DATA_DIR, "data_T006_T030")
SYNT_DIR = os.path.join(DATA_DIR, "syn_T006_T030")
CMTFILE = os.path.join(DATA_DIR, "CMTSOLUTION")


def copy_files(files, destdir):
    for f in files:
        print(os.path.join(destdir, os.path.basename(f)))
        shutil.copy(f, os.path.join(destdir, os.path.basename(f)))


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
                                   file_format="txt",velocity=True,
                                   wave_type="local", wave_weight=1.0)
    return dcon


def construct_dcon_three():
    """
    Data Container with two stations
    """
    dcon = DataContainer(parlist=PARLIST[:9])
    os.chdir(DATA_DIR)
    window_file = os.path.join(DATA_DIR,
                               "flexwin_T006_T030.output")
    dcon.add_measurements_from_sac(window_file, tag="T006_T030",
                                   file_format="txt", velocity=True,
                                   wave_type="local", wave_weight=1.0)
    return dcon


def test_inversion(cmtsource, tmpdir):

    outdir = os.path.join(str(tmpdir))
    newdata_dir = os.path.join(outdir, "data_T006_T030")
    newsyn_dir = os.path.join(outdir, "syn_T006_T030")
    if not os.path.exists(newdata_dir):
        os.makedirs(newdata_dir)

    # Copy OBSD files to new directory
    files = glob.glob(os.path.join(OBSD_DIR, "*.sac.d"))
    copy_files(files, newdata_dir)


    # Construct data container with the old files.
    dcon_two = construct_dcon_two()

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
    srcinv.plot_new_synt_seismograms(outputdir=outdir)
    srcinv.write_new_cmtfile(outputdir=outdir)
    srcinv.write_new_syn(outputdir=newsyn_dir, suffix="short")


    # Load cmt form tmpdir
    cmtfile = glob.glob(os.path.join(outdir, "*.inv"))[0]
    cmtsourcenew = CMTSource.from_CMTSOLUTION_file(cmtfile)

    # Create new datacontainer
    window_file = os.path.join(DATA_DIR,
                               "flexwin_T006_T030.output.two_stations")
    copy_files([window_file], outdir)
    newwinfile = os.path.join(outdir, "flexwin_T006_T030.output.two_stations")

    os.chdir(outdir)
    dcon_new = DataContainer()
    dcon_new.add_measurements_from_sac(newwinfile, tag="T006_T030",
                                   file_format="txt")

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

    G = Gradient3d(cmtsourcenew, dcon_new, grad3d_config)
    G.search()
    G.write_summary_json(outdir)


if __name__ == "__main__":


    cmt = CMTSource.from_CMTSOLUTION_file(CMTFILE)
    outdir = "/Users/lucassawade/inversion_test"
    newdata_dir = os.path.join(outdir, "data_T006_T030")
    if not os.path.exists(newdata_dir):
        os.makedirs(newdata_dir)
    newsyn_dir = os.path.join(outdir, "syn_T006_T030")

    # Copy OBSD files to new directory
    files = glob.glob(os.path.join(OBSD_DIR, "*.sac.d"))
    copy_files(files, newdata_dir)

    # Construct data container with the old files.
    dcon_two = construct_dcon_three()

    weight_config = DefaultWeightConfig(
        normalize_by_energy=True, normalize_by_category=True,
        comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
        love_dist_weight=1.0, pnl_dist_weight=1.0,
        rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    config = Config(9, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                    zero_trace=True, weight_data=True, wave_weight=True,
                    station_correction=True, envelope_coef=0.5,
                    weight_config=weight_config,
                    bootstrap=True, bootstrap_repeat=50)

    srcinv = Cmt3D(cmt, dcon_two, config)
    srcinv.source_inversion()
    srcinv.plot_summary(outputdir=outdir, figure_format='pdf')
    srcinv.plot_new_synt_seismograms(outputdir=outdir, figure_format='pdf',
                                     suffix='cmt')
    srcinv.write_new_cmtfile(outputdir=outdir)
    srcinv.write_new_syn(outputdir=newsyn_dir, suffix="short")
    srcinv.write_summary_json(outputdir=outdir)
    # Load cmt form tmpdir
    cmtfile = glob.glob(os.path.join(outdir, "*.inv"))[0]
    cmtsourcenew = CMTSource.from_CMTSOLUTION_file(cmtfile)

    # Create new datacontainer
    window_file = os.path.join(DATA_DIR,
                               "flexwin_T006_T030.output")
    copy_files([window_file], outdir)
    newwinfile = os.path.join(outdir, "flexwin_T006_T030.output")

    os.chdir(outdir)

    dcon_new = DataContainer()
    dcon_new.add_measurements_from_sac(newwinfile, tag="T006_T030",
                                       file_format="txt", velocity=True,
                                       only_observed=True,
                                       wave_type="local", wave_weight=1.0)

    grad3d_config = Gradient3dConfig(method="gn", weight_data=True,
                                     weight_config=weight_config,
                                     taper_type="tukey",
                                     c1=1e-4, c2=0.9,
                                     idt=0.0, ia=1.,
                                     nt=10, nls=20,
                                     crit=0.01,
                                     precond=False, reg=False,
                                     bootstrap=True, bootstrap_repeat=25,
                                     bootstrap_subset_ratio=0.6,
                                     mpi_env=True)

    G = Gradient3d(cmtsourcenew, dcon_new, grad3d_config)
    G.search()
    G.write_summary_json(outdir)
    G.write_new_syn(outdir, file_format='sac')
    G.plot_new_synt_seismograms(outdir, figure_format='pdf', suffix="grad")
