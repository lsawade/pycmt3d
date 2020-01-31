#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for geo_data_utilities combines cmt3d and grid3d

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
import numpy as np
import pytest
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # NOQA
from pycmt3d import CMTSource
from pycmt3d import DataContainer
from pycmt3d import DefaultWeightConfig, Config
from pycmt3d.constant import PARLIST
from pycmt3d import Cmt3D
from pycmt3d.geo_data_util import Member
from pycmt3d.geo_data_util import GeoMap


# Most generic way to get the data geology path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")
GEO_DATA_DIR = os.path.join(DATA_DIR, "geo_data")
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


def weight_sum(metas):
    sumw = 0
    for meta in metas:
        sumw += np.sum(meta.weights)
    return sumw


def test_member():
    """Tests the features member"""

    coordinates = [np.array([[0, 0],
                             [1, 0],
                             [1, 1],
                             [0, 1]]),
                   np.array([[1, 1],
                             [2, 1],
                             [2, 2],
                             [1, 2]])]

    lithology = 'Test Lithology'
    stratigraphy = 'Test Stratigraphy'
    description = 'Test Description'
    bbox = np.array([0, 0, 2, 2])

    member = Member(coordinates=coordinates, stratigraphy=stratigraphy,
                    bbox=bbox, description=description,
                    lithology=lithology)

    np.testing.assert_array_almost_equal(member.coordinates, coordinates)
    np.testing.assert_array_almost_equal(member.bbox, bbox)
    assert lithology == member.lithology
    assert stratigraphy == member.stratigraphy
    assert description == member.description


def test_member_from_dict():

    coordinates = [np.array([[0, 0],
                             [1, 0],
                             [1, 1],
                             [0, 1]]),
                   np.array([[1, 1],
                             [2, 1],
                             [2, 2],
                             [1, 2]])]

    lithology = 'Test Lithology'
    stratigraphy = 'Test Stratigraphy'
    description = 'Test Description'
    bbox = np.array([0, 0, 2, 2])

    d = {"coordinates": coordinates,
         "description": description,
         "lithology": lithology,
         "stratigraphy": stratigraphy,
         "bbox": bbox}

    member = Member.from_dict(d)

    np.testing.assert_array_almost_equal(member.coordinates, coordinates)
    np.testing.assert_array_almost_equal(member.bbox, bbox)
    assert lithology == member.lithology
    assert stratigraphy == member.stratigraphy
    assert description == member.description

def test_GeoMap():
    """Test GeoMap class"""

    geoxml = os.path.join(GEO_DATA_DIR, "geo_test.gml")
    geomap = GeoMap.from_xml(geoxml)

    geomap.save_json(os.path.join(GEO_DATA_DIR, "test.json"))



