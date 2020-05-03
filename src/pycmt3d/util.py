#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
General util functions

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function, division, absolute_import
import json
import numpy as np
from scipy import signal
from obspy.geodetics import locations2degrees
from obspy.core.trace import Trace
import scipy
from obspy import UTCDateTime
import datetime


def datetime_parser(dct):
    """Checks dictionary for UTC datetime strings and
    converts them."""

    for k, v in dct.items():
        if isinstance(v, str) and "UTC" in v:
            try:
                dct[k] = UTCDateTime(v[3:])
            except Exception:
                pass
    return dct


def default(o):
    if isinstance(o, (UTCDateTime, datetime.date, datetime.datetime)):
        return "UTC" + o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()


def dump_json(content, filename):
    with open(filename, "w") as fh:
        json.dump(content, fh, indent=2, sort_keys=2, default=default)


def load_json(filename):
    with open(filename, "r") as fh:
        d = json.load(fh)  # , object_hook=datetime_parser)
    return d


def get_trwin_tag(trwin):
    """
    trwin.tag is usually the period band, so
    category would be like "27_60.BHZ", "27_60.BHR", "27_60.BHT",
    and "60_120.BHZ", "60_120.BHR", "60_120.BHT".
    """
    return "%s.%s" % (trwin.tags['obsd'], trwin.channel)


def get_cmt_par(cmt):
    """
    Get cmt information as array
    """
    return np.array([cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp,
                     cmt.m_tp, cmt.depth_in_m, cmt.longitude,
                     cmt.latitude, cmt.time_shift, cmt.half_duration])


def distance(lat1, lon1, lat2, lon2):
    """
    Given two points location by (latitude, longitude) and return
    the distance on sphere between two points, unit in degree
    :return: distance in degree
    """
    return locations2degrees(lat1, lon1, lat2, lon2)


def normalize_array(array, factor):
    return factor * np.array(array)


def get_window_idx(win_time, dt):
    """
    Get window index from window time and dt
    """
    def _get_win_idx(win, delta):
        istart = int(win[0] / delta)
        iend = int(win[1] / delta)
        if istart < 0:
            raise ValueError("Start index(%d) smaller than 0")
        if istart >= iend:
            raise ValueError("Start index(%d) larger or equal than "
                             "end index(%d)" % (istart, iend))
        return np.array([istart, iend])

    win_time = np.array(win_time)
    win_idx = np.zeros(win_time.shape)

    if len(win_time.shape) < 1:
        raise ValueError("lenght of window is %d" % (len(win_time)))
    elif len(win_time.shape) == 1:
        win_idx = _get_win_idx(win_time, dt)
    else:
        for _i, _win in enumerate(win_time):
            win_idx[_i] = _get_win_idx(_win, dt)

    return win_idx


def check_trace_consistent(tr1, tr2, mode="part"):
    """
    Check if two traces are consistent with each other.
    If mode is 'part', only starttime and dt is compared
    If mode is 'full', npts is also compared
    """
    _options = ["part", "full"]
    if mode not in _options:
        raise ValueError("mode(%s) must be within %s" % (mode, _options))

    if not np.isclose(tr1.stats.delta, tr2.stats.delta):
        raise ValueError("DT of two traces are not the same: %f, %f"
                         % (tr1.stats.delta, tr2.stats.delta))

    if not np.isclose(tr1.stats.starttime - tr2.stats.starttime, 0):
        raise ValueError("Starttime of two traces not the same: %s, %s"
                         % (tr1.stats.starttime, tr2.stats.starttime))

    if mode == "full":
        if tr1.stats.npts != tr2.stats.npts:
            raise ValueError("NPTS not the same: %d, %d" % (tr1.stats.npts,
                                                            tr2.stats.npts))
    else:
        return


def sum_matrix(data, coef=None):
    """
    Sum a list of matrix with same dimension(sum over first index)
    :return: sum = coef[i] * data[i]
    """
    if coef is None:
        coef = np.ones(len(data))
    elif len(coef) != len(data):
        raise ValueError("dimension of coef and data not the same")

    sum_value = coef[0] * data[0]
    for _idx in range(1, len(coef)):
        sum_value += coef[_idx] * data[_idx]
    return sum_value


def random_select(nsamples, nselected=1, replace=True):
    """
    Draw nselected number of samples from nsamples,
    index [0, nsamples-1]
    :param nsamples: the total number of samples
    :type nsamples: int
    :param nselected: the number of ssamples drawed
    :type nselected: int
    :return: select position array. If selected twice, then on the same
        index, value would be 2.
    """
    subset_array = np.zeros(nsamples)
    location_array = np.random.choice(nsamples, nselected,
                                      replace=replace)
    for _idx in location_array:
        subset_array[_idx] += 1
    return subset_array


def _float_to_str(value):
    """
    Convert float value to a specific precision string

    :param value:
    :return: string of the value
    """
    return "%.5f" % value


def _float_array_to_str(array):
    """
    Convert float array to string

    :return:
    """
    string = "[  "
    for ele in array:
        string += "%.3e," % ele
    string += "]"
    return string


def tukey_window(window_length, alpha=0.2):
    """
    !!! OBSOLETE !!!
    The Tukey window, also known as the tapered cosine window,
    can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2).
    At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
    We use the same reference as MATLAB to provide the same results
    in case users compare a MATLAB output to this function output
    ---------
    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html

    """
    # Special cases
    if alpha <= 0:
        return np.ones(window_length)  # rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x < alpha/2
    w[first_condition] = \
        0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2)))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x >= (1 - alpha/2)
    w[third_condition] = \
        0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))

    return w


def construct_taper(npts, taper_type="tukey", alpha=0.2):
    """
    Construct taper based on npts

    :param npts: the number of points
    :param taper_type:
    :param alpha: taper width
    :return:
    """
    taper_type = taper_type.lower()
    _options = ['hann', 'boxcar', 'tukey']
    if taper_type not in _options:
        raise ValueError("taper type option: %s" % taper_type)
    if taper_type == "hann":
        taper = signal.hann(npts)
    elif taper_type == "boxcar":
        taper = signal.boxcar(npts)
    elif taper_type == "tukey":
        taper = signal.tukey(npts, alpha=alpha)
    else:
        raise ValueError("Taper type not supported: %s" % taper_type)
    return taper


def nextpoweroftwo(number):
    """ Returns next power of two following 'number' """
    new_number = 1
    while new_number <= 2*number:
        new_number *= 2
    return new_number


def timeshift_trace_np2(tr: Trace, t0: float):
    """Takes in a seismic trace and shifts it in time using the fft."""

    N = len(tr.data)
    Nfix = nextpoweroftwo(N)

    # Get frequency vector
    freq = np.fft.fftfreq(Nfix, d=tr.stats.delta)

    # Compute timeshifted signal using fft
    tr.data = np.real(scipy.ifft(
        scipy.fft(tr.data, n=Nfix)
        * np.exp(-1j*2*np.pi*freq*t0)))[0:N]


def timeshift_mat(M, t0: float, delta: float):
    """Takes in a ismic trace and shifts it in time using the fft."""

    N = M.shape[1]
    Nfix = nextpoweroftwo(N)

    # Get frequency vector
    freq = np.fft.fftfreq(Nfix, d=delta)

    # Compute timeshifted signal using fft
    M = np.real(scipy.ifft(
        scipy.fft(M, n=Nfix, axis=1)
        * np.exp(-1j*2*np.pi*freq*t0)))[:, 0:N]

    return M


def timeshift_trace_pad(tr: Trace, t0: float):
    """Takes in a seismic trace and shifts it in time using the fft."""

    N = len(tr.data)
    Nfix = nextpoweroftwo(N)

    # Get frequency vector
    freq = np.fft.fftfreq(Nfix, d=tr.stats.delta)

    # Compute timeshifted signal using fft
    tr.data = np.real(scipy.ifft(
        scipy.fft(tr.data, n=Nfix)
        * np.exp(-1j*2*np.pi*freq*t0)))[0:N]


def timeshift_trace(tr: Trace, t0: float):
    """Takes in a seismic trace and shifts it in time using the fft."""

    N = len(tr.data)

    # Get frequency vector
    freq = np.fft.fftfreq(N, d=tr.stats.delta)

    # Compute timeshifted signal using fft
    tr.data = np.real(scipy.ifft(
        scipy.fft(tr.data) * np.exp(-1j*2*np.pi*freq*t0)))


def timeshift_trace_roll_pad(tr: Trace, t0: float):
    """Takes in a seismic trace and shifts it in time using the fft."""

    nt0 = int(np.round(t0 / tr.stats.delta))

    # Compute timeshifted signal using roll
    tr.data = np.roll(np.pad(tr.data, (0, 1000), mode='constant',
                             constant_values=(0, 0)), nt0)[0:-1000]


def timeshift_trace_roll(tr: Trace, t0: float):
    """Takes in a seismic trace and shifts it in time using the fft."""

    nt0 = int(np.round(t0 / tr.stats.delta))

    # Compute timeshifted signal using roll
    tr.data = np.roll(tr.data, nt0)


def to_velocity(tr: Trace):
    """Converts trace to velocity.

    Arguments:
        tr: Obspy trace

    """

    # Get frequency vector
    freq = np.fft.fftfreq(tr.stats.npts, d=tr.stats.delta)

    tr.data = np.real(scipy.ifft(scipy.fft(tr.data) * 1j*np.pi*freq))
