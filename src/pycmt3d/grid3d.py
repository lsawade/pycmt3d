#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for grid search for origin time and scalar moment for CMT source

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function, division, absolute_import
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import multiprocessing
import psutil
from joblib import delayed
from joblib import Parallel

from . import logger
from .weight import Weight
from .data_container import MetaInfo
from .measure import calculate_variance_on_trace
from .measure import calculate_waveform_misfit_on_trace
from .measure import construct_matrices, compute_misfit, get_window_list
# from .plot_util import PlotStats
from .util import timeshift_trace
from .util import random_select
import time
from .gradient3d import Gradient

def del_line(n=1):
    """deletes one printed line."""
    time.sleep(0.001)
    sys.stdout.write("\033[F")  # back to previous line
    sys.stdout.write("\033[K")  # clear line
    sys.stdout.flush()


class Grid3dConfig(object):
    def __init__(self, origin_time_inv=True, time_start=-5.0, time_end=5.0,
                 dt_over_delta=1, energy_inv=True,
                 energy_start=0.8, energy_end=1.2, denergy=0.1,
                 energy_keys=None, energy_misfit_coef=None,
                 weight_data=False, weight_config=None, use_new=False,
                 taper_type="tukey", bootstrap=True, bootstrap_repeat=20,
                 bootstrap_subset_ratio=0.4):

        self.origin_time_inv = origin_time_inv
        self.time_start = time_start
        self.time_end = time_end
        self.dt_over_delta = dt_over_delta

        self.energy_inv = energy_inv
        self.energy_start = energy_start
        self.energy_end = energy_end
        self.denergy = denergy

        # Bootstrap parameters
        self.bootstrap = bootstrap
        self.bootstrap_repeat = bootstrap_repeat
        self.bootstrap_subset_ratio = bootstrap_subset_ratio

        # energy_keys could contain ["power_l1", "power_l2", "cc_amp", "chi"]
        if energy_keys is None:
            energy_keys = ["power_l1", "power_l2", "cc_amp"]
            if energy_misfit_coef is None:
                energy_misfit_coef = [0.75, 0.25, 1.0]
        else:
            if energy_misfit_coef is None:
                raise ValueError("energy_misfit_coef must be provided"
                                 "according to energy_keys")

        if len(energy_misfit_coef) != len(energy_keys):
            raise ValueError("Length of energy keys and coef must be"
                             " the same: %s, %s" %
                             (energy_keys, energy_misfit_coef))

        self.energy_keys = energy_keys
        self.energy_misfit_coef = np.array(energy_misfit_coef)

        self.weight_data = weight_data
        self.weight_config = weight_config

        self.taper_type = taper_type

        # If the data_container contains new_synthetic data, the new
        # synthetic data can be used for the gridsearch.
        self.use_new = use_new


class Grid3d(object):
    """
    Class that handle the grid search solver for origin time and moment scalar
    """
    def __init__(self, cmtsource, data_container, config):

        self.cmtsource = cmtsource
        self.data_container = data_container
        self.config = config

        self.metas = []

        self.new_cmtsource = None

        self.t00_best = None
        self.t00_misfit = None
        self.t00_array = None

        self.t00_mean = None
        self.t00_std = None
        self.t00_var = None

        self.m00_best = None
        self.m00_misfit = None
        self.m00_array = None

        self.m00_mean = None
        self.m00_std = None
        self.m00_var = None

        self.misfit_grid = None
        self.cat_misfit_grid = None

        self.var_all = None
        self.var_all_new = None
        self.var_reduction = None

    def setup_window_weight(self):
        """
        Use Window information to setup weight for each window.

        :returns:
        """
        self._init_metas()

        logger.info("*" * 15)
        logger.info("Start weighting...")
        weight_obj = Weight(self.cmtsource, self.data_container,
                            self.config.weight_config)
        weight_obj.setup_weight()

        for meta, weight_meta in zip(self.metas, weight_obj.metas):
            if meta.obsd_id != weight_meta.obsd_id or \
                    meta.synt_id != weight_meta.synt_id:
                raise ValueError("weight.metas and self.metas are different"
                                 "on meta: %s %s" % (meta.id, weight_meta.id))
            meta.weights = weight_meta.weights
            meta.prov.update(weight_meta.prov)

    def _init_metas(self):
        """
        Initialize the self.metas list. Keep the same order with the
        data container
        """
        for trwin in self.data_container:
            metainfo = MetaInfo(obsd_id=trwin.obsd_id, synt_id=trwin.synt_id,
                                weights=trwin.init_weight, Aws=[], bws=[],
                                Aes=[], bes=[], prov={})
            self.metas.append(metainfo)

    def search(self):
        """Searches for origin time first and then for the energy"""

        # This sets up everything
        self.setup_window_weight()

        # Searches for origin time
        self.grid_search_origin_time()

        # Searches for energy
        self.grid_search_energy()

        self.prepare_new_cmtsource()
        self.prepare_new_synthetic()

    def grid_search(self):
        """Searches for origin time and energy simultaneously over a grid of
        time shifts and moments."""

        # Setup window weights
        self.setup_window_weight()

        # Run grid search
        self.grid_search_taM()

        self.grid_search_origin_time()
        self.grid_search_energy()

        # Run bootstrap statistic
        # self.grid_search_bootstrap()

        # Output new data
        self.prepare_new_cmtsource()
        self.prepare_new_synthetic()

        # Calculate variance reduction
        self.calculate_variance()


    def prepare_new_cmtsource(self):
        newcmt = deepcopy(self.cmtsource)

        logger.info("Preparing new cmtsource...")
        if self.config.origin_time_inv:
            newcmt.cmt_time += self.t00_best
            logger.info("\tAdding time shift to cmt origin time:"
                        "%s + %fsec= %s"
                        % (self.cmtsource.cmt_time, self.t00_best,
                           newcmt.cmt_time))

        if self.config.energy_inv:
            attrs = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
            for attr in attrs:
                newval = self.m00_best * getattr(newcmt, attr)
                setattr(newcmt, attr, newval)
            logger.info("\tMultiply scalar moment change by %f%%"
                        % (self.m00_best * 100))

        self.new_cmtsource = newcmt

    def prepare_new_synthetic(self):
        logger.info("Reconstruct new synthetic seismograms...")
        for trwin in self.data_container:
            if self.config.use_new:
                if "new_synt" not in trwin.datalist:
                    raise ValueError("new synt is not in trwin(%s) "
                                     "datalist: %s"
                                     % (trwin, trwin.datalist.keys()))
                else:
                    new_synt = trwin.datalist["new_synt"].copy()
            else:
                new_synt = trwin.datalist["synt"].copy()

            if self.config.origin_time_inv:
                new_synt.stats.starttime += self.t00_best
            if self.config.energy_inv:
                new_synt.data *= self.m00_best
            trwin.datalist["new_synt"] = new_synt

        for meta, trwin in zip(self.metas, self.data_container):
            obsd = trwin.datalist["obsd"]
            new_synt = trwin.datalist["new_synt"]
            meta.prov["new_synt"] = \
                calculate_variance_on_trace(obsd, new_synt, trwin.windows,
                                            self.config.taper_type)
            # because calculate_variance_on_trace assumes obsd and new_synt
            # starting at the same time(which is not the case since we
            # correct the starting time of new_synt)
            if self.config.origin_time_inv:
                meta.prov["new_synt"]["tshift"] -= self.t00_best

    def calculate_tshift(self):
        """
        This step actually calculate the whole measurements between
        obsd and synt
        """
        for meta, trwin in zip(self.metas, self.data_container):
            obsd = trwin.datalist["obsd"]
            synt = trwin.datalist["synt"]
            meta.prov["synt"] = \
                calculate_variance_on_trace(obsd, synt, trwin.windows,
                                            self.config.taper_type)

    def grid_search_origin_time(self):

        logger.info("Origin time grid search")
        self.calculate_tshift()

        t00_s = self.config.time_start
        t00_e = self.config.time_end
        dt00 = self.config.dt_over_delta * \
            self.data_container[0].datalist['obsd'].stats.delta

        logger.info("Grid search time start and end: [%8.3f, %8.3f]"
                    % (t00_s, t00_e))
        logger.info("Grid search time interval:%10.3f" % dt00)

        tshifts = []
        weights = []
        for meta in self.metas:
            tshifts.extend(meta.prov["synt"]["tshift"])
            weights.extend(meta.weights)
        tshifts = np.array(tshifts)

        if self.config.weight_data:
            weights = np.array(weights)
        else:
            weights = np.ones(len(tshifts))

        # Grid search vector
        t00_array = np.arange(t00_s, t00_e+dt00, dt00)
        nt00 = t00_array.shape[0]
        final_misfits = np.zeros(nt00)

        for i in range(nt00):
            t00 = t00_array[i]
            final_misfits[i] = np.sum(weights * (tshifts - t00) ** 2)

        min_idx = final_misfits.argmin()
        t00_best = t00_array[min_idx]

        logger.info("Minimum t00(relative to cmt origin time): %6.3f"
                    % t00_best)
        if min_idx == 0 or min_idx == (nt00 - 1):
            logger.warning("Origin time search hit boundary, which means"
                           "search range should be reset")
        self.time_best = t00_best
        # self.t00_best = t00_best
        # self.t00_array = t00_array
        # self.t00_misfit = final_misfits

    def calculate_misfit_for_m00(self, m00):
        """Computes misfit for amplitude scaling."""
        power_l1s = []
        power_l2s = []
        cc_amps = []
        chis = []

        for trwin in self.data_container:
            obsd = trwin.datalist["obsd"].copy()

            if self.config.use_new:
                if "new_synt" not in trwin.datalist:
                    raise ValueError("new synt is not in trwin(%s) "
                                     "datalist: %s"
                                     % (trwin, trwin.datalist.keys()))
                else:
                    synt = trwin.datalist["new_synt"].copy()
            else:
                synt = trwin.datalist["synt"].copy()

            # Using the updated timeshift
            timeshift_trace(synt, self.time_best)

            synt.data *= m00
            measures = \
                calculate_variance_on_trace(obsd, synt, trwin.windows)
            power_l1s.extend(measures["power_l1"])
            power_l2s.extend(measures["power_l2"])
            cc_amps.extend(measures["cc_amp"])
            chis.extend(measures["chi"])

        measures = {"power_l1": np.array(power_l1s),
                    "power_l2": np.array(power_l2s),
                    "cc_amp": np.array(cc_amps),
                    "chi": np.array(chis)}
        return measures

    def calculate_misfit_on_grid(self, m00, t0, weights, counter, N):
        """Computes amplitude and cross correlation misfit for moment scaling
        as well as timeshift."""

        logger.info("%d/%d" % (counter, N))

        misfits = []

        for trwin in self.data_container:
            obsd = trwin.datalist["obsd"].copy()

            if self.config.use_new:
                if "new_synt" not in trwin.datalist:
                    raise ValueError("new synt is not in trwin(%s) "
                                     "datalist: %s"
                                     % (trwin, trwin.datalist.keys()))
                else:
                    synt = trwin.datalist["new_synt"].copy()
            else:
                synt = trwin.datalist["synt"].copy()

            # Shift trace in time
            timeshift_trace(synt, t0)

            # Scale amplitude
            synt.data *= m00
            measures = calculate_waveform_misfit_on_trace(obsd, synt,
                                                          trwin.windows)
            misfits.extend(measures)

        misfit = np.sum(np.array(misfits) * weights)
        return misfit


    def calculate_misfit_on_subset(self, m00, t0, subset, weights):
        """Computes misfit for amplitude scaling as well as timeshift."""

        misfits = []

        for k, trwin in enumerate(self.data_container):
            if subset[k] != 0:
                obsd = trwin.datalist["obsd"]

                if self.config.use_new:
                    if "new_synt" not in trwin.datalist:
                        raise ValueError("new synt is not in trwin(%s) "
                                         "datalist: %s"
                                         % (trwin, trwin.datalist.keys()))
                    else:
                        synt = trwin.datalist["new_synt"].copy()
                else:
                    synt = trwin.datalist["synt"].copy()

                # Shift trace in time
                timeshift_trace(synt, t0)

                # Scale amplitude
                synt.data *= m00
                measures = calculate_waveform_misfit_on_trace(obsd, synt,
                                                              trwin.windows)

                misfits.extend(measures)

        new_weights = self.get_bootstrap_weights(weights,
                                                     subset)

        misfit = np.sum(np.array(misfits) * new_weights)

        return misfit

    def grid_search_taM(self):
        """Grid search over time and Magnitude."""

        logger.info('Grid Search:')

        # Moment parameters
        m00_s = self.config.energy_start
        m00_e = self.config.energy_end
        dm00 = self.config.denergy
        logger.info("Energy Start and End: [%6.3f, %6.3f]"
                    % (m00_s, m00_e))
        logger.info("Energy Interval: %6.3f" % dm00)

        # Energy vector
        self.m00_array = np.arange(m00_s, m00_e + dm00, dm00)
        nm00 = self.m00_array.shape[0]

        # Timeshift parameters
        t00_s = self.config.time_start
        t00_e = self.config.time_end
        dt00 = self.config.dt_over_delta * \
            self.data_container[0].datalist['obsd'].stats.delta
        print(self.data_container[0].datalist['obsd'].stats.delta)
        logger.info("Time Start and End: [%8.3f, %8.3f]"
                    % (t00_s, t00_e))
        logger.info("Time Interval:%10.3f" % dt00)

        # Grid search vector
        self.t00_array = np.arange(t00_s, t00_e + dt00, dt00)
        nt00 = self.t00_array.shape[0]

        # Empty misfit grid
        final_misfits = np.zeros((nt00, nm00))

        cat_misfits = {}
        for key in self.config.energy_keys:
            cat_misfits[key] = np.zeros((nt00, nm00))

        if self.config.weight_data:
            weights = []
            for meta in self.metas:
                weights.extend(meta.weights)
            weights = np.array(weights)
        else:
            if self.data_container.nwindows:
                weights = np.ones(self.data_container.nwindows)
            else:
                weights = np.ones(len(self.data_container.nwindows))

        obsd, synt, delta, tapers = construct_matrices(
            self.data_container, weights, self.config.use_new)

        logger.info("Gradient search:\n--------------------")
        G = Gradient(obsd, synt, tapers, delta, method="n", crit=0.01, 
                     nt=20, nls=20)
        G.gradient()

        self.dtf = G.dt
        self.af = G.a
        self.dt_list = G.dt_list
        self.a_list = G.a_list
        self.c_list = G.cost_list

        logger.info("Timeshift found: %f" % G.dt)
        logger.info("Amplitude found: %f" % G.a)
        logger.info("Gradient iterations: %f" % G.it)

        logger.info("Gauss Newton:\n--------------------")
        GN = Gradient(obsd, synt, tapers, delta, method="gn", crit=0.01, 
                     nt=20, nls=20)
        GN.gradient()

        self.gndtf = GN.dt
        self.gnaf = GN.a
        self.gndt_list = GN.dt_list
        self.gna_list = GN.a_list
        self.gnc_list = GN.cost_list

        self.m00_best = self.gnaf
        self.t00_best = self.gndtf

        logger.info("Timeshift found: %f" % GN.dt)
        logger.info("Amplitude found: %f" % GN.a)
        logger.info("Gradient iterations: %f" % GN.it)


        # exit()
        N = nt00*nm00
        logger.info("Looping ....")
        logger.info("Number of iterations: %d" % N)
        counter = 0
        num_cores = psutil.cpu_count(logical=False)
        logger.info("Number of cores: %d" % num_cores)
        if num_cores == 1:
            for i in range(nt00):
                for j in range(nm00):
                    counter += 1
                    t00 = self.t00_array[i]
                    m00 = self.m00_array[j]

                    final_misfits[i, j] = compute_misfit(obsd, synt, tapers,
                                                         m00, t00, delta,
                                                         counter, N)
                    print("misfit: ", final_misfits[i, j])

        else:
            tt, mm = np.meshgrid(self.t00_array, self.m00_array)
            shape = tt.shape

            results = Parallel(n_jobs=num_cores,
                               backend='multiprocessing')(delayed(
                compute_misfit)(
                obsd, synt, tapers, mm0, tt0,
                delta, _i+1, N) for _i, (tt0, mm0) in
                enumerate(zip(tt.flatten(), mm.flatten())))

            final_misfits = np.array(results).reshape(shape).T

        logger.info("Done!")
        # find minimum
        min_idx = final_misfits.argmin()
        min_t_idx, min_M_idx = np.unravel_index(min_idx, final_misfits.shape)

        # Get Values
        t00_best = self.t00_array[min_t_idx]
        m00_best = self.m00_array[min_M_idx]

        if min_t_idx == 0 or min_t_idx == (nt00 - 1):
            logger.warning("Time search reaches boundary, which means the"
                           "search range should be reset.")
        logger.info("Best t0: %2.2f s" % t00_best)

        if min_M_idx == 0 or min_M_idx == (nm00 - 1):
            logger.warning("Energy search reaches boundary, which means the"
                           "search range should be reset.")
        logger.info("Best M0: %6.3f" % m00_best)

        self.m00_best_grid = m00_best
        self.t00_best_grid = t00_best

        self.misfit_grid = final_misfits
        # self.cat_misfit_grid = cat_misfits

    def get_bootstrap_weights(self, weights, random_array):
        """ Uses the random_array to get the weights corresponding to the
        subselected windows.

        :param weights:
        :param random_array:
        :return:
        """

        choice_array = []
        for k, trwin in enumerate(self.data_container):
            for l in range(trwin.nwindows):
                if random_array[k]:
                    choice_array.append(True)
                else:
                    choice_array.append(False)

        return weights[choice_array]

    def grid_search_bootstrap(self):
        """Grid search over time and Magnitude multiple times with subsets to
        evaluate standard deviation, mean, and variance"""

        logger.info('Bootstrap Grid Search:')

        # Moment parameters
        m00_s = self.config.energy_start
        m00_e = self.config.energy_end
        dm00 = self.config.denergy
        logger.info("Energy Start and End: [%6.3f, %6.3f]"
                    % (m00_s, m00_e))
        logger.info("Energy Interval: %6.3f" % dm00)

        # Energy vector
        m00_array = np.arange(m00_s, m00_e + dm00, dm00)
        nm00 = m00_array.shape[0]

        # Timeshift parameters
        t00_s = self.config.time_start
        t00_e = self.config.time_end
        dt00 = self.config.dt_over_delta * \
            self.data_container[0].datalist['obsd'].stats.delta

        logger.info("Time Start and End: [%8.3f, %8.3f]"
                    % (t00_s, t00_e))
        logger.info("Time Interval:%10.3f" % dt00)

        # Grid search vector
        t00_array = np.arange(t00_s, t00_e + dt00, dt00)
        nt00 = t00_array.shape[0]

        # Setup Bootstrap inversion
        ntrwins = len(self.data_container)
        print(ntrwins)
        n_subset = \
            max(int(self.config.bootstrap_subset_ratio * ntrwins), 1)
        logger.info("Bootstrap Repeat: %d  Subset Ratio: %f Nsub: %d"
                    % (self.config.bootstrap_repeat,
                       self.config.bootstrap_subset_ratio, n_subset))

        # Empty misfit grid
        final_misfits = np.zeros((self.config.bootstrap_repeat, nt00, nm00))

        cat_misfits = {}
        for key in self.config.energy_keys:
            cat_misfits[key] = np.zeros((self.config.bootstrap_repeat,
                                         nt00, nm00))

        if self.config.weight_data:
            weights = []
            for meta in self.metas:
                weights.extend(meta.weights)
            weights = np.array(weights)
        else:
            if self.data_container.nwindows:
                weights = np.ones(self.data_container.nwindows)
            else:
                weights = np.ones(len(self.data_container.nwindows))

        # Create array for best values
        t00_best_array = np.zeros(self.config.bootstrap_repeat)
        m00_best_array = np.zeros(self.config.bootstrap_repeat)


        logger.info("Looping ....")
        timer0 = time.time()
        num_cores = psutil.cpu_count(logical = False)
        logger.info("Number of cores: %d" % num_cores)

        for k in range(self.config.bootstrap_repeat):

            logger.info("Bootstrap: %d/%d" % (k, self.config.bootstrap_repeat))
            random_array = random_select(
                ntrwins, nselected=n_subset, replace=0)
            if num_cores == 1:
                for i in range(nt00):
                    for j in range(nm00):
                        t00 = t00_array[i]
                        m00 = m00_array[j]
                        final_misfits[k, i, j] = \
                            self.calculate_misfit_on_subset(
                            m00, t00, random_array, weights)

            else:
                tt, mm = np.meshgrid(self.t00_array, self.m00_array)
                shape = tt.shape

                results = Parallel(n_jobs=num_cores,
                                   backend='multiprocessing')(
                    delayed(self.calculate_misfit_on_subset)(
                        tt0, mm0, random_array, weights) for (tt0, mm0) in
                    zip(tt.flatten(), mm.flatten()))

                final_misfits[k, :, :] = np.array(results).reshape(shape).T


            # find minimum
            min_idx = final_misfits[k, :, :].argmin()
            min_t_idx, min_M_idx = np.unravel_index(
                min_idx, final_misfits[k, :, :].shape)

            # Get Values
            t00_best_array[k] = t00_array[min_t_idx]
            m00_best_array[k] = m00_array[min_M_idx]

            if min_t_idx == 0 or min_t_idx == (nt00 - 1):
                logger.warning("Time search reaches boundary, which means the"
                               "search range should be reset.")
            logger.info("Best t0: %2.2f s" % t00_best_array[k])

            if min_M_idx == 0 or min_M_idx == (nm00 - 1):
                logger.warning("Energy search reaches boundary, "
                               "which means the"
                               "search range should be reset.")
            logger.info("Best M0: %6.3f" % m00_best_array[k])

        logger.info("Done!")
        logger.info("Time elapsed: %4.f\n" % (time.time() - timer0))

        self.m00_std = np.std(m00_best_array) * self.cmtsource.M0
        self.m00_var = np.var(m00_best_array) * self.cmtsource.M0
        self.m00_mean = np.mean(m00_best_array) * self.cmtsource.M0
        self.t00_std = np.std(t00_best_array)
        self.t00_var = np.var(t00_best_array)
        self.t00_mean = np.mean(t00_best_array) + self.cmtsource.time_shift

        logger.info("t0 stats: Mean: %2.2f s - STD: %2.2f s - Var: %3.4f "
                    "s**2" % (self.t00_mean, self.t00_std, self.t00_var))
        logger.info("M0 stats: Mean: %4e - STD: %4e s - Var: %4e"
                    "s**2" % (self.m00_mean, self.m00_std, self.m00_var))

    def grid_search_energy(self):
        """Searches for Energy only """

        logger.info('Energy grid Search')

        m00_s = self.config.energy_start
        m00_e = self.config.energy_end
        dm00 = self.config.denergy
        logger.info("Grid search energy start and end: [%6.3f, %6.3f]"
                    % (m00_s, m00_e))
        logger.info("Grid search energy interval: %6.3f" % dm00)

        m00_array = np.arange(m00_s, m00_e+dm00, dm00)
        nm00 = m00_array.shape[0]

        final_misfits = np.zeros(nm00)
        cat_misfits = {}
        for key in self.config.energy_keys:
            cat_misfits[key] = np.zeros(nm00)

        if self.config.weight_data:
            weights = []
            for meta in self.metas:
                weights.extend(meta.weights)
            weights = np.array(weights)
        else:
            if self.data_container.nwindows:
                weights = np.ones(self.data_container.nwindows)
            else:
                weights = np.ones(len(self.data_container.nwindows))

        for i in range(nm00):
            m00 = m00_array[i]
            logger.info("Looping on m00: %f" % m00)
            measures = \
                self.calculate_misfit_for_m00(m00)

            if self.config.energy_keys != "None":
                for key_idx, key in enumerate(self.config.energy_keys):
                    cat_val = np.sqrt(np.sum(measures[key]**2 * weights))
                    cat_misfits[key][i] = cat_val
                    final_misfits[i] += \
                        self.config.energy_misfit_coef[key_idx] * cat_val

        # find minimum
        min_idx = final_misfits.argmin()
        m00_best = m00_array[min_idx]

        if min_idx == 0 or min_idx == (nm00 - 1):
            logger.warning("Energy search reaches boundary, which means the"
                           "search range should be reset")
        logger.info("best m00: %6.3f" % m00_best)

        self.energy_best = m00_best
        # self.m00_best = m00_best
        # self.m00_array = m00_array
        # self.m00_misfit = final_misfits
        # self.m00_cat_misfit = cat_misfits

    def write_new_cmtfile(self, outputdir="."):
        suffix = "grid"
        if self.config.origin_time_inv:
            suffix += ".time"
        if self.config.energy_inv:
            suffix += ".energy"
        fn = os.path.join(outputdir, "%s.%s.inv" % (self.cmtsource.eventname,
                                                    suffix))
        logger.info("New cmtsource file: %s" % fn)
        self.new_cmtsource.write_CMTSOLUTION_file(fn)

    def calculate_variance(self):
        """ Computes the variance reduction"""
        var_all = 0.0
        var_all_new = 0.0

        # calculate metrics for each trwin
        for meta, trwin in zip(self.metas, self.data_container.trwins):
            obsd = trwin.datalist['obsd']
            synt = trwin.datalist['synt']

            # calculate old variance metrics
            meta.prov["synt"] = \
                calculate_variance_on_trace(obsd, synt, trwin.windows,
                                            self.config.taper_type)

            new_synt = trwin.datalist['new_synt']
            # calculate new variance metrics
            meta.prov["new_synt"] = \
                calculate_variance_on_trace(obsd, new_synt, trwin.windows,
                                            self.config.taper_type)

            var_all += np.sum(0.5 * meta.prov["synt"]["chi"] * meta.weights)
            var_all_new += np.sum(0.5 * meta.prov["new_synt"]["chi"]
                                  * meta.weights)

        logger.info(
            "Total Variance Reduced from %e to %e ===== %f %%"
            % (var_all, var_all_new, (var_all - var_all_new) / var_all * 100))
        logger.info("*" * 20)

        self.var_all = var_all
        self.var_all_new = var_all_new
        self.var_reduction = (var_all - var_all_new) / var_all

    def plot_stats_histogram(self, outputdir=".", figure_format="pdf"):
        """
        Plot the histogram of meansurements inside windows for
        old and new synthetic seismograms
        """
        figname = os.path.join(outputdir, "window.stats.%s" % figure_format)
        plot_util = PlotStats(self.data_container, self.metas, figname)
        plot_util.plot_stats_histogram()

    def plot_misfit_summary(self, outputdir=".", figure_format="pdf"):
        """
        Plot histogram and misfit curve of origin time result

        :param outputdir:
        :return:
        """
        figname = os.path.join(outputdir, "tshift.misfit.%s" % figure_format)
        logger.info("tshift misfit figure: %s" % figname)
        self.plot_tshift_misfit_summary(figname)

        figname = os.path.join(outputdir, "energy.misfit.%s" % figure_format)
        logger.info("energy misfit figure: %s" % figname)
        self.plot_energy_misfit_summary(figname)

    def plot_tshift_misfit_summary(self, figname):
        plt.figure()
        plt.plot(self.t00_array, self.t00_misfit, label="misfit values")

        idx = np.where(self.t00_array == self.t00_best)[0]
        plt.plot(self.t00_best, self.t00_misfit[idx], "r*",
                 markersize=10, label="min misfit")

        plt.xlabel("time shift(sec)")
        plt.ylabel("misfit values")
        plt.grid()
        plt.legend(numpoints=1)
        plt.savefig(figname)

    def plot_energy_misfit_summary(self, figname):
        """
        Plot histogram of dlnA

        :param outputdir:
        :return:
        """
        keys = self.config.energy_keys
        nkeys = len(keys)
        ncols = nkeys + 1
        plt.figure(figsize=(5*ncols, 5))

        min_idx = np.where(self.m00_array == self.m00_best)[0]

        for idx, key in enumerate(self.config.energy_keys):
            plt.subplot(1, nkeys+1, idx+1)
            plt.plot(self.m00_array, self.m00_cat_misfit[key], label="misfit")
            plt.plot(self.m00_best, self.m00_cat_misfit[key][min_idx], "r*",
                     markersize=10, label="min misfit")
            plt.xlabel("scalar moment")
            plt.ylabel("%s misift" % key)
            plt.legend(numpoints=1)
            plt.grid()

        plt.subplot(1, nkeys+1, nkeys+1)
        plt.plot(self.m00_array, self.m00_misfit, label="misfit values")

        plt.plot(self.m00_best, self.m00_misfit[min_idx], "r*",
                 markersize=10, label="min misfit")

        plt.xlabel("scalar moment change")
        plt.ylabel("Overall misfit")
        plt.grid()
        plt.legend(numpoints=1)
        plt.tight_layout()
        plt.savefig(figname)

    def plot_grid(self, figurename=None):

        min_m_idx = np.where(self.m00_array == self.m00_best)[0]
        min_t_idx = np.where(self.t00_array == self.t00_best)[0]

        tt, mm = np.meshgrid(self.t00_array, self.m00_array)

        if figurename is None:
            ax = plt.gca()

        else:
            plt.figure()
            ax = plt.axes()

        ctf = ax.pcolormesh(tt, (mm - 1) * 100,
                           np.log10(self.misfit_grid.T),
                           cmap='Greys', edgecolor=None)
        ax.plot(self.time_best * np.ones_like(self.m00_array),
                (self.m00_array - 1) * 100, "k--",
                self.t00_array,
                (self.energy_best * np.ones_like(self.t00_array) - 1) * 100,
                "k--", lw=0.5)
        ax.plot(0, 0, "ks", markeredgecolor='k',
                markersize=5, label="start", zorder=10)
        ax.plot(self.t00_best_grid,
                (self.m00_best_grid - 1) * 100,
                "w*", markeredgecolor='k',
                 markersize=10, label="grid")
        ax.plot(np.array(self.dt_list), (np.array(self.a_list) - 1.) * 100.,
                "r")
        ax.plot(np.array(self.gndt_list), (np.array(self.gna_list) - 1.) * 100.,
                "b")
        ax.plot(self.dtf,
                (self.af - 1.) * 100.,
                "r*", markeredgecolor='k',
                markersize=10, label="Hessian")
        ax.plot(self.gndtf,
                (self.gnaf - 1.) * 100.,
                "b*", markeredgecolor='k',
                markersize=10, label="GN")
        ax.plot(self.time_best, (self.energy_best - 1) * 100,
                "k*", markeredgecolor='k',
                markersize=10, label="Line")


        plt.legend(prop={'size': 6}, fancybox=False, framealpha=1)

        plt.colorbar(ctf)
        ax.set_xlabel("$\\Delta t$")
        ax.set_ylabel("% Change in $M_0$")

        if figurename is None:
            pass
        elif figurename == 'show':
            plt.show()
        else:
            plt.savefig(figurename)

    def plot_cost(self, figurename=None):

        if figurename is None:
            ax = plt.gca()

        else:
            plt.figure()
            ax = plt.axes()

        ax.plot(self.c_list, "r",
                label="Hessian ($\mathcal{C}_{min} = %.3f$)" % self.c_list[-1])
        ax.plot(self.gnc_list, "b",
                label="GN  ($\mathcal{C}_{min} = %.3f)$" % self.gnc_list[-1])

        plt.legend(prop={'size': 6}, fancybox=False, framealpha=1)

        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Misfit reduction")

        if figurename is None:
            pass
        elif figurename == 'show':
            plt.show()
        else:
            plt.savefig(figurename)
            