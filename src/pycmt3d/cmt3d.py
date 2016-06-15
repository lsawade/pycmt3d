#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import)
import os
import numpy as np
from copy import deepcopy

from . import logger
from .util import random_select, sum_matrix, _float_array_to_str
from .util import _get_cmt_par, dump_json
from .measure import compute_derivatives, calculate_variance_on_trace
from .measure import compute_new_syn_on_trwin
from .plot_util import plot_seismograms, PlotInvSummary, PlotStats
from .data_container import MetaInfo
from .weight import Weight
from .constant import NPARMAX
from .solver import solver
from .log_util import print_inversion_summary


def generate_newcmtsource(oldcmt, new_cmt_par):
    """
    Convert new_cmt_par array to self.new_cmtsource

    :return:
    """
    newcmt = deepcopy(oldcmt)

    time_shift = new_cmt_par[9]
    new_cmt_time = oldcmt.origin_time + time_shift
    # copy old one
    attrlist = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp",
                "depth_in_m", "longitude", "latitude", "cmt_time",
                "half_duration"]

    for idx, attr in enumerate(attrlist):
        if attr == "cmt_time":
            val = new_cmt_time
        else:
            val = new_cmt_par[idx]
        setattr(newcmt, attr, val)

    return newcmt


class Cmt3D(object):
    """
    Class that handles the solver part of source inversion

    :param cmtsource: earthquake source
    :type cmtsource: :class:`pycmt3d.CMTSource`
    :param data_container: all data and window
    :type data_container: :class:`pycmt3d.DataContainer`
    :param config: configuration for source inversion
    :type config: :class:`pycmt3d.Config`
    """

    def __init__(self, cmtsource, data_container, config,
                 logfile="log.txt"):

        self.cmtsource = cmtsource
        self.data_container = data_container
        self.config = config
        self._logfile = logfile

        self.metas = []

        # new cmt par from the inversion
        self.new_cmtsource = None

        # variance information
        self.var_all = None
        self.var_all_new = None
        self.var_reduction = None

        # bootstrap stat var
        self.par_mean = np.zeros(NPARMAX)
        self.par_std = np.zeros(NPARMAX)
        self.par_var = np.zeros(NPARMAX)
        self.std_over_mean = np.zeros(self.par_mean.shape)

        self.print_cmtsource_summary()

    @property
    def cmt_par(self):
        """
        cmt array: [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, depth, lon, lat,
                    time_shift, half_duration]
        """
        return _get_cmt_par(self.cmtsource)

    @property
    def new_cmt_par(self):
        """
        New cmt param array
        """
        return _get_cmt_par(self.new_cmtsource)

    def setup_window_weight(self):
        """
        Use Window information to setup weight for each window.

        :returns:
        """
        if not self.config.weight_data:
            # only use the initial weight information provided by
            # the user in the window function. No extra weighting
            return

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

    def setup_measurement_matrix(self):
        """
        Calculate A and b for all windows

        :return:
        """
        logger.info("*" * 15)
        logger.info("Set up inversion matrix")

        self._init_metas()
        for meta, trwin in zip(self.metas, self.data_container):
            for win_idx in range(trwin.nwindows):
                # loop over each window
                # here, A and b are from raw measurements
                # and no weightings has been applied yet
                # Attention here, we use dcmt_par_scaled here; otherwise
                # A1 and b1 would be too small.
                Aw, bw, Ae, be = compute_derivatives(
                    trwin.datalist, trwin.windows[win_idx],
                    self.config.parlist, self.config.dcmt_par_scaled,
                    self.config.taper_type)

                meta.Aws.append(Aw)
                meta.bws.append(bw)
                meta.Aes.append(Ae)
                meta.bes.append(be)

    def invert_solver(self, A, b, envelope_flag=False):
        npar = self.config.npar
        cmt_par_scaled = self.cmt_par[:npar] / self.config.scale_vector
        new_cmt_par_scaled = \
            solver(npar, A, b, cmt_par_scaled, self.config.zero_trace,
                   self.config.double_couple, envelope_flag,
                   self.config.damping, self.config.max_nl_iter)

        new_cmt_par = self.cmt_par.copy()
        new_cmt_par[:npar] = \
            new_cmt_par_scaled * self.config.scale_vector[:npar]

        return generate_newcmtsource(self.cmtsource, new_cmt_par)

    def invert_cmt(self):
        """
        ensemble all measurements together to form Matrix A and vector
        b to solve the A * (dm) = b
        A is the Hessian Matrix and b is the misfit

        :return:
        """
        logger.info("*"*15)
        logger.info("CMT Inversion")
        logger.info("*"*15)

        npar = self.config.npar
        Aw_all = np.zeros([npar, npar])
        bw_all = np.zeros(npar)
        Ae_all = np.zeros([npar, npar])
        be_all = np.zeros(npar)
        # ensemble A and b
        for _meta in self.metas:
            Aw = sum_matrix(_meta.Aws, coef=_meta.weights)
            bw = sum_matrix(_meta.bws, coef=_meta.weights)
            Ae = sum_matrix(_meta.Aes, coef=_meta.weights)
            be = sum_matrix(_meta.bes, coef=_meta.weights)
            Aw_all += Aw
            bw_all += bw
            Ae_all += Ae
            be_all += be

        ecoef = self.config.envelope_coef
        A_all = (1 - ecoef) * Aw_all + ecoef * Ae_all
        b_all = (1 - ecoef) * bw_all + ecoef * be_all

        logger.info("Inversion Matrix A(with scaled cmt perturbation) is "
                    "as follows:")
        logger.info("\n%s" % ('\n'.join(map(_float_array_to_str, A_all))))
        logger.info("Inversion Matrix Aw(with scaled cmt perturbation) is "
                    "as follows:")
        logger.info("\n%s" % ('\n'.join(map(_float_array_to_str, Aw_all))))
        logger.info("Inversion Matrix Ae(with scaled cmt perturbation) is "
                    "as follows:")
        logger.info("\n%s" % ('\n'.join(map(_float_array_to_str, Ae_all))))
        logger.info("Condition number of A: %10.2f" % (np.linalg.cond(A_all)))
        logger.info("RHS vector b(with scaled cmt perturbation) is "
                    "as follows:")

        logger.info("b_all")
        logger.info(b_all)
        logger.info("[%s]" % (_float_array_to_str(b_all)))

        # source inversion
        logger.info("-" * 10 + " inversion " + "-" * 10)
        self.new_cmtsource = self.invert_solver(
            A_all, b_all, envelope_flag=(not np.isclose(ecoef, 0)))
        logger.info("-" * 10 + " waveform inversion " + "-" * 10)
        self.new_cmtsource_waveform = self.invert_solver(
            Aw_all, bw_all, envelope_flag=False)
        logger.info("-" * 10 + " envelope inversion " + "-" * 10)
        self.new_cmtsource_envelope = self.invert_solver(
            Ae_all, be_all, envelope_flag=True)
        logger.info("-" * 20)

    def invert_bootstrap(self):
        """
        It is used to evaluate the mean, standard deviation, and variance
        of new parameters

        :return:
        """
        A_bootstrap = []
        b_bootstrap = []
        n_subset = int(self.config.bootstrap_subset_ratio * self.nwins)
        for i in range(self.config.bootstrap_repeat):
            random_array = random_select(
                self.nwins, sample_number=n_subset)
            A = sum_matrix(random_array * self.weight_array, self.A1_all)
            b = sum_matrix(random_array * self.weight_array, self.b1_all)
            A_bootstrap.append(A)
            b_bootstrap.append(b)

        # inversion of each subset
        new_par_array = np.zeros((self.config.bootstrap_repeat, NPARMAX))
        for i in range(self.config.bootstrap_repeat):
            new_par = self.invert_solver(A_bootstrap[i], b_bootstrap[i])
            new_par_array[i, :] = new_par

        # statistical analysis
        self.par_mean = np.mean(new_par_array, axis=0)
        self.par_std = np.std(new_par_array, axis=0)
        self.par_var = np.var(new_par_array, axis=0)
        for _ii in range(self.par_mean.shape[0]):
            if self.par_mean[_ii] != 0:
                # in case of 0 value
                self.std_over_mean[_ii] = \
                        np.abs(self.par_std[_ii] / self.par_mean[_ii])
            else:
                self.std_over_mean[_ii] = 0.

    def source_inversion(self):
        """
        the Source Inversion method
        :return:
        """
        self.setup_measurement_matrix()
        # for meta in self.metas:
        #    print("A1:", meta.A1s[0])
        #    print("b1:", meta.b1s[0])
        self.setup_window_weight()
        self.invert_cmt()

        self.calculate_variance()

        # if self.config.bootstrap:
        #    self.invert_bootstrap()
        print_inversion_summary(self.config.npar, self.cmtsource,
                                self.new_cmtsource)

    def calculate_variance(self):
        """
        Calculate variance reduction based on old and new source solution

        :return:
        """
        var_all = 0.0
        var_all_new = 0.0

        self.stats_before = {}
        self.stats_after = {}

        self.compute_new_syn()

        # calculate metrics for each trwin
        for meta, trwin in zip(self.metas, self.data_container.trwins):
            obsd = trwin.datalist['obsd']
            synt = trwin.datalist['synt']

            # calculate old variance metrics
            [v1, d1, tshift1, cc1, dlnA1, cc_amp1] = \
                calculate_variance_on_trace(obsd, synt, trwin.windows,
                                            self.config.taper_type)
            meta.prov["synt"] = {"v": v1, "d": d1, "tshift": tshift1,
                                 "cc": cc1, "dlnA": dlnA1,
                                 "cc_amp": cc_amp1}

            new_synt = trwin.datalist['new_synt']
            # calculate new variance metrics
            [v2, d2, tshift2, cc2, dlnA2, cc_amp2] = \
                calculate_variance_on_trace(obsd, new_synt, trwin.windows,
                                            self.config.taper_type)
            meta.prov["new_synt"] = {"v": v2, "d": d2, "tshift": tshift2,
                                     "cc": cc2, "dlnA": dlnA2,
                                     "cc_amp": cc_amp2}

            var_all += np.sum(0.5 * v1 * meta.weights)
            var_all_new += np.sum(0.5 * v2 * meta.weights)

        logger.info(
            "Total Variance Reduced from %e to %e ===== %f %%"
            % (var_all, var_all_new, (var_all - var_all_new) / var_all * 100))

        self.var_all = var_all
        self.var_all_new = var_all_new
        self.var_reduction = (var_all - var_all_new) / var_all

    def calculate_kai_total_value(self):
        """
        Calculate the sum of kai value

        :return:
        """
        kai_before = {}
        kai_after = {}
        for tag in self.stats_before.keys():
            kai_before[tag] = np.sum(self.stats_before[tag][:, -1])
            kai_after[tag] = np.sum(self.stats_after[tag][:, -1])
        return kai_before, kai_after

    def compute_new_syn(self):
        """
        Compute new synthetic for each trwin in self.data_container
        """
        npar = self.config.npar
        dm_scaled = \
            (self.new_cmt_par[:npar] - self.cmt_par[:npar]) / \
            self.config.scale_vector
        for trwin in self.data_container:
            compute_new_syn_on_trwin(trwin.datalist, self.config.parlist,
                                     self.config.dcmt_par_scaled, dm_scaled)

    def write_new_cmtfile(self, outputdir="."):
        """
        Write new_cmtsource into a file
        """
        if self.config.double_couple:
            suffix = "ZT_DC"
        elif self.config.zero_trace:
            suffix = "ZT"
        else:
            suffix = "no_constraint"
        outputfn = "%s.%dp_%s.inv" % (
            self.cmtsource.eventname, self.config.npar, suffix)
        cmtfile = os.path.join(outputdir, outputfn)
        logger.info("New cmt file: %s" % cmtfile)

        self.new_cmtsource.write_CMTSOLUTION_file(cmtfile)

    def print_cmtsource_summary(self):
        """
        Print the inversion summary in the logger
        """
        logger.info("===== CMT Source Information =====")
        logger.info("\n%s" % self.cmtsource)

    def plot_summary(self, outputdir=".", figure_format="png",
                     mode="global"):
        """
        Plot inversion summary, including source parameter change,
        station distribution, and beach ball change.

        :param outputdir: output directory
        :return:
        """
        eventname = self.cmtsource.eventname
        npar = self.config.npar
        if self.config.double_couple:
            suffix = "ZT_DC"
        elif self.config.zero_trace:
            suffix = "ZT"
        else:
            suffix = "no_constraint"
        outputfn = "%s.%dp_%s.inv" % (eventname, npar, suffix)
        outputfn = os.path.join(outputdir, outputfn)
        figurename = outputfn + "." + figure_format

        logger.info("Source inversion summary figure: %s" % figurename)

        plot_util = PlotInvSummary(
            data_container=self.data_container, config=self.config,
            cmtsource=self.cmtsource,
            nregions=self.config.weight_config.azi_bins,
            new_cmtsource=self.new_cmtsource, bootstrap_mean=self.par_mean,
            bootstrap_std=self.par_std, var_reduction=self.var_reduction,
            mode=mode)
        plot_util.plot_inversion_summary(figurename=figurename)

    def plot_stats_histogram(self, outputdir=".", figure_format="png"):
        """
        Plot inversion histogram, including histograms of tshift, cc,
        dlnA, cc_amp, chi values before and after the inversion.

        :param outputdir:
        :return:
        """
        if self.config.double_couple:
            constr_str = "ZT_DC"
        elif self.config.zero_trace:
            constr_str = "ZT"
        else:
            constr_str = "no_constraint"

        if not self.config.weight_config.normalize_by_energy:
            prefix = "%dp_%s." % (self.config.npar, constr_str) + \
                "no_normener"
        else:
            prefix = "%dp_%s.%s" % (
                self.config.npar, constr_str, self.config.norm_mode)

        if not self.config.weight_config.normalize_by_category:
            prefix += ".no_normcat"
        else:
            prefix += ".normcat"
        figname = "%s.%s.dlnA.%s" % (self.cmtsource.eventname, prefix,
                                     figure_format)
        figname = os.path.join(outputdir, figname)

        plot_util = PlotStats(self.data_container, self.metas, figname)
        plot_util.plot_stats_histogram()

    def write_weight_log(self, filename):
        """
        write out weight log file
        """
        weights = {}
        for meta in self.metas:
            weights[meta.obsd_id] = meta.weights
        dump_json(weights, filename)

    def plot_new_synt_seismograms(self, outputdir, figure_format="png"):
        """
        Plot the new synthetic waveform
        """
        plot_seismograms(self.data_container, outputdir,
                         self.cmtsource, figure_format=figure_format)

    def write_new_syn(self, outputdir=".", file_format="sac"):
        """
        Write out the new synthetic waveform
        """
        logger.info("New synt output dir: %s" % outputdir)
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        if 'new_synt' not in self.data_container.traces[0].datalist.keys():
            raise ValueError("new synt not computed yet")

        eventname = self.cmtsource.eventname
        if self.config.double_couple:
            constr_str = "ZT_DC"
        elif self.config.zero_trace:
            constr_str = "ZT"
        else:
            constr_str = "no_constr"
        suffix = "%dp_%s" % (self.config.npar, constr_str)

        self.data_container.write_new_syn_file(
            file_format=file_format, outputdir=outputdir, eventname=eventname,
            suffix=suffix)
