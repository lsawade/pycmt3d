#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for Gauss Newton and Hessian optimization of
the scalar moment and timeshift.
 To run this code in mpi mode the child function
 "grad_child_mpi.py" is essential!

:copyright:
    Lucas Sawade (lsawade@princeton.edu), 2020
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""

from __future__ import print_function, division, absolute_import
import os
import sys
import numpy as np
from copy import deepcopy
import time
import psutil
from joblib import delayed
from joblib import Parallel
import matplotlib.pyplot as plt
from collections import defaultdict

# Internal imports
from .source import CMTSource
from .data_container import DataContainer
from .data_container import MetaInfo
from . import logger
from .weight import Weight, setup_energy_weight
from .measure import calculate_variance_on_trace, pad
from .util import timeshift_mat, timeshift_trace_pad
from .util import get_window_idx
from .util import get_trwin_tag
from .util import construct_taper
from .util import dump_json
from .mpi_utils import broadcast_dict
from .mpi_utils import get_result_dictionaries
from .mpi_utils import split
from . import plot_util


plt.switch_backend('agg')


def get_number_of_cores(bootstrap):
    """Returns the number of appropriate cores
    for spawning the MPI process"""

    # Use psutil to get physical cores available on node
    # -3 because psutil will give all cores, but the main script
    # is already occupying one core (sounds like it should be -1
    # but we do -3 for safety)
    avail = psutil.cpu_count(logical=False) - 3

    if avail < 1:
        avail = 1

    return min([avail, bootstrap])


def check_wolfe(alpha, fcost, fcost_new, q, qnew, c1=1e-4, c2=0.9,
                strong=False):
    """

    :param alpha: step length
    :param fcost: previous cost
    :param fcost_new: new cost
    :param q: descent direction * gradient previous
    :param qnew: descent direction * gradient previous next
    :param c1:
    :param c2:
    :param strong:
    :return:
    """

    # Init wolfe boolean
    w1 = False
    w2 = False
    w3 = True

    # Check descent direction
    if q > 0:
        w3 = False
        # raise ValueError('Not a descent dir');

    # Check first wolfe
    if fcost_new <= fcost + c1 * alpha * q:
        w1 = True

    # Check second wolfe
    if strong is False:
        if qnew >= c2 * q:
            w2 = True

    else:
        if np.abs(qnew) >= np.abs(c2 * q):
            w2 = True

    return w1, w2, w3


def update_alpha(alpha, al, ar, wolfe, factor=1):
    w1 = wolfe[0]
    w2 = wolfe[1]
    w3 = wolfe[2]

    good = False

    # Manage alpha in consequence
    if w3 is False:  # not a descent direction... quit
        raise ValueError('Not a descent direction... STOP')

    if w1 is True and w2 is True:  # both are satisfied then terminate
        good = True

    elif w1 is False:  # not a sufficient decrease, we've been too far
        ar = alpha
        alpha = (al + ar) * 0.5

    elif w1 is True and w2 is False:
        al = alpha
        if ar > 0:  # sufficient decrease but too close already backeted
            # decrease in interval
            alpha = (al + ar) * 0.5
        else:  # sufficient decrease but too close, then increase a
            alpha = factor * alpha

    return good, alpha, al, ar


def precondition(B, g):
    # Preconditioning
    fac = 1 / np.array([B[0, 0], B[1, 1]])
    H = np.diag(fac)
    return H, g


def damping(B, damping=0.01):
    """Takes in a matrix before inverting it and adds the trace to the diagonal
    elements
    :param B: Matrix to invert
    :param damping: damping factor. Default 0.01
    :return:
    """
    return B + damping * np.trace(B) * np.eye(B.shape[0])


def reguralization(B):
    lb = np.median(B)
    lbm = np.eye(2) * lb

    return B + lbm


class Gradient3dConfig(object):
    def __init__(self, method="gn", weight_data=False, weight_config=None,
                 taper_type="tukey",
                 c1=1e-4, c2=0.9,
                 idt: float = 0.0, ia=1.,
                 nt: int = 50, nls: int = 20,
                 crit: float = 0.01,
                 precond: bool = False, reg: bool = False,
                 damping=None,
                 bootstrap=True, bootstrap_repeat=20,
                 bootstrap_subset_ratio=0.4,
                 parallel: bool = True, mpi_env: bool = True):
        """Configuration for the gradient method.

        Args:
            method (str): descent algorithm. Defaults to "gn".
            weight_data (bool,optional): contains the weightdata.
                                         Defaults to False.
            weight_config ([type], optional): contains the weightdata
                                              configuration. Defaults to None.
            taper_type (str, optional): Taper for window selection.
                                        Defaults to "tukey".
            c1 (float, optional): Wolfe condition parameter. Defaults to 1e-4
            c2 (float, optional): Wolfe condition parameter. Defaults to 0.9
            idt (float, optional): Starting value for the timeshift.
                                   Defaults to 0.0
            ia (float, optional): Starting value for the moment scaling
                                  factor. Defaults to 1.0
            nt (int, optional): Maximum number of iterations. Defaults to 50.
            nls (int, optional): Maximum number of linesearch iterations.
                                 Defaults to 20.
            crit (float, optional): critical misfit reduction.
                                    Defaults to 1e-3.
            precond (bool, optional): If True uses preconditioner on
                                      the Hessian. Defaults to False.
            reg (bool, optional): If True reguralizes the Hessian.
                                  Defaults to False.
            bootstrap (bool, optional): Whether to perform a bootstrap
                                        statistic. Defaults to True.
            bootstrap_parallel (int, optional): whether to perform the
                                                bootstrap
                                                statistic in parallel.
                                                0 = serial,
                                                1 = parallel and let the
                                                program figure out
                                                the number of cores.
                                                all other integers
                                                define the number of cores.
            bootstrap_repeat (int, optional): number of repeats to be done.
                                              Defaults to 20.
            bootstrap_subset_ratio (float, optional): taking a certain ratio of
                                                      the available
                                                      data to compute a
                                                      bootstrap statistic.
                                                      Defaults to 0.4.

        Raises:
            ValueError: if wrong method string is input.
        """

        # Method of choice
        if method in ["gn", "n"]:
            self.method = method
        else:
            raise ValueError("Chosen method not supported.")

        # Start values
        self.idt = idt
        self.ia = ia

        # Method parameters
        self.nt = nt
        self.nls = nls
        self.crit = crit
        self.damping = damping
        self.reg = reg
        self.precond = precond

        # Wolfe paramters
        self.c1 = c1
        self.c2 = c2

        # MPI
        self.parallel = parallel
        self.mpi_env = mpi_env

        # Bootstrap parameters
        self.bootstrap = bootstrap
        self.bootstrap_repeat = bootstrap_repeat
        self.bootstrap_subset_ratio = bootstrap_subset_ratio

        # Weightdata
        self.weight_data = weight_data
        self.weight_config = weight_config

        self.taper_type = taper_type


class Gradient3d(object):

    def __init__(self, cmtsource: CMTSource, data_container: DataContainer,
                 config: Gradient3dConfig):
        """Gradient method that takes in windows observed and synthetic data and
        then uses a the analytical derivates with respect to moment scaling and
        as well as timeshift to compute optimal timeshift in the cmt solution
        and the corresponding
        """

        self.cmtsource = cmtsource
        self.data_container = data_container
        self.config = config

        self.metas = []
        self.new_cmtsource = None

        # timeshift parameters
        self.t00_best = None
        self.t00_mean = None
        self.t00_std = None
        self.t00_var = None

        # Mmoment
        self.m00_best = None
        self.m00_mean = None
        self.m00_std = None
        self.m00_var = None

        # Misfit reduction
        self.chi_list = None

        # misfit reduction
        self.stats = dict()
        self.var_all = None
        self.var_all_new = None
        self.var_reduction = None

        # internal vairables to be set
        self.synt = None
        self.ssynt = None  # shifted synthetics
        self.obsd = None
        self.delta = None
        self.npts = None
        self.nwin = None
        self.tapers = None
        self.windows = None

        # Space holders
        self.bootstrap_mean = np.array([np.nan])
        self.bootstrap_std = np.array([np.nan])
        self.maxcost_array = np.array([np.nan])
        self.mincost_array = np.array([np.nan])
        self.stdcost_array = np.array([np.nan])
        self.meancost_array = np.array([np.nan])

        # Prepare Matrices (otherwise it's going to be hard
        # to compute the residual)
        self.setup_window_weight()
        self.prepare_inversion_data()

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

        setup_energy_weight(self.metas, self.data_container)

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
        """Performs the gradient descent method."""

        # Create Gradient method class
        G = Gradient(self.obsd, self.synt, self.tapers, self.delta,
                     method=self.config.method,
                     ia=self.config.ia, idt=self.config.idt,
                     nt=self.config.nt, nls=self.config.nls,
                     crit=self.config.crit,
                     precond=self.config.precond,
                     reg=self.config.reg)
        G.gradient()

        # Extract values
        self.t00_best = G.dt
        self.m00_best = G.a
        self.chi_list = G.cost_list[:G.it]

        # Bootstrap if wanted
        if self.config.bootstrap:
            self.bootstrap()

        # Post processing
        self.prepare_new_cmtsource()
        self.prepare_new_synthetic()
        self.calculate_variance()

    def bootstrap(self):
        """Performs bootstrap analysis on repeated gradient computations.
        """

        self.ntraces = self.synt.shape[0]
        self.n_subset = int(np.ceil(self.config.bootstrap_subset_ratio
                                    * self.ntraces))
        logger.info("Bootstrap Repeat: %d  Subset Ratio: %f Nsub: %d"
                    % (self.config.bootstrap_repeat,
                       self.config.bootstrap_subset_ratio,
                       self.n_subset))

        # Preallocate variables
        bootstrap_t = np.zeros(self.config.bootstrap_repeat)
        bootstrap_m = np.zeros(self.config.bootstrap_repeat)
        bootstrap_cost_lists = []
        maxlen = 0

        timer0 = time.time()

        if not self.config.parallel:
            self.num_cores = 1
        else:
            self.num_cores = get_number_of_cores(self.config.bootstrap_repeat)

        logger.info("Number of cores: %d" % self.num_cores)
        logger.info("Bootstrap - Looping ....")

        np.random.seed(1234)
        random_arrays = []
        for _i in range(self.config.bootstrap_repeat):
            # Get random array
            random_array = np.random.choice(self.ntraces, self.n_subset,
                                            replace=True)
            random_arrays.append(random_array)

        if self.num_cores == 1:
            np.random.seed(1234)
            for _i in range(self.config.bootstrap_repeat):
                # Compute bootstrap stuff
                bt, bm, bcost, costlen = self.bootstrap_wrapper(
                    _i, self.config, self.obsd[random_arrays[_i], :],
                    self.synt[random_arrays[_i], :],
                    self.tapers[random_arrays[_i], :], self.delta)

                # Put everything into arrays
                bootstrap_t[_i] = bt
                bootstrap_m[_i] = bm
                bootstrap_cost_lists.append(bcost[:costlen])
                maxlen = np.max(np.array([maxlen, costlen]))

            # Compute stats
            self.bootstrap_mean = np.array([np.mean(bootstrap_m),
                                            np.mean(bootstrap_t)])
            self.bootstrap_std = np.array([np.std(bootstrap_m),
                                           np.std(bootstrap_t)])

            # Fix bootstrap cost list.
            self.cost_array = np.zeros((self.config.bootstrap_repeat, maxlen))
            for _i, clist in enumerate(bootstrap_cost_lists):
                self.cost_array[_i, :len(clist)] = np.array(clist)
                self.cost_array[_i, len(clist):] = clist[-1]

        elif self.num_cores > 1 and self.config.mpi_env:

            # Sample result dictionary to measure size

            jobs = split(list(range(self.config.bootstrap_repeat)),
                         self.num_cores)

            job_len = len(jobs[0])

            sample_d = {"dt": np.zeros(job_len, dtype=np.float),
                        "a": np.zeros(job_len, dtype=np.float),
                        "cost": np.zeros((job_len, self.config.nt),
                                         dtype=np.float),
                        "cost_len": 9999 * np.ones(job_len, dtype=np.int)}

            # Dictionary to be broadcast to the workers
            bcast_dict = {"obsd": self.obsd,
                          "synt": self.synt,
                          "tapers": self.tapers,
                          "delta": self.delta,
                          "repeat": self.config.bootstrap_repeat,
                          "n_subset": self.n_subset,
                          "ntraces": self.ntraces,
                          "config": self.config,
                          "randarray": random_arrays,
                          "jobs": jobs,
                          "job_len": job_len,
                          "sample_d": sample_d}

            try:
                from mpi4py import MPI
            except Exception as e:
                print(e)
                ValueError("mpi4py not installed?")

            arg_list = ["-m", "pycmt3d.grad_child_mpi"]

            comm = MPI.COMM_SELF.Spawn(sys.executable,
                                       args=arg_list,
                                       maxprocs=self.num_cores)
            # Broad cast data dictionary
            broadcast_dict(bcast_dict, comm)

            # Get back results from the workers
            list_of_result_dicts = \
                get_result_dictionaries(sample_d, comm, self.num_cores)

            comm.Disconnect()

            # sys.exit()
            bootstrap_cost_len = []
            counter = 0
            import pprint
            pprint.pprint(list_of_result_dicts)
            for _i, result in enumerate(list_of_result_dicts):
                for _j, clen in enumerate(result["cost_len"]):
                    if clen != 9999:
                        bootstrap_t[counter] = result["dt"][_j]
                        bootstrap_m[counter] = result["a"][_j]
                        bootstrap_cost_lists.append(
                            result["cost"][_j].tolist())
                        bootstrap_cost_len.append(clen)
                        maxlen = np.max(np.array([maxlen, clen]))
                        counter += 1
            # Compute stats
            self.bootstrap_mean = np.array([np.mean(bootstrap_m),
                                            np.mean(bootstrap_t)])
            self.bootstrap_std = np.array([np.std(bootstrap_m),
                                           np.std(bootstrap_t)])

            print(bootstrap_cost_lists)
            self.cost_array = np.array(bootstrap_cost_lists)[:, :maxlen]
            for _i, (row, clen) in enumerate(zip(self.cost_array,
                                                 bootstrap_cost_len)):
                self.cost_array[_i, clen:] = row[clen - 1]

        else:

            results = Parallel(n_jobs=self.num_cores)(
                delayed(self.bootstrap_wrapper)(
                    k, config=self.config,
                    obsd=self.obsd[random_arrays[k], :],
                    synt=self.synt[random_arrays[k], :],
                    tapers=self.tapers[random_arrays[k], :],
                    delta=self.delta)
                for k in range(self.config.bootstrap_repeat))

            bootstrap_cost_len = []
            for _i, result in enumerate(results):
                bootstrap_t[_i] = result[0]
                bootstrap_m[_i] = result[1]
                bootstrap_cost_lists.append(result[2])
                bootstrap_cost_len.append(result[3])
                maxlen = np.max(np.array([maxlen, result[3]]))

            # Compute stats
            self.bootstrap_mean = np.array([np.mean(bootstrap_m),
                                            np.mean(bootstrap_t)])
            self.bootstrap_std = np.array([np.std(bootstrap_m),
                                           np.std(bootstrap_t)])

            # Fix bootstrap cost list.
            self.cost_array = np.zeros((self.config.bootstrap_repeat, maxlen))
            for _i, clist in enumerate(bootstrap_cost_lists):
                self.cost_array[_i, :] = clist[:maxlen]
                self.cost_array[_i, bootstrap_cost_len[_i]:] = \
                    clist[bootstrap_cost_len[_i] - 1]

            # # Compute stats
            # self.bootstrap_mean = np.array([np.mean(bootstrap_m),
            #                                 np.mean(bootstrap_t)])
            # self.bootstrap_std = np.array([np.std(bootstrap_m),
            #                                np.std(bootstrap_t)])
            #
            # # Fix bootstrap cost list.
            # self.cost_array = np.zeros(
            #     (self.config.bootstrap_repeat, maxlen))
            # for _i, clist in enumerate(bootstrap_cost_lists):
            #     self.cost_array[_i, :len(clist)] = np.array(clist)
            #     self.cost_array[_i, len(clist):] = clist[-1]

        self.maxcost_array = np.max(self.cost_array, axis=0)
        self.mincost_array = np.min(self.cost_array, axis=0)
        self.stdcost_array = np.std(self.cost_array, axis=0)
        self.meancost_array = np.mean(self.cost_array, axis=0)

        logger.info("Total Bootstrap time: %.2f" % (time.time() - timer0))

    @staticmethod
    def bootstrap_wrapper(k, config, obsd, synt, tapers, delta):
        """This function simply wraps around the gradient subset method
        to efficiently compute bootstrap subsets"""

        logger.info("Bootstrap Repeat: %d/%d start..."
                    % (k, config.bootstrap_repeat))

        G = Gradient(obsd,
                     synt,
                     tapers,
                     delta,
                     method=config.method,
                     ia=config.ia, idt=config.idt,
                     nt=config.nt, nls=config.nls,
                     crit=config.crit,
                     precond=config.precond,
                     reg=config.reg, damping=config.damping)
        G.gradient()

        bootstrap_t = G.dt
        bootstrap_m = G.a
        bootstrap_cost_list = G.cost_list

        cost_list_len = len(G.cost_list[:G.it])

        logger.info("Bootstrap Repeat: %d/%d done."
                    % (k, config.bootstrap_repeat))

        return bootstrap_t, bootstrap_m, bootstrap_cost_list, cost_list_len

    def prepare_inversion_data(self):
        """Computes amplitude and cross correlation misfit for moment scaling
        as well as timeshift.
        """

        npts_list = []
        for trwin in self.data_container.trwins:
            npts_list.append(trwin.datalist['obsd'].stats.npts)

        self.npts = np.max(npts_list)
        self.delta = self.data_container[0].datalist['obsd'].stats.delta
        self.nwin = len(self.data_container.trwins)

        # Parameters needed for inversion
        self.obsd = np.zeros((self.nwin, self.npts))
        self.synt = np.zeros((self.nwin, self.npts))
        self.tapers = np.zeros((self.nwin, self.npts))

        # Prepare weights
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

        counter = 0

        for _k, trwin in enumerate(self.data_container):
            self.obsd[_k, :] = pad(trwin.datalist["obsd"].data, self.npts)
            self.synt[_k, :] = pad(trwin.datalist["synt"].data, self.npts)

            for _win_idx in range(trwin.windows.shape[0]):
                istart, iend = get_window_idx(trwin.windows[_win_idx],
                                              self.delta)

                self.tapers[_k, istart:iend] = \
                    construct_taper(iend - istart, taper_type='tukey') \
                    * weights[counter]
                counter += 1

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

    def prepare_new_cmtsource(self):
        """Takes the values for the timeshift and the scaling factor and
        creates a new set of synthetics."""

        newcmt = deepcopy(self.cmtsource)

        logger.info("Preparing new cmtsource...")

        newcmt.cmt_time += self.t00_best
        logger.info("\tAdding time shift to cmt origin time:"
                    "%s + %fsec= %s"
                    % (self.cmtsource.cmt_time, self.t00_best,
                       newcmt.cmt_time))

        attrs = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
        for attr in attrs:
            newval = self.m00_best * getattr(newcmt, attr)
            setattr(newcmt, attr, newval)
        logger.info("\tMultiply scalar moment by %f%%"
                    % (self.m00_best * 100))

        self.new_cmtsource = newcmt

    def prepare_new_synthetic(self):
        logger.info("Reconstruct new synthetic seismograms...")

        for trwin in self.data_container:
            new_synt = trwin.datalist["synt"].copy()

            # Fix traces
            timeshift_trace_pad(new_synt, self.t00_best)
            new_synt.data *= self.m00_best
            trwin.datalist["new_synt"] = new_synt

    def plot_stats_histogram(self, outputdir=".", figure_format="pdf"):
        """
        Plot inversion histogram, including histograms of tshift, cc,
        power_l1, power_l2, cc_amp, chi values before and after the
        inversion.
        :param outputdir:
        :return:
        """
        constr_str = "grad"
        if not self.config.weight_config.normalize_by_energy:
            prefix = "%s.%s" % (constr_str, "no_normener")
        else:
            prefix = "%s.%s" % (constr_str, "normener")

        if not self.config.weight_config.normalize_by_category:
            prefix += ".no_normcat"
        else:
            prefix += ".normcat"
        figname = "%s.%s.stats.%s" % (self.cmtsource.eventname, prefix,
                                      figure_format)
        figname = os.path.join(outputdir, figname)

        plots = plot_util.PlotStats(self.data_container, self.metas, figname)
        plots.plot_stats_histogram()

    def plot_new_synt_seismograms(self, outputdir, figure_format="pdf"):
        """
        Plot the new synthetic waveform
        """
        plot_util.plot_seismograms(self.data_container, outputdir,
                                   self.cmtsource, figure_format=figure_format)

    def write_new_syn(self, outputdir=".", file_format="sac"):
        """
        Write out the new synthetic waveform
        """
        file_format = file_format.lower()
        logger.info("New synt output dir: %s" % outputdir)
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        if 'new_synt' not in self.data_container.trwins[0].datalist.keys():
            raise ValueError("new synt not computed yet")

        eventname = self.cmtsource.eventname
        suffix = "grad"

        if file_format == "sac":
            self.data_container.write_new_synt_sac(outputdir=outputdir,
                                                   suffix=suffix)
        elif file_format == "asdf":
            file_prefix = \
                os.path.join(outputdir, "%s.%s" % (eventname, suffix))
            self.data_container.write_new_synt_asdf(file_prefix=file_prefix)
        else:
            raise NotImplementedError("file_format(%s) not recognised!"
                                      % file_format)

    def write_new_cmtfile(self, outputdir="."):
        """
        Write new_cmtsource into a file
        """
        suffix = "grad"
        outputfn = "%s.%s.inv" % (self.cmtsource.eventname, suffix)
        cmtfile = os.path.join(outputdir, outputfn)
        logger.info("New cmt file: %s" % cmtfile)

    def extract_metadata(self, cat_name, meta_varname):
        data_old = []
        data_new = []
        cat_data = self.metas_sort[cat_name]
        for meta in cat_data:
            data_old.extend(meta.prov["synt"][meta_varname])
            data_new.extend(meta.prov["new_synt"][meta_varname])

        return data_old, data_new

    def sort_metas(self):
        """sort metas into different categories for future plotting """
        metas_sort = defaultdict(list)
        key_map = defaultdict(set)

        # Get the computed meta information.
        metas = self.metas

        for trwin, meta in zip(self.data_container, metas):
            comp = trwin.channel
            cat_name = get_trwin_tag(trwin)
            key_map[comp].add(cat_name)
            metas_sort[cat_name].append(meta)

        self.metas_sort = metas_sort
        self.key_map = key_map

    def extract_stats(self):
        """This function uses the info save in the metas to compute
        the component means, stds, mins, and maxs."""

        # Define list of statistics to get.
        vtype_list = ["tshift", "cc", "power_l1", "power_l2", "cc_amp", "chi"]

        # Sort the metadata into categories (040_100.obsd.BHZ is a category eg)
        self.sort_metas()

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
            cat_dict[varname] = {"before": data_before,
                                 "after": data_after}

        return cat_dict

    def write_summary_json(self, outputdir=".", mode="global"):
        """This function uses all computed statistics and outputs a json
        file. Content will include the statistics table.
        cost reduction in. """

        eventname = self.cmtsource.eventname
        outputfn = "%s.grad.stats.json" % (eventname)
        outputfn = os.path.join(outputdir, outputfn)
        filename = outputfn

        logger.info("Grid search summary file: %s" % filename)

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

        # Compute Stats
        self.extract_stats()
        outdict["stats"] = self.stats

        outdict["G"] = {
            "tshift": self.t00_best,
            "ascale": self.m00_best,
            "bootstrap_mean": self.bootstrap_mean.tolist(),
            "bootstrap_std": self.bootstrap_std.tolist(),
            "chi_list": self.chi_list,
            "meancost_array": self.meancost_array.tolist(),
            "stdcost_array": self.stdcost_array.tolist(),
            "maxcost_array": self.maxcost_array.tolist(),
            "mincost_array": self.mincost_array.tolist()
        }

        outdict["var_reduction"] = self.var_reduction
        outdict["config"] = {
            "c1": self.config.c1,
            "c2": self.config.c2,
            "idt": self.config.idt,
            "ia": self.config.ia,
            "nt": self.config.nt,
            "nls":  self.config.nls,
            "crit":  self.config.crit,
            "reg":  self.config.reg,
            "method": self.config.method,
            "precond": self.config.precond,
            "damping":  self.config.damping,
            "taper_type": self.config.taper_type,
            "bootstrap":  self.config.bootstrap,
            "bootstrap_repeat":  self.config.bootstrap_repeat,
            "bootstrap_subset_ratio":  self.config.bootstrap_subset_ratio,
            "weight_config":
                {"normalize_by_energy":
                 self.config.weight_config.normalize_by_energy,
                 "normalize_by_category":
                 self.config.weight_config.normalize_by_category}
        }
        outdict["mode"] = mode

        dump_json(outdict, filename)


class Gradient(object):

    def __init__(self, obsd: np.ndarray, synt: np.ndarray,
                 tapers: np.ndarray, delta: np.float, method: str = "gn",
                 ia: float = 1.0, idt: float = 0.0, nt=20, nls=20,
                 c1: float = 1e-4, c2: float = 0.9,
                 damping: float or None = None,
                 crit: float = 1e-3, precond: bool = False,
                 reg: bool = False):

        self.obsd = obsd
        self.synt = synt
        self.ssynt = synt
        self.shifted = synt
        self.tapers = tapers
        self.delta = delta

        # Method of choice Gauss-Newton or Newton
        if method in ["gn", "n"]:
            self.method = method
        else:
            raise ValueError("Chosen method not supported.")

        # Method parameters
        self.c1 = c1
        self.c2 = c2
        self.it = 1
        self.nt = nt
        self.nls = nls
        self.damping = damping
        self.precond = precond
        self.reg = reg

        self.dt = idt
        self.a = ia
        self.m = np.array([ia, idt])

        # Placeholders
        self.dt_list = [self.dt]
        self.a_list = [self.a]

        # Compute residual and cost function
        self.m = np.array([self.a, self.dt])
        self.m_list = [self.m]
        self.res = self.compute_residual()
        self.chi0 = self.compute_misfit()
        self.chi = self.chi0 * 1.
        self.chip = self.chi0 * 1.
        self.cost_list = [0] * self.nt
        self.cost_list[0] = 1
        self.crit = crit

    def gradient(self):

        if self.precond:
            logger.info("Preconditioner not set...")

        while self.chi / self.chi0 > self.crit and self.it < self.nt:

            logger.debug("Iter: %3d -- C=%4f -- dt=%4f -- A=%4f"
                         % (self.it, self.chi / self.chi0,
                            self.dt, self.a))

            # Compute Analytical hessian using Newton's method
            self.B = self.compute_hessian()
            self.g = self.compute_gradient()

            if self.damping is not None:
                logger.debug("Damping:")
                logger.debug("Cond. number prior to damping: %.2f"
                             % np.linalg.cond(self.B))
                self.B = damping(self.B, damping=self.damping)
                logger.debug("Cond. number post damping: %.2f"
                             % np.linalg.cond(self.B))
                self.g += self.damping * self.m

            # Preconditioning To make Hessian stabile
            if self.precond:
                self.H, self.g = precondition(self.B, self.g)
            else:
                # inverting the matrix
                self.H = np.linalg.inv(self.B)

            if (-(self.H @ self.g) @ self.g > 0):
                logger.info("Gradient f**ked")
                logger.info("B: " + self.arraystr(self.B))
                logger.info("g: " + self.arraystr(self.g))

            # Compute new step
            self.dm = self.H @ -self.g

            # Compute better alpha
            self.line_search()

            self.m_list.append(self.m)
            self.forward()

            self.res = self.compute_residual()
            self.chi = self.compute_misfit()

            self.dt = self.m[1]
            self.a = self.m[0]
            self.dt_list.append(self.dt)
            self.a_list.append(self.a)

            if np.abs((self.chi - self.chip) / self.chip) < 1e-3:
                logger.debug("No improvement!")
                break

            self.chip = self.chi
            self.cost_list[self.it] = self.chi / self.chi0
            self.it += 1

    def line_search(self):
        """Performs the linesearch for optimal alpha using
        wolfe conditions.
        """

        # Linesearch
        ar = 0
        al = 0
        alpha = 1
        q = self.g @ self.dm

        for _i in range(self.nls):

            # print linesearch iteration info
            logger.debug("LS_Iter: %d -- alpha=%4f" % (_i, alpha))

            # Get new measurements
            mnew = self.m + alpha * self.dm

            self.forward(mnew)
            self.res = self.compute_residual()
            gnew = self.compute_gradient()
            fcost_new = self.compute_misfit()

            # Compute q
            qnew = gnew @ self.dm

            # Check wolfe condition
            wolfe = check_wolfe(alpha, self.chi, fcost_new, q, qnew,
                                c1=self.c1, c2=self.c2, strong=False)

            # Update alpha using wolfe conditions
            good, alpha, al, ar = update_alpha(alpha, al, ar, wolfe)

            if good:
                self.chi = fcost_new
                self.m = mnew
                break

        if not good:
            self.chi = fcost_new

    def forward(self, m=None):
        """Computes the forward data using the most recent model
        vector.
        """
        if m is None:
            self.shifted = timeshift_mat(self.synt, self.m[1], self.delta)
            self.ssynt = self.m[0] * self.shifted
        else:
            self.shifted = timeshift_mat(self.synt, m[1], self.delta)
            self.ssynt = m[0] * self.shifted

    def compute_hessian(self):
        """Computes Hessian depending on the method chosen.
        """

        if self.method == "n":
            return self.compute_B()
        else:
            return self.compute_JJ()

    def compute_gradient(self):
        """Computes Hessian depending on the method chosen.
        """

        if self.method == "gn":
            return self.compute_b()
        else:
            return self.compute_g()

    def compute_b(self):
        """Computes the Jr = b RHS of the Gauss Newton method.
        """

        # Derivatives of the data with respect to model parameters
        dsda = - self.ssynt
        dsddt = self.a * np.gradient(self.ssynt, self.delta, axis=-1)

        return np.array([np.sum(self.res * dsda * self.tapers),
                         np.sum(self.res * dsddt * self.tapers)])

    def compute_JJ(self):
        """Gauss Newton approach with JtJ delta m.""""""Computes the Gauss Newton approximate Hessian.
        """

        dsda = - self.shifted
        dsddt = self.a * np.gradient(self.shifted, self.delta, axis=-1)

        J11 = np.sum(dsda ** 2 * self.tapers)
        J22 = np.sum(dsddt ** 2 * self.tapers)
        J21 = np.sum(dsda * dsddt * self.tapers)

        return np.array([[J11, J21], [J21, J22]])

    def compute_g(self):
        """Computing the analytical gradient with respect to the model parameters
        a and t0.
        """

        dsdt = np.gradient(self.shifted, self.delta, axis=-1)

        dCdt = np.sum(self.res * self.a * dsdt * self.tapers)

        dCda = - np.sum(self.res * self.shifted * self.tapers)

        return np.array([dCda, dCdt]).T

    def compute_B(self):
        """Computes analytical Hessian.

        Returns:
            nd.array: 2x2
        """

        dsdt = np.gradient(self.shifted, self.delta, axis=-1)

        d2sdt2 = np.gradient(dsdt, self.delta, axis=-1)

        d2Cda2 = np.sum(self.shifted ** 2 * self.tapers)

        d2Cdt2 = np.sum(((dsdt * self.a) ** 2
                         - d2sdt2 * self.res * self.a)
                        * self.tapers)

        d2Cdadt = np.sum((dsdt * (self.res - self.a * self.shifted)
                         * self.tapers))

        return np.array([[d2Cda2, d2Cdadt],
                         [d2Cdadt, d2Cdt2]])

    def compute_misfit(self):
        """Takes in a set of data (needs to be same as original obsd data
        and computes the misfit between the input and observed data.
        """
        return 0.5 * np.sum(self.tapers * (self.obsd - self.ssynt) ** 2,
                            axis=None)

    def compute_residual(self):
        """Takes in a set of data (needs to be same as original obsd data
        and computes the misfit between the input and observed data.
        """

        return (self.obsd - self.ssynt)

    @staticmethod
    def arraystr(array):
        """ Creates one line string of comma seperated array elements"""
        if type(array) == list:
            array = np.array(array)
        elif type(array) == np.ndarray:
            pass
        else:
            raise ValueError("Array type not supported")
        string = ", ".join(["%f" % el for el in array.flatten()])

        return string
