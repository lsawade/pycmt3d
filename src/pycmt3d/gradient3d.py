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
import sys
import numpy as np
from copy import deepcopy
import time
import psutil
from joblib import delayed
from joblib import Parallel
import matplotlib.pyplot as plt
# Internal imports
from .source import CMTSource
from .data_container import DataContainer
from .data_container import MetaInfo
from . import logger
from .weight import Weight, setup_energy_weight
from .measure import calculate_variance_on_trace
from .util import timeshift_mat
from .util import get_window_idx
from .util import construct_taper
from .mpi_utils import broadcast_dict
from .mpi_utils import get_result_dictionaries
from .mpi_utils import split

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
    # PB = Pinv @ B
    # Pg = Pinv @ g
    print("B", B)
    print("H", H)
    print("g", g)
    print("Hg", - H @ g)
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
                 use_new=False,
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
            use_new (bool, optional): Choice of whether to use synthetics from
                                      a cmt3d inversion. Defaults to False.
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

        # If the data_container contains new_synthetic data, the new
        # synthetic data can be used for the gridsearch.
        self.use_new = use_new


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
            for _i, result in enumerate(list_of_result_dicts):
                for _j, clen in enumerate(result["cost_len"]):
                    if clen != 9999:
                        bootstrap_t[counter] = result["dt"][_j]
                        bootstrap_m[counter] = result["a"][_j]
                        bootstrap_cost_lists.append(result["cost"][_j])
                        bootstrap_cost_len.append(clen)
                        maxlen = np.max(np.array([maxlen, clen]))
                        counter += 1

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

        self.npts = self.data_container[0].datalist['obsd'].stats.npts
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
                print(meta.weights)
                weights.extend(meta.weights)
            weights = np.array(weights)
        else:
            if self.data_container.nwindows:
                weights = np.ones(self.data_container.nwindows)
            else:
                weights = np.ones(len(self.data_container.nwindows))

        counter = 0

        for _k, trwin in enumerate(self.data_container):
            self.obsd[_k, :] = trwin.datalist["obsd"].data

            if self.config.use_new:
                if "new_synt" not in trwin.datalist:
                    raise ValueError("new synt is not in trwin(%s) "
                                     "datalist: %s"
                                     % (trwin, trwin.datalist.keys()))
                else:
                    self.synt[_k, :] = trwin.datalist["new_synt"].copy()
            else:
                self.synt[_k, :] = trwin.datalist["synt"].data

            for _win_idx in range(trwin.windows.shape[0]):
                istart, iend = get_window_idx(trwin.windows[_win_idx],
                                              self.delta)

                self.tapers[_k, istart:iend] = \
                    construct_taper(iend - istart, taper_type='tukey') \
                    * weights[counter]
                counter += 1

        fig = plt.figure(figsize=(1.25*10, 10))
        ax = plt.gca()
        matmat = ax.matshow(self.tapers)
        plt.colorbar(matmat)
        ax.set_aspect(self.npts/self.nwin)
        # plt.savefig("/Users/lucassawade/test.pdf")
        plt.savefig("/scratch/gpfs/lsawade/test.pdf")
        plt.close(fig)

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
        logger.info("\tMultiply scalar moment change by %f%%"
                    % (self.m00_best * 100))

        self.new_cmtsource = newcmt

    def prepare_new_synthetic(self):
        logger.info("Reconstruct new synthetic seismograms...")

        for trwin in self.data_container:

            # Get trace
            if self.config.use_new:
                if "new_synt" not in trwin.datalist:
                    raise ValueError("new synt is not in trwin(%s) "
                                     "datalist: %s"
                                     % (trwin, trwin.datalist.keys()))
                else:
                    new_synt = trwin.datalist["new_synt"].copy()
            else:
                new_synt = trwin.datalist["synt"].copy()

            # Fix traces
            new_synt.stats.starttime += self.t00_best
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
            meta.prov["new_synt"]["tshift"] -= self.t00_best


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
        self.res = self.compute_residual()
        self.chi0 = self.compute_misfit()
        self.chi = self.chi0 * 1.
        self.chip = self.chi0 * 1.
        self.cost_list = [0] * self.nt
        self.cost_list[0] = 1
        self.crit = crit
        self.m = np.array([self.a, self.dt])
        self.m_list = [self.m]

    def gradient(self):

        if self.precond:
            logger.info("Preconditioner not set...")

        while self.chi / self.chi0 > self.crit and self.it <= self.nt:

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
            self.ssynt = self.m[0] * timeshift_mat(self.synt, self.m[1],
                                                   self.delta)
        else:
            self.ssynt = m[0] * timeshift_mat(self.synt, m[1], self.delta)

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

        dsda = - self.ssynt
        dsddt = self.a * np.gradient(self.ssynt, self.delta, axis=-1)

        J11 = np.sum(dsda ** 2 * self.tapers)
        J22 = np.sum(dsddt ** 2 * self.tapers)
        J21 = np.sum(dsda * dsddt * self.tapers)

        return np.array([[J11, J21], [J21, J22]])

    def compute_g(self):
        """Computing the analytical gradient with respect to the model parameters
        a and t0.
        """
        dsdt = np.gradient(self.ssynt, self.delta, axis=-1)

        # d2sdt2 = np.gradient(dsdt, self.delta, axis=-1)

        dCdt = np.sum(self.res * self.delta * self.a * dsdt)

        dCda = - np.sum(self.res * self.delta * self.ssynt)

        return np.array([dCda, dCdt]).T

    def compute_B(self):
        """Computes analytical Hessian.

        Returns:
            nd.array: 2x2
        """

        dsdt = np.gradient(self.ssynt, self.delta, axis=-1)

        d2sdt2 = np.gradient(dsdt, self.delta, axis=-1)

        d2Cda2 = np.sum(self.ssynt ** 2 * self.delta * self.tapers)

        d2Cdt2 = np.sum(((dsdt * self.a) ** 2
                         - d2sdt2 * self.res * self.a) * self.delta
                        * self.tapers)

        d2Cdadt = np.sum((dsdt * (self.res - self.a * self.ssynt))
                         * self.delta * self.tapers)

        return np.array([[d2Cda2, d2Cdadt],
                         [d2Cdadt, d2Cdt2]])

    def compute_misfit(self):
        """Takes in a set of data (needs to be same as original obsd data
        and computes the misfit between the input and observed data.
        """
        return np.sum(self.tapers * (self.obsd - self.ssynt) ** 2 * self.delta,
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
