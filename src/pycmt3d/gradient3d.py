#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for Gauss Newton and Hessian optimization of
the scalar moment and timeshift.

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
import matplotlib.pyplot as plt
from copy import deepcopy

import multiprocessing
import psutil
from joblib import delayed
from joblib import Parallel

# Internal imports
from .source import CMTSource
from .data_container import DataContainer
from .data_container import MetaInfo
from . import logger
from .weight import Weight
from .measure import calculate_variance_on_trace
from .measure import calculate_waveform_misfit_on_trace
from .measure import construct_matrices, compute_misfit, get_window_list
# from .plot_util import PlotStats
from .util import timeshift_trace
from .util import random_select
import time



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
    if strong == False:
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
    if w3 is False: # not a descent direction... quit
        raise ValueError('Not a descent direction... STOP')

    if w1 is True and w2 is True:  # both are satisfied then terminate
        good = True

    elif w1 is False:  # not a sufficient decrease, we've been too far
        ar = alpha
        alpha = (al + ar)*0.5

    elif w1 is True and w2 is False:
        al = alpha
        if ar > 0:  # sufficient decrease but too close already backeted
                    # decrease in interval
            alpha = (al+ ar)*0.5
        else:  # sufficient decrease but too close, then increase a
            alpha = factor * alpha

    return good, alpha, al, ar


def precondition(B, g):

    # Preconditioning
    fac = 1/np.array([B[0, 0], B[1, 1]])
    ng = np.sum(g**2)
    H = np.diag(fac)
    # PB = Pinv @ B
    # Pg = Pinv @ g
    print("B", B)
    print("H", H)
    print("g", g)
    print("Hg", - H @ g )
    return H, g

def reguralization(B):

    lb = np.median(B)
    lbm = np.eye(2) * lb

    return B + lbm

class Gradient3dConfig(object):
    def __init__(self, method="gn", weight_data=False, weight_config=None, use_new=False,
                 taper_type="tukey",
                 c1=1e-4, c2=0.9, 
                 idt: float = 0.0, ia = 1.,
                 nt: int = 50, nls: int = 20,
                 crit: float = 0.01,
                 precond: bool = False, reg: bool = False,
                 bootstrap=True, bootstrap_repeat=20,
                 bootstrap_subset_ratio=0.4):
        """Configuration for the gradient method.
        
        Args:
            method (str): descent algorithm. Defaults to "gn".
            weight_data (bool,optional): contains the weightdata.
                                         Defaults to False.
            weight_config ([type], optional): contains the weightdata
                                              configuration. Defaults to None.
            use_new (bool, optional): Choice of whether to use synthetics from a
                                      cmt3d inversion. Defaults to False.
            taper_type (str, optional): Taper for window selection. Defaults to "tukey".
            c1 (float, optional): Wolfe condition parameter. Defaults to 1e-4
            c2 (float, optional): Wolfe condition parameter. Defaults to 0.9
            idt (float, optional): Starting value for the timeshift.
                                   Defaults to 0.0
            ia (float, optional): Starting value for the moment scaling 
                                  factor. Defaults to 1.0
            nt (int, optional): Maximum number of iterations. Defaults to 50.
            nls (int, optional): Maximum number of linesearch iterations.
                                 Defaults to 20.
            crit (float, optional): critical misfit reduction. Defaults to 1e-3.
            precond (bool, optional): If True uses preconditioner on
                                      the Hessian. Defaults to False.
            reg (bool, optional): If True reguralizes the Hessian. Defaults to False.                                      
            bootstrap (bool, optional): Whether to perform a bootstrap 
                                        statistic. Defaults to True.
            bootstrap_parallel (int, optional): whether to perform the bootstrap
                                                statistic in parallel. 0 = serial,
                                                1 = parallel and let the program figure out
                                                the number of cores. all other integers
                                                define the number of cores.
            bootstrap_repeat (int, optional): number of repeats to be done. Defaults to 20.
            bootstrap_subset_ratio (float, optional): taking a certain ratio of the available
                                                      data to compute a bootstrap statistic. 
                                                      Defaults to 0.4.
        
        Raises:
            ValueError: if wrong method string is input.
        """


        # Method of choice
        if method in ["gn", "n"]:
            self.method = method
        else:
            raise ValueError("Chosen method not supported.")
        
        # Wolfe paramters
        self.c1 = c1
        self.c2 = c2

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
        self.synth = None
        self.ssynth = None # shifted synthetics
        self.obsd = None
        self.delta = None
        self.npts = None
        self.nwin = None
        self.tapers = None
        self.windows = None

        # Prepare Matrices (otherwise it's going to be hard 
        # to compute the residual)
        self.prepare_inversion_data()


    def search(self):
        """Performs the gradient descent method."""

        # Create Gradient method class
        G = Gradient(self.obsd, self.synth, self.tapers, method=self.config.method,
                     ia=self.config.ia, dt=self.config.idt,
                     nt=self.config.nt, nls=self.config.nls, crit=self.config.crit,
                     precond= self.config.precond,
                     reg=self.config.reg)
        G.gradient()

        # Extract values
        self.t00_best = G.dt
        self.m00_best = G.a
        self.chi_list = G.cost_list

        self.prepare_new_cmtsource()
        self.prepare_new_synthetic()
        

    def prepare_inversion_data(self):
        """Computes amplitude and cross correlation misfit for moment scaling
        as well as timeshift.
        """

        self.npts = data_container[0].datalist['obsd'].stats.npts
        self.delta = data_container[0].datalist['obsd'].stats.delta
        self.nwin = len(data_container.trwins)

        # Parameters needed for inversion
        self.obsd = np.zeros((nwin, npts))
        self.synt = np.zeros((nwin, npts))
        self.tapers = np.zeros((nwin, npts))

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
                                            delta)

                self.tapers[_k, istart:iend] = \
                    construct_taper(iend - istart, taper_type='hann') \
                    * weights[counter]
                counter += 1

        # Create first instance of shifted traces
        self.ssynth = copy.deepcopy(self.synt)

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
            if self.config.origin_time_inv:
                meta.prov["new_synt"]["tshift"] -= self.t00_best

    def plot_cost(self, figurename=None):

        if figurename is None:
            ax = plt.gca()

        else:
            plt.figure()
            ax = plt.axes()

        ax.plot(self.c_list, "r",
                label="%s ($\mathcal{C}_{min} = %.3f$)" % (self.config.method, self.c_list[-1]))
        plt.legend(prop={'size': 6}, fancybox=False, framealpha=1)
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Misfit reduction")

        if figurename is None:
            pass
        elif figurename == 'show':
            plt.show()
        else:
            plt.savefig(figurename)
    
    def bootstrap_search(self):
        pass



class Gradient(object):
    
    def __init__(self,  obsd: np.ndarray, synth: np.ndarray, delta: float,
                tapers: np.ndarray, method: str = "gn",
                ia: float = 1.0, idt: float = 0.0, nt=20, nls=20,
                crit: float = 1e-3, precond: bool = False,
                reg: bool = False):

        self.obsd = obsd
        self.synth = synth
        self.ssynth = synth
        self.delta = delta

        # Method of choice Gauss-Newton or Newton
        if method in ["gn", "n"]:
            self.method = method
        else:
            raise ValueError("Chosen method not supported.")
        
        self.dt = idt
        self.a = ia

        # Placeholders
        self.dt_list = [self.dt]
        self.a_list = [self.a]

        # Compute residual and cost function
        self.res = self.compute_residual()
        self.chi0 = self.compute_misfit()
        self.chi = self.chi0 * 1.
        self.chip = self.chi0 * 1.
        self.cost_list = [1]
        self.crit = crit
        self.m = np.array([self.a, self.dt])
        self.m_list = [self.m]

        self.it = 1
        self.nt = nt

        self.nls = nls
        self.precond = precond
        self.reg = reg

    def gradient(self):

        if self.precond:
            logger.info("Using preconditioner for matrix inversion...")

        while self.chi/self.chi0 > self.crit and self.it <= self.nt:

            logger.info("Iter: %3d -- C=%4f -- dt=%4f -- A=%4f"
                        % (self.it, self.chi/self.chi0, self.dt, self.a))

            # Compute Analytical hessian using Newton's method
            self.B = self.compute_hessian()
            self.g = self.compute_gradient()

            # Preconditioning To make Hessian stabile
            if self.precond:
                self.H, self.g = precondition(B, g)
            else:
                # inverting the matrix
                self.H = np.linalg.inv(B)

            if (-(H @ g) @ g > 0):
                logger.info("Gradient f**ked")
                logger.info("B: " + self.arraystr(B))
                logger.info("g: " + self.arraystr(g))

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
                logger.info("No improvement!")
                break

            self.chip = self.chi
            self.cost_list.append(self.chi/self.chi0)
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
            logger.info("LS_Iter: %d -- alpha=%4f"
                        % (_i, alpha))

            # Get new measurements
            mnew = self.m + alpha * self.dm

            self.a = mnew[0]
            self.dt = mnew[1]
            self.forward()
            self.ssynt *= mnew[0]
            self.res = self.compute_residual()
            g_new = self.compute_gradient()
            fcost_new = self.compute_misfit()

            # Compute q
            qnew = gnew @ self.dm

            # Check wolfe condition
            wolfe = check_wolfe(alpha, self.chi, fcost_new, q, qnew,
                                c1=self.config.c1, c2=self.config.c2, strong=False)

            # Update alpha using wolfe conditions
            good, alpha, al, ar = update_alpha(alpha, al, ar, wolfe)

            if good:
                self.chi = fcost_new
                self.m = mnew
                break

        if not good:
            self.chi = fcost_new

    def forward(self):
        """Computes the forward data using the most recent model
        vector.
        """
        self.ssynth = self.m[0] * timeshift_mat(self.synt, m[1], self.delta)

    def compute_hessian(self):
        """Computes Hessian depending on the method chosen.
        """

        if self.method == "gn":
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

        return np.array([np.sum(self.res * self.delta * dsda * self.tapers),
                      np.sum(self.res * self.delta * dsddt * self.tapers)])

    def compute_JJ(self):
        """Gauss Newton approach with JtJ delta m.""""""Computes the Gauss Newton approximate Hessian.
        """

        dsda = - self.ssynt
        dsddt = self.a * np.gradient(self.ssynt, self.delta, axis=-1)

        J11 = np.sum(dsda ** 2 * self.delta * self.tapers)
        J22 = np.sum(dsddt ** 2 * self.delta * self.tapers)
        J21 = np.sum(dsda * dsddt * self.delta * self.tapers)

        return np.array([[J11, J21], [J21, J22]])

    def compute_g(self):
        """Computing the analytical gradient with respect to the model parameters
        a and t0.
        """
        dsdt = np.gradient(self.ssynt, self.delta, axis=-1)

        d2sdt2 = np.gradient(dsdt, self.delta, axis=-1)

        dCdt = np.sum(self.res * self.delta * self.a * dsdt * self.tapers)

        dCda = - np.sum(self.res * self.delta * self.ssynt * self.tapers)

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
        
        Args:
            synth (numpy.ndarray): input data
        
        Returns:
            float
                                
        """
        return np.sum(self.tapers * (self.obsd - self.ssynth) ** 2 * self.delta,
                    axis=None)
    
    def compute_residual(self):
        """Takes in a set of data (needs to be same as original obsd data
        and computes the misfit between the input and observed data.
        
        Args:
            synth (numpy.ndarray): input data
        
        Returns:
            float
                                
        """
        return self.tapers * (self.obsd - self.ssynth)

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