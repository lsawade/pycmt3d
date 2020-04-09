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

from .gradient3d import Gradient
from mpi4py import MPI
import numpy as np
from copy import deepcopy
from .mpi_utils import get_bcast_dictionary
from .mpi_utils import send_result_dictionary
from . import logger

comm = MPI.Comm.Get_parent()

# Get dictionary that is broadcast to every worker
d = get_bcast_dictionary(comm)

rank = comm.Get_rank()

results = deepcopy(d["sample_d"])


for _i, job in enumerate(d["jobs"][rank]):

    if comm.Get_rank() == 0:
        logger.info("--> Computing approximately %d of %d ..."
                    % ((_i * comm.Get_size() + 1),
                        d["config"].bootstrap_repeat))

    ind = d["randarray"][job]

    G = Gradient(d["obsd"][ind, :],
                 d["synt"][ind, :],
                 d["tapers"][ind, :],
                 d["delta"],
                 method=d["config"].method,
                 ia=d["config"].ia, idt=d["config"].idt,
                 nt=d["config"].nt, nls=d["config"].nls,
                 crit=d["config"].crit,
                 precond=d["config"].precond,
                 reg=d["config"].reg)

    G.gradient()
    results["dt"][_i] = G.dt
    results["a"][_i] = G.a
    results["cost"][_i, :] = np.array(G.cost_list)
    results["cost_len"][_i] = G.it


# Send back the populated dictionary
send_result_dictionary(results, comm)

comm.Disconnect()
