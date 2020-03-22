#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for sending and receiving data from workers
and vice versa.

:copyright:
    Lucas Sawade (lsawade@princeton.edu), 2020
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

try:
    from mpi4py import MPI
except Exception as e:
    print(e)
    ValueError("mpi4py probably not installed")

import sys
import numpy as np
import pickle

"""MPI context manager"""


class MPI_comm(object):
    def __init__(self, file: str = None, mpi_size: int = None,
                 master: bool = True):
        if master:
            if type(file) == list:
                arg_list = file
            elif type(file) == str:
                arg_list = [file]
            else:
                ValueError("Wrong filetype.")

            self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                                            args=arg_list,
                                            maxprocs=mpi_size)
        else:
            self.comm = MPI.Comm.Get_parent()

    def __enter__(self):
        return self.comm

    def __exit__(self, type, value, traceback):
        self.comm.Disconnect()


def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]


"""
From master to worker: Dictionaries
"""


def broadcast_dict(d, comm):
    """Given a dictionary and a communicator. this function
    broadcasts the dictionary"""

    # Convert to string
    d_string = pickle.dumps(d)

    # Get bytesize
    dsize = get_dictionary_size(d)

    # Broad casting
    comm.Bcast([dsize, MPI.INT], root=MPI.ROOT)
    comm.Bcast([d_string, MPI.CHAR], root=MPI.ROOT)


def get_bcast_dictionary(comm):
    """This function gets a dictionary broadcasted
    from a parent process using a string and its
    bytesize. Works for any
    """
    # Create a an integer to ask for bytesize of string
    dsize = np.array(0, dtype="i")

    comm.Bcast([dsize, MPI.INT], root=0)

    # Preallocate string bytesize
    d_string = bytearray(dsize)

    # Receive string/bytearray
    comm.Bcast([d_string, MPI.CHAR], root=0)

    # Recreate dixtionary from string
    d = pickle.loads(d_string)

    return d


"""
Gather results dictionary. This means we need a worker
side function that sends the dictionary and a master side
function that Gathers the dictionaries.
"""


def get_dictionary_size(d: dict):
    """[summary]

    Args:
        d (dict): dictionary

    Returns:
        dictionary size in bytes
    """

    return np.array(sys.getsizeof(pickle.dumps(d)), 'i')


def send_result_dictionary(d: dict, comm):
    """Sends result dictionary to master worker.

    Args:
        d ([type]): dictionary of sorts
        comm ([type]): [description]
    """
    # Convert to string
    d_string = pickle.dumps(d)

    # Get bytesize
    dsize = get_dictionary_size(d)
    size = comm.Get_size()

    # Set rec size
    dsize_receive = bytearray(dsize) * size

    # Broad casting
    comm.Gather([d_string, MPI.CHAR], [dsize_receive, MPI.CHAR], root=0)


def get_result_dictionaries(sample_d, comm, size):
    """Receiving the results from the workers

    Args:
        sample_d (dict): sample dictionary to estimate size
        comm (mpi4py.MPI.Intercomm): MPI.Intercomm
        size (int): number of workers
                    IMPORTANT! ***NOT*** comm.Get_size()

    Returns:
        list: list of results dictionaries of the same format
              as the sample dictionary
    """

    # First determine size of incoming dictionary
    dsize = get_dictionary_size(sample_d)

    d_strings = bytearray(size * dsize)

    # Try to gather dictionaries
    comm.Gather([dsize, MPI.CHAR], [d_strings, MPI.CHAR], root=MPI.ROOT)

    """
    What is happening below is the worst form of trickery.
    When getting dictionaries from the workers they are transformed
    into "pickled" bytearrays and concatenate to the fill the variable
    d_strings. The problem now is that concatenation of bytearrays
    makes the concatenated bytearray smaller in size (I don't know why.)
    For that reason, what I'm doing is pretty hacky/ugly. I found the splitter
    of the pickled dictionary as seen below.
    I use it to split the byte array into dictionary pieces. Now since the
    bytearray has become smaller, there will be "end-bytes" that are leftover.
    Therefore, the strings[:-1]. to omit the last bit of unpopulated bytes.
    Then, I add the splitter again to be able to "unpickl" the dictionary.

    I don't know about you, but I think that's pretty dirty.

    """

    list_of_dicts = []
    splitter = b'\x00q(tq)bu.'
    bytestrings = d_strings.split(splitter)

    for string in bytestrings[:-1]:
        single_dict = pickle.loads(string + splitter)
        list_of_dicts.append(single_dict)

    return list_of_dicts
