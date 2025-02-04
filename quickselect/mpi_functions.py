#!/usr/bin/env python3

import sys
from mpi4py import MPI


def pre_process(leader=0, tag=0, comm=MPI.COMM_WORLD):
    """
    Read data from stdin and distribute evenly to machines in comm
    If the data size is not evenly divisible by num_procs,
    send remainder to leader. Return size of data and local data.
    """
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    n = 0
    my_data = []
    if rank == leader:
        for line in sys.stdin:
            n += 1
            comm.send(int(line), dest=n % num_procs, tag=tag)
    n = comm.bcast(n, root=leader)
    length = n // num_procs
    if rank == 0:
        length += n % num_procs
    for _ in range(length):
        my_data.append(comm.recv(source=leader, tag=tag))
    return n, my_data
