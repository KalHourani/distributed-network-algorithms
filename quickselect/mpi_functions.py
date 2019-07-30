#!/usr/bin/env python3

import sys
from mpi4py import MPI


def pre_process(comm, my_data, leader=0, tag=0):
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    n = 0
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
    return n
