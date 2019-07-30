#!/usr/bin/env python3

import sys
from mpi4py import MPI
import time
import mpi_functions


def main(verbose=False):
    # MPI pre-processing
    comm = MPI.COMM_WORLD
    n, my_data = mpi_functions.pre_process()
    if len(sys.argv) > 1:
        verbose = True
    # begin timing execution
    comm.barrier()
    t0 = time.time()
    global_max = global_maximum(my_data)
    print_output(verbose, global_max, t0, comm)


def send_local_max(my_data, leader=0, tag=0, comm=MPI.COMM_WORLD):
    """
    Send maximum of my_data to leader.
    """
    local_max = max(my_data)
    comm.send(local_max, dest=leader, tag=tag)


def global_maximum(my_data, leader=0, tag=0, comm=MPI.COMM_WORLD):
    """
    Leader determines global maximum among all machines from my_data arrays.
    """
    send_local_max(my_data, leader=leader, tag=tag, comm=comm)
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    global_max = float("-inf")
    if rank == leader:
        for i in range(num_procs):
            global_max = max(global_max, comm.recv(source=i, tag=0))
    return global_max


def print_output(verbose, global_max, t0, comm, leader=0):
    total_time = time.time() - t0
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if rank == leader:
        if verbose:
            print("Global max is {0}. Took {1} seconds with {2} processes.".format(global_max, total_time, num_procs))
        else:
            print(num_procs, total_time)

if __name__ == "__main__":
    main()
