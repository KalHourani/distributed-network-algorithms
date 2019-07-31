#!/usr/bin/env python3

import mpi_functions
from mpi4py import MPI


def main(comm=MPI.COMM_WORLD):
    n, my_nodes = mpi_functions.pre_process()
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    

def fragment_moe(my_nodes, comm=MPI.COMM_WORLD):
    for node in my_nodes:
        for edge in node.mwoe():
            comm.send()
