#!/usr/bin/env python3

import sys
from mpi4py import MPI
import time
import mpi_functions
import random


def main(verbose=False):
    # MPI pre-processing
    comm = MPI.COMM_WORLD
    my_data = []
    n = mpi_functions.pre_process(comm, my_data)
    order_stat = order_statistic(n)
    if len(sys.argv) > 1:
        verbose = True
    # begin timing execution
    comm.barrier()
    t0 = time.time()
    pivot = communicate_pivots(comm, my_data, order_stat)
    print_output(verbose, order_stat, t0, comm, pivot)


def order_statistic(n):
    if len(sys.argv) > 2:
        order_stat = int(sys.argv[2])
    else:
        order_stat = n // 2
    if not 1 <= order_stat <= n:
        order_stat = n // 2
    return order_stat


def partition(array, left, right, pivot):
    """ Partition an array into [S1,S2]
     * where elements of S1 are less than
     * or equal to pivot and elements of
     * S2 are greater than pivot
     * using the Hoare partition scheme.
     """
    i, j = left, right
    while True:
        print(i, j)
        while i <= right and array[i] <= pivot:
            i += 1
        while j >= left and array[j] > pivot:
            j -= 1
        if i >= j:
            return j
        array[i], array[j] = array[j], array[i]


def weighted_random(array, weights):
    s = random.randint(1, sum(weights))
    for i, weight in enumerate(weights):
        s -= weight
        if s <= 0:
            return array[i]


def communicate_pivots(comm, my_data, order_stat, leader=0):
    finished = False
    left, right = 0, len(my_data) - 1
    left_partition = False
    rank = comm.Get_rank()
    while not finished:
        if left < right:
            random_index = random.randint(left, right)
        else:
            random_index = right
        pivot = my_data[random_index]
        weight = max(right - left + 1, 0)
        pivots = comm.gather(pivot, leader)
        weights = comm.gather(weight, leader)
        if rank == leader:
            pivot = weighted_random(pivots, weights)
        pivot = comm.bcast(pivot, leader)
        index = partition(my_data, left, right, pivot)
        s1_size = index - left + 1
        s1_sizes = comm.gather(s1_size, leader)
        if rank == leader:
            num_elements = sum(s1_sizes)
            if num_elements == order_stat:
                finished = True
            elif num_elements > order_stat:
                left_partition = True
            else:
                order_stat -= num_elements
                left_partition = False
        order_stat = comm.bcast(order_stat, leader)
        left_partition = comm.bcast(left_partition, leader)
        finished = comm.bcast(finished, leader)
        if left_partition:
            right = index
        else:
            left = index + 1
    return pivot


def ordinal_suffix(n):
    if n % 10 == 1:
        return "st"
    elif n % 10 == 2:
        return "nd"
    elif n % 10 == 3:
        return "rd"
    else:
        return "th"


def print_output(verbose, order_stat, t0, comm, pivot, leader=0):
    total_time = time.time() - t0
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if rank == leader:
        if verbose:
            print("The {0}{1} order statistic is {2}. Took {3} seconds with {4} processes.".format(order_stat, ordinal_suffix(order_stat), pivot, total_time, num_procs))
        else:
            print(num_procs, total_time)

if __name__ == "__main__":
    main()
