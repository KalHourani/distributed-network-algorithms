#!/usr/bin/env python3

import random
import sys
import time

from mpi4py import MPI

import mpi_functions


def main(verbose=False):
    # MPI pre-processing
    comm = MPI.COMM_WORLD
    n, my_data = mpi_functions.pre_process()
    order_stat = order_statistic(n)
    if len(sys.argv) > 1:
        verbose = True
    # begin timing execution
    comm.barrier()
    t0 = time.time()
    pivot = quick_select(my_data, order_stat)
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


def check_number_of_elements(my_data, left, right, pivot, order_stat, leader=0, comm=MPI.COMM_WORLD):
    """
    Checks if number of elements found is equal to the order statistic. Returns triple
    (finished, left_partition, order_stat):
    finished - designates if we have found the order statistic
    left_partition - designates whether to proceed on left or right of pivot
    """
    rank = comm.Get_rank()
    index = partition(my_data, left, right, pivot)
    s1_size = index - left + 1
    s1_sizes = comm.gather(s1_size, root=leader)
    finished, left_partition = False, False
    if rank == leader:
        num_elements = sum(s1_sizes)
        if num_elements ==  order_stat:
            finished = True
        else:
            finished = False
            if num_elements > order_stat:
                left_partition = True
            else:
                left_partition = False
                order_stat -= num_elements
    return finished, left_partition, order_stat, index


def random_index(left, right):
    if left < right:
        return random.randint(left, right)
    else:
        return right


def quick_select(my_data, order_stat, leader=0, comm=MPI.COMM_WORLD):
    finished = False
    left, right = 0, len(my_data) - 1
    left_partition = False
    rank = comm.Get_rank()
    while not finished:
        pivot = communicate_pivots(my_data, left, right, leader=0)
        finished, left_partition, order_stat, index = check_number_of_elements(my_data, left, right, pivot, order_stat, leader=leader, comm=comm)
        order_stat = comm.bcast(order_stat, root=leader)
        left_partition = comm.bcast(left_partition, root=leader)
        finished = comm.bcast(finished, leader)
        if left_partition:
            right = index
        else:
            left = index + 1
    return pivot


def communicate_pivots(my_data, left, right, leader=0, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    index = random_index(left, right)
    pivot = my_data[index]
    weight = max(right - left + 1, 1)
    pivots = comm.gather(pivot, root=leader)
    weights = comm.gather(weight, root=leader)
    if rank == leader:
        pivot = weighted_random(pivots, weights)
    pivot = comm.bcast(pivot, root=leader)
    return pivot
    

def ordinal_suffix(n):
    if n % 10 == 0 or 11 <= n % 100 <= 19 or n % 10 > 3:
        return "th"
    if n % 10 == 1:
        return "st"
    elif n % 10 == 2:
        return "nd"
    elif n % 10 == 3:
        return "rd"


def ordinal(n):
    return "{0}{1}".format(n, ordinal_suffix(n))


def print_output(verbose, order_stat, t0, comm, pivot, leader=0):
    total_time = time.time() - t0
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if rank == leader:
        if verbose:
            print("The {0} order statistic is {1}.".format(ordinal(order_stat), pivot))
            print("Took {0} seconds with {1} processes.".format(total_time, num_procs))
        else:
            print(num_procs, total_time)

if __name__ == "__main__":
    main()
