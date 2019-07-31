#!/usr/bin/env python3

import sys
from mpi4py import MPI

fragment_id = {}


class Node:
    def __init__(self, id):
        self.id = id
        self.neighbors = []
        self.weights = {}
        self.root = False

    def edge_weight(self, neighbor, weight):
        self.weights[neighbor] = weight

    def mwoe(self):
        self.neighbors.sort(key=lambda neighbor: weights[neighbor])
        global fragment_id
        for neighbor in self.neighbors:
            if fragment_id[neighbor] != fragment_id[self.id]:
                yield neighbor


def pre_process(leader=0, tag=0, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    n = 0
    if rank == leader:
        n = int(sys.stdin.readline())
        graph = [0] * n
        for line in sys.stdin:
            if not line.strip():
                continue
            u_id, v_id, weight = [float(x) for x in line.split(' ')]
            u_id, v_id = int(u_id), int(v_id)
            if not graph[u_id]:
                graph[u_id] = Node(u_id)
            graph[u_id].neighbors.append(v_id)
            graph[u_id].weights[v_id] = weight
            if not graph[v_id]:
                graph[v_id] = Node(v_id)
            graph[v_id].neighbors.append(u_id)
            graph[v_id].weights[u_id] = weight
        for node in graph:
            comm.send(node, dest=node.id % num_procs, tag=tag)
            fragment_id[node.id] = 0
    n = comm.bcast(n, root=leader)
    length = n // num_procs
    if rank == 0:
        length += n % num_procs
    my_data = []
    for _ in range(length):
        my_data.append(comm.recv(source=leader, tag=tag))
    return n, my_data


def main(comm=MPI.COMM_WORLD):
    n, my_nodes = pre_process()
    rank = comm.Get_rank()
    for node in my_nodes:
        print(rank, node.id, node.neighbors)

if __name__ == "__main__":
    main()
