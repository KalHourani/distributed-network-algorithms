 #!/usr/bin/env python3

import mpi_functions
import collections


class Node:
    def __init__(self, id, neighbors):
        self.id = id
        self.neighbors = neighbors
        self.weights = collections.defaultdict(lambda: 0)
        self.fragment_id = id

    def edge_weight(self, neighbor, weight):
        self.weights[neighbor] = weight

    def mwoe(fragment_id):
        self.neighbors.sort(key=lambda neighbor: weights[neighbor])
        for neighbor in neighbors:
            if neighbor.