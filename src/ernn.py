import networkx as nx
import random
import heapq
import time
import sys

class ERNN:
    def __init__(self, graph, modifiable_edges):
        self.graph = graph
        self.modifiable_edges = modifiable_edges

    def compute_rnn(self, target_facility, facilities, users):
        # Computes RNN for a target facility
        facility_distances = {}
        for facility in facilities:
            distances = nx.single_source_dijkstra_path_length(self.graph, facility)
            facility_distances[facility] = distances

        rnn_set = set()
        for user in users:
            min_distance = float('inf')
            nearest_facility = None
            for facility in facilities:
                dist = facility_distances[facility].get(user, float('inf'))
                if dist < min_distance:
                    min_distance = dist
                    nearest_facility = facility
            if nearest_facility == target_facility:
                rnn_set.add(user)

        return rnn_set
    
    def edge_inspection(self, facility, facilities, users, budget):
        pass