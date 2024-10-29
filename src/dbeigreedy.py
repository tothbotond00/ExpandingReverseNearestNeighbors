import networkx as nx
import random
import heapq
import time
import sys
import ernn

END_IN_ROUNDS = 20

# Graph class with Greedy DBEI Algorithm (Algorithm 3)
class DBEIGreedyAlgorithm(ernn.ERNN):
    def edge_inspection(self, facility, facilities, users, budget):
        distances = nx.single_source_dijkstra_path_length(self.graph, facility)
        current_rnn = self.compute_rnn(facility, facilities, users)
        start_rnn = current_rnn

        # Priority queue to explore edges by their distance
        edge_queue = []
        for u, v, data in self.graph.edges(data=True):
            if ((u, v) in self.modifiable_edges or (v, u) in self.modifiable_edges):
                edge_distance = min(distances.get(u), distances.get(v))
                heapq.heappush(edge_queue, (edge_distance, u, v, data['weight']))

        upgraded_edges = []
        upgraded_count = 0

        # Copy graph for incremental updates
        graph_copy = self.graph.copy()
        
        end_counter = 0

        while upgraded_count < budget and edge_queue:
            if end_counter > END_IN_ROUNDS:
                break 
            edge_distance, u, v, weight = heapq.heappop(edge_queue)

            print(f"Inspecting edge ({u}, {v}) with distance {edge_distance} and weight {weight}")

            # Simulate upgrading the edge by setting weight to 0
            original_weight = graph_copy[u][v]['weight']
            graph_copy[u][v]['weight'] = 0
            self.graph = graph_copy

            # Compute new RNN size after upgrading the edge
            new_rnn = self.compute_rnn(facility, facilities, users)

            # Check if this edge upgrade improves the RNN gain
            if len(new_rnn) > len(current_rnn):
                print(f"Upgrading edge ({u}, {v}) increases RNN size from {len(new_rnn)} to {len(current_rnn)}")
                current_rnn = new_rnn
                upgraded_edges.append((u, v, weight))
                upgraded_count += 1
            else:
                # Restore original weight if no gain
                graph_copy[u][v]['weight'] = original_weight
                print(f"Restoring edge ({u}, {v}) to original weight, no RNN gain")
            end_counter += 1

        return upgraded_edges, len(current_rnn), len(start_rnn)