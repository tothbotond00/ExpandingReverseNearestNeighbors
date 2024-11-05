import networkx as nx
import random
import time
import ernn

END_IN_ROUNDS = 20

# Graph class with Basic implemetation (Algorithm 2) - NP hard problem Brute Force
class BasicAlgorithm(ernn.ERNN):
    def edge_inspection(self, facility, facilities, users, budget):
        distances = nx.single_source_dijkstra_path_length(self.graph, facility)
        edges_by_distance = []

        # Sort only modifiable edges by their distance from the facility
        for u, v, data in self.graph.edges(data=True):
            if (u, v) in self.modifiable_edges or (v, u) in self.modifiable_edges:
                edge_distance = min(distances.get(u), distances.get(v))
                weight = data['weight']
                edges_by_distance.append((edge_distance, u, v, weight))

        # Sort edges by distance from the facility (closest first)
        edges_by_distance.sort()

        upgraded_edges = []
        current_rnn = self.compute_rnn(facility, facilities, users)
        start_rnn = current_rnn
        print(f"Initial RNN Size for Facility {facility}: {len(current_rnn)}")

        # Keep track of how many edges have been upgraded (according to the budget)
        upgraded_count = 0

        # Create a copy of the graph to simulate upgrades
        graph_copy = self.graph.copy()

        end_counter = 0
        for edge_distance, u, v, weight in edges_by_distance:
            if upgraded_count >= budget or end_counter > END_IN_ROUNDS:
                break  # Stop once the budget is exhausted

            # Simulate upgrading edge (set weight to 0) in the copied graph
            original_weight = graph_copy[u][v]['weight']
            graph_copy[u][v]['weight'] = 0
            self.graph = graph_copy  # Update the graph reference for computation

            new_rnn = self.compute_rnn(facility, facilities, users)

            # Check if the RNN size increases
            if len(new_rnn) > len(current_rnn):
                upgraded_edges.append((u, v, weight))
                current_rnn = new_rnn
                upgraded_count += 1
                print(f"Upgraded best edge ({u}, {v}) increased RNN size to {len(current_rnn)}")
            else:
                # Restore original weight if no gain
                graph_copy[u][v]['weight'] = original_weight
                #print(f"Restoring edge ({u}, {v}) to original weight, no RNN gain")
            end_counter += 1

        self.graph = graph_copy  # Update the main graph after upgrading edges
        return upgraded_edges, len(current_rnn), len(start_rnn)