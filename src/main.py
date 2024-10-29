import networkx as nx
import random
import time
import sys
import basic
import dbeigreedy
import ernn
from quickchart import QuickChart

# Function to load a test dataset and simulate DBEI algorithm
def load_test_dataset(file_path):
    G = nx.Graph()

    with open(file_path, 'r') as f:
        next(f)  # Skip the first line containing metadata
        for line in f:
            u, v, weight = map(int, line.strip().split())
            G.add_edge(u, v, weight=weight)

    return G

# Simulating the selection of POIs and testing DBEI
def run_test(file_path, num_of_pois=100, rounds=4, budget=5, rnd_seed=1234, type_g="basic"):
    G = load_test_dataset(file_path)
    
    # Randomly select 30% of the edges as modifiable
    all_edges = list(G.edges)
    random.seed(rnd_seed)
    modifiable_edges = set(random.sample(all_edges, int(len(all_edges) * 0.3)))

    if (type_g == "basic"):
        ernn = basic.BasicAlgorithm(G, modifiable_edges)
    elif (type_g == "dbei"):
        ernn = dbeigreedy.DBEIGreedyAlgorithm(G, modifiable_edges)
    
    # Select random POIs (facilities) and assign the rest as users
    all_nodes = list(G.nodes)
    random.seed(rnd_seed)
    facilities = set(random.sample(all_nodes, num_of_pois))
    users = set(all_nodes) - facilities

    total_gain = 0
    total_runtime = 0
    rnn_sizes = []
    old_rnn_sizes = []

    for i in range(rounds):
        target_facility = random.choice(list(facilities))
        print(f"Round {i + 1}: Target Facility = {target_facility}")
        
        start_time = time.time()
        # Run the DBEI algorithm for this target facility
        upgraded_edges, rnn_size, old_size = ernn.edge_inspection(target_facility, facilities, users, budget)
        runtime = time.time() - start_time

        rnn_sizes.append(rnn_size)
        old_rnn_sizes.append(old_size)
        total_gain += rnn_size - old_size
        total_runtime += runtime

        print(f"RNN Size = {rnn_size}, Upgraded Edges: {upgraded_edges}, Runtime: {runtime:.4f} seconds")

    average_gain = total_gain / rounds
    average_runtime = total_runtime / rounds
    print(f"RNN sizes after each other: {rnn_sizes}")
    print(f"Average RNN Size: {average_gain}")
    print(f"Average Runtime: {average_runtime:.4f} seconds")
    qc = QuickChart()
    qc.width = 500
    qc.height = 300
    qc.device_pixel_ratio = 2.0
    qc.config = {
        "type": 'bar',
        "data": {
            "labels": list(range(rounds)),
            "datasets": [{
                "label": 'Starting RNN',
                "data": old_rnn_sizes,
            }, {
                "label": 'New RNN',
                "data": rnn_sizes,
            }]
        },
        "options": {
            "title": {
                "display": True,
                "position": "top",
                "fontSize": 12,
                "fontFamily": "sans-serif",
                "fontColor": "#666666",
                "fontStyle": "bold",
                "padding": 10,
                "lineHeight": 1.2,
                "text": f"AVG GAIN: {average_gain} AVG TIME: {average_runtime:.4f} BUGET: {budget} POI NUMBER: {num_of_pois}"
            },
        }
    }
    qc.to_file(f'./img/{sys.argv[1]}_{sys.argv[5]}_{rnd_seed}.png')

# run DBEI test
run_test(f'data/{sys.argv[1]}.tmp', num_of_pois=int(sys.argv[2]), rounds=int(sys.argv[3]), budget=int(sys.argv[4]), rnd_seed=int(random.uniform(1, 10000)), type_g=sys.argv[5])
