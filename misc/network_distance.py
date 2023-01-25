from fiction import pyfiction
import os
import networkx as nx
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))


def topological_generations(G):
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    zero_indegree = [v for v, d in G.in_degree() if d == 0]

    while zero_indegree:
        node = zero_indegree[0]
        if len(zero_indegree) > 1:
            zero_indegree = zero_indegree[1:]
        else:
            zero_indegree = []

        for child in G.neighbors(node):
            indegree_map[child] -= 1
            if indegree_map[child] == 0:
                zero_indegree.insert(0, child)
                del indegree_map[child]
        yield node


def topological_sort(G):
    for generation in topological_generations(G):
        yield generation


if __name__ == "__main__":
    path = os.path.join(dir_path, "benchmarks/fontes18/xor5_r1.v")
    # path = os.path.join(dir_path, "mux21.v")
    network = pyfiction.read_logic_network(path)
    network = pyfiction.fanout_substitution(network)

    DG = nx.DiGraph()

    # add nodes
    DG.add_nodes_from(network.pis())
    DG.add_nodes_from(network.pos())
    DG.add_nodes_from(network.gates())

    # add edges
    for x in range(max(network.gates()) + 1):
        for pre in network.fanins(x):
            DG.add_edge(pre, x)

    path = nx.all_pairs_shortest_path_length(DG.to_undirected())  # This is a generator
    dpath = {x[0]: x[1] for x in path}  # Create a dictionary from it
    ordered_dict = dict(sorted(dpath.items()))
    ordered_dict_node_2 = dict(sorted(dpath[2].items()))
    print(ordered_dict_node_2)  # = 2
    print(ordered_dict_node_2.values())

    from node2vec import Node2Vec

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(DG, dimensions=12, walk_length=20, num_walks=200, workers=4)  # Use temp_folder for big graphs

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Look for most similar nodes
    print(model.wv["40"])
