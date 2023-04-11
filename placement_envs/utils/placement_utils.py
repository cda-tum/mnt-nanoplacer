from fiction import pyfiction
import os
import networkx as nx
import math


def map_to_multidiscrete(action, layout_width):
    b = int(action / layout_width)
    a = action % layout_width
    return [a, b]


def map_to_discrete(a, b, layout_width):
    action = 0
    action += a
    action += b * layout_width
    return action


def create_action_list(benchmark, function):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, f"../../benchmarks/{benchmark}/{function}.v")
    network = pyfiction.read_logic_network(path)
    params = pyfiction.fanout_substitution_params()
    params.strategy = pyfiction.substitution_strategy.DEPTH
    network = pyfiction.fanout_substitution(network, params)

    DG = nx.DiGraph()

    # add nodes
    DG.add_nodes_from(network.pis())
    DG.add_nodes_from(network.pos())
    DG.add_nodes_from(network.gates())

    # add edges
    for x in range(max(network.gates()) + 1):
        for pre in network.fanins(x):
            DG.add_edge(pre, x)

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

    actions = list(topological_sort(DG))

    node_to_action = {}
    for action in actions:
        if network.is_pi(action):
            node_to_action[action] = "INPUT"
        elif network.is_po(action):
            node_to_action[action] = "OUTPUT"
        elif network.is_inv(action):
            node_to_action[action] = "INV"
        elif network.is_and(action):
            node_to_action[action] = "AND"
        elif network.is_or(action):
            node_to_action[action] = "OR"
        elif network.is_nand(action):
            node_to_action[action] = "NAND"
        elif network.is_nor(action):
            node_to_action[action] = "NOR"
        elif network.is_xor(action):
            node_to_action[action] = "XOR"
        elif network.is_xnor(action):
            node_to_action[action] = "XNOR"
        elif network.is_maj(action):
            node_to_action[action] = "MAJ"
        elif network.is_fanout(action):
            node_to_action[action] = "FAN-OUT"
        elif network.is_buf(action):
            node_to_action[action] = "BUF"
        else:
            raise Exception(f"{action}")
    return network, node_to_action, actions, DG
