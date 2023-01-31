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
    benchmark = "fontes18"
    # for file in os.listdir(os.path.join(dir_path, "benchmarks", benchmark)):
    # print(file)
    file = "FA.v"
    path = os.path.join(dir_path, "..", "benchmarks", benchmark, file)
    print(path)
    # path = os.path.join(dir_path, "mux21.v")
    network = pyfiction.read_logic_network(path)
    depth_params = pyfiction.fanout_substitution_params()
    depth_params.strategy = pyfiction.substitution_strategy.BREADTH
    network = pyfiction.fanout_substitution(network, depth_params)

    DG = nx.DiGraph()

    # add nodes
    DG.add_nodes_from(network.pis())
    DG.add_nodes_from(network.pos())
    DG.add_nodes_from(network.gates())

    # add edges
    for x in range(max(network.gates()) + 1):
        for pre in network.fanins(x):
            DG.add_edge(pre, x)

    actions_test = network.nodes()
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
            print(action)

    i = 0
    for action in node_to_action:
        i += 1
        print(f"{i}: {node_to_action[action]}")
    print(len(actions))

    # params = pyfiction.orthogonal_params()
    params = pyfiction.exact_params()
    params.border_io = True
    params.crossings = True
    params.scheme = "USE"
    params.timeout = 7200000
    import time

    start = time.time()
    # layout = pyfiction.exact_cartesian(network, params)
    # layout = pyfiction.orthogonal(network, params)
    end = time.time()
    print(end - start)
    # print(layout)
    # cell_layout = pyfiction.apply_qca_one_library(layout)
    # pyfiction.write_qca_layout_svg(
    #     cell_layout, f"{file}_ortho.svg", pyfiction.write_qca_layout_svg_params()
    # )
    # print(pyfiction.equivalence_checking(network, layout))

    # pos = nx.spring_layout(DG, k=0.1)
    # plt.figure(3, figsize=(30, 30))
    # nx.draw(DG, pos=pos)
    # nx.draw_networkx_labels(DG, pos=pos)
    # plt.show()

    # path = nx.all_pairs_shortest_path_length(DG)  # This is a generator
    # dpath = {x[0]: x[1] for x in path}  # Create a dictionary from it
    #    To find e.g. the distance between node 1 and node 4:
    # print(dpath[1][4])  # = 2

    # clpl: 18x26=468 (ortho) <1s
    # clpl: TO (exact) after 4200s
    # clpl: 15x15=225 900s (RL)
    # xor5_r1: 14x33 = 462 (ortho)
    # xor5_r1: TO after 7200s
    # xor5_r1: 16x16 = 256
    # parity: 48x120=
    # xor5R: 17x41= (ortho)
    # xor5R: (RL)
    # 1bitAdderMaj: 14x36 = 504
