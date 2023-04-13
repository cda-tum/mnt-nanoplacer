import os

import networkx as nx
from fiction import pyfiction


def map_to_multidiscrete(action: int, layout_width: int) -> tuple[int, int]:
    """Map a discrete action to the corresponding coordinate on a Cartesian grid.

    :param action:         Discrete action used by the RL agent
    :param layout_width    Width of the layout

    :return:               Coordinate on Cartesian grid
    """
    x = action % layout_width
    y = int(action / layout_width)

    return x, y


def map_to_discrete(x: int, y: int, layout_width: int) -> int:
    """Inverse function of 'map_to_multidiscrete'.
    Takes the coordinate on a Cartesian grid and maps it to a single discrete number.

    :param x:               X-coordinate
    :param y:               Y-ccordinate
    :param layout_width:    Width of the layout

    :return:                Discrete representation of the coordinate
    """
    action = 0
    action += x
    action += y * layout_width
    return action


def topological_generations(dg: nx.DiGraph) -> int:
    """Create a topological ordering of a network in a depth-first way and yields each node.

    :param dg:         Logic network (graph)

    :return:           Current node from the network
    """
    indegree_map = {v: d for v, d in dg.in_degree() if d > 0}
    zero_indegree = [v for v, d in dg.in_degree() if d == 0]

    while zero_indegree:
        node = zero_indegree[0]
        zero_indegree = zero_indegree[1:] if len(zero_indegree) > 1 else []

        for child in dg.neighbors(node):
            indegree_map[child] -= 1
            if indegree_map[child] == 0:
                zero_indegree.insert(0, child)
                del indegree_map[child]
        yield node


def topological_sort(dg: nx.DiGraph) -> int:
    """Create a topological ordering of a network in a depth-first way and yields each node.

    :param dg:         Logic network

    :return:           Current node
    """

    yield from topological_generations(dg)


def create_action_list(benchmark, function) -> tuple[pyfiction.logic_network, dict[int, str], list[int], nx.DiGraph]:
    """Create a topological odering of the network and a mapping of node to gate type.

    :param benchmark:    Benchmark set
    :param function:     Function in the benchmark set

    :return:    network:           Network of the logic function
    :return:    node_to_action:    Dictionary mapping node to gate type
    :return:    actions:           Topological sort of the network nodes
    :return:    dg:                Digraph representation of the logic network
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, f"../../benchmarks/{benchmark}/{function}.v")
    network = pyfiction.read_logic_network(path)
    params = pyfiction.fanout_substitution_params()
    params.strategy = pyfiction.substitution_strategy.DEPTH
    network = pyfiction.fanout_substitution(network, params)

    dg = nx.DiGraph()

    # add nodes
    dg.add_nodes_from(network.pis())
    dg.add_nodes_from(network.pos())
    dg.add_nodes_from(network.gates())

    # add edges
    for x in range(max(network.gates()) + 1):
        for pre in network.fanins(x):
            dg.add_edge(pre, x)

    actions = list(topological_sort(dg))

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
    return network, node_to_action, actions, dg
