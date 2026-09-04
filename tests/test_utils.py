import networkx as nx

from mnt.nanoplacer.placement_envs.utils.placement_utils import (
    create_action_list,
    map_to_discrete,
    map_to_multidiscrete,
    topological_generations,
    topological_sort,
)


def test_map_to_multidiscrete() -> None:
    assert map_to_multidiscrete(5, 3) == (2, 1)
    assert map_to_multidiscrete(0, 3) == (0, 0)


def test_map_to_discrete() -> None:
    assert map_to_discrete(2, 1, 3) == 5
    assert map_to_discrete(0, 0, 3) == 0


def test_topological_sort() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

    result = list(topological_generations(graph))
    assert result in [[1, 2, 3, 4], [1, 3, 2, 4]]
    result = list(topological_sort(graph))
    assert result in [[1, 2, 3, 4], [1, 3, 2, 4]]


def test_create_action_list() -> None:
    _, node_to_action, actions, graph, pi_names, po_names = create_action_list("trindade16", "mux21")

    assert node_to_action == {
        2: "INPUT",
        3: "INPUT",
        4: "INPUT",
        5: "FAN-OUT",
        6: "INV",
        7: "AND",
        8: "AND",
        9: "OR",
        10: "OUTPUT",
    }
    assert actions == [2, 3, 4, 5, 8, 6, 7, 9, 10]
    assert list(graph.edges()) == [(2, 7), (3, 8), (4, 5), (5, 6), (5, 8), (6, 7), (7, 9), (8, 9), (9, 10)]
    assert pi_names == ["in0", "in1", "in2"]
    assert po_names == ["out"]
