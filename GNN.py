from fiction import pyfiction
import os
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.nn import GCNConv
import numpy as np


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
    benchmark = "trindade16"
    # for file in os.listdir(os.path.join(dir_path, "..", "benchmarks", benchmark)):
    # print(file)
    file = "mux21.v"
    path = os.path.join(dir_path, "benchmarks", benchmark, file)
    print(path)
    # path = os.path.join(dir_path, "mux21.v")
    network = pyfiction.read_logic_network(path)
    depth_params = pyfiction.fanout_substitution_params()
    depth_params.strategy = pyfiction.substitution_strategy.BREADTH
    network = pyfiction.fanout_substitution(network, depth_params)

    DG = nx.DiGraph()


    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        return torch.tensor(np.eye(num_classes, dtype='int')[y], dtype=torch.int64)
    # add nodes
    # DG.add_nodes_from(network.pis())
    DG.add_nodes_from([(pi, {"x": to_categorical(pi, 11)}) for pi in network.pis()])
    # DG.add_nodes_from(network.pos())
    DG.add_nodes_from([(po, {"x": to_categorical(po, 11)}) for po in network.pos()])
    # DG.add_nodes_from(network.gates())
    DG.add_nodes_from([(gate, {"x": to_categorical(gate, 11)}) for gate in network.gates()])

    # add edges
    for x in range(max(network.gates()) + 1):
        for pre in network.fanins(x):
            DG.add_edge(pre, x)

    dataset = from_networkx(DG)

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 16)
            self.conv2 = GCNConv(16, dataset.num_nodes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1)


    model = GCN()
    data = dataset
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        print('here')
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
