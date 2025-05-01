from itertools import combinations

import networkx as nx
import numpy as np


def complete_graph(num_nodes: int, edges: list, weights: list):
    """Convert an incomplete graph to a complete graph.

    Example:
    Given a graph with num_nodes = 4 with:
        edges = [(0, 1), (0, 2)]
        weights = [  10,    20]

    We expected a complete graph with:
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        weights = [  10,     20,     10,     10,     20,    10]

    Args:
        num_nodes (int): Number of nodes we want in the complete graph.
        edges (list(tuple(int, int))): link that connects two nodes in the incomplete graph.
        weights (list): Numerical value that label each edge in the incomplete graph.

    Returns:
        edges (list(tuple(int, int))): All the links in the complete graph.
        weights (list): Weights of the complete graph.
    """
    # Compute the periodic distance between all edges in the complete graph.
    edge_to_distance = dict()
    for new_edge in combinations(range(num_nodes), 2):
        source, target = new_edge
        distance = abs(target - source)
        edge_to_distance[new_edge] = periodic_distance(distance, num_nodes)

    # Maps every distance in the incomplete graph to a weight
    distance_to_weight = dict()
    for idx, edge in enumerate(edges):
        source, target = edge
        distance = abs(target - source)
        distance_to_weight[distance] = weights[idx]

    # Combines the previous two dicts to get a weight for each edge in the complete graph.
    edge_to_weight = dict()
    for edge, distance in edge_to_distance.items():
        edge_to_weight[edge] = distance_to_weight[distance]

    return list(edge_to_weight.keys()), list(edge_to_weight.values())


def weighted_graph(edges: list, weights: list):
    network = nx.Graph()
    network.add_weighted_edges_from([(source, target, weight) for (source, target), weight in zip(edges, weights)])
    return network


def periodic_distance(distance, system_size):
    """Distance between two when periodic boundary conditions are applied.

    Given an edge (0, 3) you may expected a distance = 3. However if
    periodic boundary conditions are applied the result is different.

    0---1---2
     \     /
      \   /
        3

    In this case for a system_size = 4 the edge (0, 3) has a distance = 1.

    Args:
        distance (int): [description]
        system_size (int): [description]

    Returns:
        int: Shortest distance between two nodes.
    """
    return int(np.where(distance < system_size / 2, distance, system_size - distance))
