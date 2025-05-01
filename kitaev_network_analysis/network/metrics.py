import networkx as nx
import numpy as np

from .build_network import weighted_graph
from .utils import convert_graph_to_matrix, node_importance


def local_density(edges: list, weights: list):
    """Measure the node importance by getting the sum of all its links and
    dividing by its connections.

      $$ d_i = \frac{\sum_j e_{ij}}{N-1} $$

    Args:
        edges (list): link that connects two nodes in the incomplete graph.
        weights (list): Numerical value that label each edge in the incomplete graph.

    Returns:
        np.array: Each array value correspond to a node density.
            Example: np.array([1.5, 0.5])
                node_1 has a density 1.5
                node_2 has a density 0.5
    """
    sum_edges = node_importance(edges, weights)
    number_of_nodes = sum_edges.shape[1]
    return 1 / (number_of_nodes - 1) * sum_edges


def local_disparity(edges: list, weights: list):
    """Characterizes the level of local heterogeneity.
    If all the links have the same influence over a node (homogenous) we have minimum disparity (Y=1/n).
    If links over a specific node have different values, the disparity will increase.

        $$ Y_i = \frac{\sum_j e_{ij}^2}{(\sum_j e_{ij})^2} $$

    Args:
        edges (list): link that connects two nodes in the incomplete graph.
        weights (list): Numerical value that label each edge in the incomplete graph.

    Returns:
          np.array: Each array value correspond to a node disparity.
            Example: np.array([1.5, 0.5])
                node_1 has a disparity 1.5
                node_2 has a disparity 0.5
    """
    sum_edges = node_importance(edges, weights)
    weights_square = [weight**2 for weight in weights]
    sum_square_edges = node_importance(edges, weights_square)
    return np.multiply(1 / np.square(sum_edges), sum_square_edges)


def disparity(edges, weights):
    return np.mean(local_disparity(edges, weights))


def density(edges, weights):
    return np.mean(local_density(edges, weights))


def betweeness_centrality(graph, node):
    betweeness_dict = nx.betweenness_centrality(graph, weight="weight")
    return betweeness_dict[int(node)]


def average_clustering(edges, weights):
    """Clustering is defined as the ratio between closed triplets (3 nodes connected
    by 3 edges) and open triplets (3 nodes connected by 2 edges). For weighted networks, this
    concept can be generalized by the following formulas.

     $$ C = \frac{# closed triplets}{open triplets}
          = \frac{\sum_{i \neq j \neq k} e_{ij} e_{jk} e_{ki}}{\sum_k \sum_{i \neq j \neq k} e_{ik} e_{jk}}
          = \frac{Tr(A^3)}{\sum_{j \neq i}\sum_{i} (A^2)_{ij}} $$

    where the 3 formulas are equivalent and A is the adjacency matrix.
    Args:
        edges (list): link that connects two nodes in the incomplete graph.
        weights (list): Numerical value that label each edge in the incomplete graph.

    Returns:
        float: Ratio between closed triplets and open triplets.
    """
    graph = weighted_graph(edges, weights)
    matrix = convert_graph_to_matrix(graph)
    max_weight = np.max(matrix)
    normalize_matrix = matrix / max_weight

    m2 = np.linalg.matrix_power(normalize_matrix, 2)
    m3 = np.matmul(m2, normalize_matrix)

    # Computing the Trace of a matrix
    numerator = np.sum(np.diag(m3))

    # Sum over off-diagonal values
    denominator = np.sum(m2) - np.sum(np.diag(m2))
    return numerator / denominator


def average_shortest_path(edges, weights):
    """Computes the shortest path between node i and node j. Once all the
    shortest paths are computed we get the average.

      $$ S = \frac{\sum_{ij} S_{ij}}{N(N-1)} $$

    where $S_{ij}$ is the shortest path between node i and j.
    And N is the number of nodes in the graph.

    Args:
        edges (list): link that connects two nodes in the incomplete graph.
        weights (list): Numerical value that label each edge in the incomplete graph.

    Returns:
        Float: Average shortest path
    """
    graph = weighted_graph(edges, weights)
    return nx.average_shortest_path_length(graph, weight="weight")
