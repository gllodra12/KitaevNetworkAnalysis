import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix


def node_importance(edges, weights):
    matrix = adjacency_matrix(edges, weights)
    return np.sum(matrix, axis=0).reshape((1, matrix.shape[0]))


def adjacency_matrix(edges, weights):
    """Each edge has a position in the matrix and each
    matrix element has a value given by the weights.

    Args:
        edges (list): [(0, 1), (0, 2), (1, 2)]
        weights (list): [3, 2, 6]

    Returns:
        sparse matrix: Matrix representation of the network
         [[0, 3, 2]
          [3, 0, 6]
          [2, 6, 0]]
    """
    num_nodes = get_number_of_nodes(len(weights))
    rows, columns = zip(*edges)
    upper_triangular = coo_matrix((weights, (rows, columns)), shape=(num_nodes, num_nodes))
    lower_triangular = coo_matrix((weights, (columns, rows)), shape=(num_nodes, num_nodes))
    return upper_triangular + lower_triangular


def get_number_of_nodes(num_edges):
    """Given a complete graph we can compute the number of edges
    using e=n(n-1)/2, where n is the number of nodes.

    To get the number of nodes, given the number of edges we compute the
    inverse function n=(1+sqrt(1+8e))/2

    Args:
        num_edges (int): Number of edges in the graph

    Returns:
        int: Number of nodes in the graph
    """
    return int((1 + np.sqrt(1 + 8 * num_edges)) / 2)


def convert_graph_to_matrix(G, nodelist=None, dtype=None, order=None, weight="weight"):
    """Convert a each source --> target edge to a site in a matrix,
    edge (2, 4) it s mapped to 2nd row and 4th column of the matrix
    """
    if nodelist is None:
        nodelist = list(G)
    nodeset = set(nodelist)

    if len(nodelist) != len(nodeset):
        msg = "Ambiguous ordering: `nodelist` contained duplicates."
        raise nx.NetworkXError(msg)

    nlen = len(nodelist)

    A = np.full((nlen, nlen), np.nan, order=order)
    for u, nbrdict in G.adjacency():
        for v, d in nbrdict.items():
            try:
                A[u, v] = d.get(weight, 1)
            except KeyError:
                # This occurs when there are fewer desired nodes than
                # there are nodes in the graph: len(nodelist) < len(G)
                raise ValueError("Desired nodelist doesn't match with nodes in the graph.")
    A[np.isnan(A)] = 0
    A = np.asarray(A, dtype=dtype)
    return A


def eigenvalues(matrix):
    eig_value, _ = np.linalg.eig(matrix)
    return eig_value
