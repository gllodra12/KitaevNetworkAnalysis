import numpy as np
import pytest

from phase_transition_cna.network.metrics import (
    local_density, local_disparity,
    density, disparity,
    average_clustering,
    average_shortest_path
)

positive_weights = [1, 2, 3, 4, 5, 6]
mixed_weights = [-2, -1, 0, 1, 2, 3]
float_weights = [-0.8, -0.6, -0.2, 0.3, 0.5, 0.9]
network_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


@pytest.fixture
def graph():
    """ Simple weighted graph with the following structure
                  1 --- 0
                 / \
                2 - 3
    and weight distribution:
    {(0, 1): 10, (1, 2):20, (2, 3):15, (1.3):40}
    """
    edges = [(0, 1), (1, 2), (2, 3), (1, 3)]
    weights = [10, 20, 15, 40]
    return edges, weights


@pytest.mark.parametrize("edges, weights, my_result", [
    (network_edges, positive_weights, np.matrix([[2, 10/3, 4, 14/3]])),
    (network_edges, mixed_weights, np.matrix([[-1, 1/3, 1, 5/3]])),
    (network_edges, float_weights, np.matrix([[-8/15, 0, 1/5, 2/5]]))
])
def test_local_density(edges, weights, my_result):
    assert local_density(edges, weights) == pytest.approx(my_result, 1e-6)


@pytest.mark.parametrize("edges, weights, my_result", [
    (network_edges, positive_weights, np.matrix([[14/36, 42/100, 56/144, 70/196]])),
    (network_edges, mixed_weights, np.matrix([[5/9, 9, 11/9, 13/25]])),
    (network_edges, float_weights, np.matrix([[13/32, np.inf, 7/2, 55/72]]))
])
def test_local_disparity(edges, weights, my_result):
    assert local_disparity(edges, weights) == pytest.approx(my_result, 1e-6)


@pytest.mark.parametrize("edges, weights, my_result", [
    (network_edges, positive_weights, 3.5),
    (network_edges, mixed_weights, 0.5),
    (network_edges, float_weights, 1/60)
])
def test_density(edges, weights, my_result):
    assert density(edges, weights) == pytest.approx(my_result, 1e-6)


@pytest.mark.parametrize("edges, weights, my_result", [
    (network_edges, positive_weights, 0.388730),
    (network_edges, mixed_weights, 2.824444)
])
def test_disparity(edges, weights, my_result):
    assert disparity(edges, weights) == pytest.approx(my_result, abs=1e-6)


def test_shortest_path(graph):
    # Number of nodes
    n = 4
    # Shortest path between (0, 1): 10, (0, 2): 30, (0, 3): 45
    node0 = np.array([10, 30, 45])
    # Shortest path between (1, 0): 10, (1, 2): 20, (1, 3): 35
    node1 = np.array([10, 20, 35])
    # Shortest path between (2, 0): 30, (2, 1): 20, (2, 3): 15
    node2 = np.array([30, 20, 15])
    # Shortest path between (3, 0): 45, (3, 1): 35, (3, 2): 15
    node3 = np.array([45, 35, 15])

    S_ij = np.sum(node0+node1+node2+node3)
    edges, weights = graph
    assert average_shortest_path(edges, weights) == S_ij/(n*(n-1))


def test_clustering(graph):
    """ Given a graph we check if the clustering computes
    the ratio between closed triplets and open triplets.

    The function graph() have 6 closed triplets.
    Luckily, all the triplets have the same value, since only orientation is shifted.
        One closed triplet is:
            1-2-3-1 with value 20*15*40=12000
        Another closed triplet would be:
            1-3-2-1
        Next triplets will be:
            2-3-1-2
            2-1-3-2
            3-1-2-3
            3-2-1-3
        The clustering formula take into account all this triplets, even though they are the same.

    A total of 10 open triplets can be found in the function graph().
        0-1-2 --> 10*20=200
        0-1-3 --> 10*40=400
        1-2-3 --> 20*15=300
        1-3-2 --> 40*15=600
        2-1-3 --> 20*40=800
        2-1-0 --> 20*10=200
        2-3-1 --> 15*40=600
        3-1-2 --> 40*20=800
        3-1-0 --> 40*10=400
        3-2-1 --> 15*20=300

    As you we get large number, so we normalize the weights by dividing with the biggest one,
    in this case is 40.
    So, the closed triplet is:
        20/40*15/40*40/40 = 0.1875
    and the open triplets:
        10/40*20/40 = 1/8
    """
    norm = 40
    closed_triplets = 6*np.array([12000])
    open_triplets = 2*np.array([200, 300, 400, 600, 800])
    ratio = (closed_triplets/norm**3)/(np.sum(open_triplets)/norm**2)

    edges, weights = graph
    assert average_clustering(edges, weights) == ratio
