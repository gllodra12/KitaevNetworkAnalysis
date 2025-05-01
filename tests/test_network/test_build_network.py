import pytest

from phase_transition_cna.network.build_network import complete_graph, periodic_distance


@pytest.mark.parametrize("edge, system_size, result", [
    ((0, 1), 4, 1),
    ((1, 2), 4, 1),  # Translational symmetry
    ((0, 3), 4, 1),  # Periodic symmetry
    ((0, 4), 6, 2),
    ((2, 5), 6, 3),
    ((1, 5), 6, 2)   # Translational + periodic symmetry
])
def test_periodic_distance(edge, system_size, result):
    source, target = edge
    distance = abs(target-source)
    assert periodic_distance(distance, system_size) == result


def test_complete_graph():
    edges, weights = complete_graph(4, [(0, 1), (0, 2)], [10, 20])
    assert edges == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert weights == [10, 20, 10, 10, 20, 10]
