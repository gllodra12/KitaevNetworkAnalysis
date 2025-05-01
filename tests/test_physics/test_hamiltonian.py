from openfermion import get_sparse_operator
import pytest

from phase_transition_cna.physics import spinless_kitaev_chain
from utils import four_chain_hamiltonian


@pytest.mark.parametrize("hopping, chemical, delta", [(1, 1, 0.5), (3, 1, 0.5)])  # Topological phase.  # Trivial phase
def test_hamiltonian_with_periodicity(hopping, chemical, delta):
    """Check if spinless_kitaev_chain behaves as expected
    for a system with 4 sites with periodic boundary conditions.
    """
    hamiltonian = spinless_kitaev_chain(4, hopping, chemical, delta)
    result = get_sparse_operator(hamiltonian).todense()
    expected = four_chain_hamiltonian(hopping, chemical, delta, 0).todense()
    assert result == pytest.approx(expected, 1e-3)


@pytest.mark.parametrize("hopping, chemical, delta", [(1, 1, 0.5), (3, 1, 0.5)])  # Topological phase.  # Trivial phase
def test_hamiltonian_without_periodicity(hopping, chemical, delta):
    """Check if spinless_kitaev_chain behaves as expected
    for a system with 4 sites without periodic boundary conditions.
    """
    # Parameters
    theta = 0
    periodicity = False

    hamiltonian = spinless_kitaev_chain(4, hopping, chemical, delta, theta, periodicity)
    result = get_sparse_operator(hamiltonian).todense()
    expected = four_chain_hamiltonian(hopping, chemical, delta, theta, periodicity).todense()
    assert result == pytest.approx(expected, 1e-3)
