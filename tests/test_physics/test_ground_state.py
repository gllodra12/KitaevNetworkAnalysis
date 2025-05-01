import pytest

import numpy as np
from openfermion import get_sparse_operator
from scipy.sparse.linalg import eigsh

from phase_transition_cna.physics.ground_state import get_ground_matrix, get_ground_state_func
from phase_transition_cna.physics import spinless_kitaev_chain


@pytest.mark.parametrize("hopping, chemical, delta", [(1, 1, 0.5), (3, 1, 0.5)])  # Topological phase.  # Trivial phase
def test_ground_state(hopping, chemical, delta):
    """A ground state is a vector that represents the state of minimum
    energy of the system.

    Two grounds state will have the same physical meaning if they are equal
    or proportional (since both vector will point to the same direction).

    To check if both vectors are proportional we use two functions:
        - are_proportional([0, 1, 2, 0], [0, 2, 4, 0]) --> True
        - equal_number_of_zeros([0, 1, 2, 0], [0, 2, 4, 0]) --> True

    Finally, quantum state must be unitary.
    """
    hamiltonian = spinless_kitaev_chain(4, hopping, chemical, delta)
    _, ground_state = get_ground_state_func(hamiltonian)

    expected_energies, expected_states = get_eigenvalues_eigenvectors(hamiltonian)
    idx_min_energy = np.argmin(expected_energies)
    expected_gs = expected_states[:, idx_min_energy]

    assert are_proportional(ground_state, expected_gs)
    assert equal_number_of_zeros(ground_state, expected_gs)
    assert is_unitary(ground_state)


@pytest.mark.parametrize("hopping, chemical, delta", [(1, 1, 0.5), (3, 1, 0.5)])  # Topological phase.  # Trivial phase
def test_ground_value(hopping, chemical, delta):
    """Checking if our package gets the same ground_value (lowest eigenvalue)
    as Scipy package.
    """
    hamiltonian = spinless_kitaev_chain(4, hopping, chemical, delta)
    energy_gs, _ = get_ground_state_func(hamiltonian)

    expected_energies, _ = get_eigenvalues_eigenvectors(hamiltonian)
    expected_energy_gs = np.min(expected_energies)
    assert energy_gs == pytest.approx(expected_energy_gs, 1e-6)


@pytest.mark.parametrize("num_sites", [4, 6])
def test_shape(num_sites):
    """For a 2-level system the matrix should grow as 2^n (n is the size of the chain)"""
    hamiltonian = spinless_kitaev_chain(num_sites, 1, 1, 0.5)
    assert get_ground_matrix(hamiltonian).shape == (2**num_sites, 2**num_sites)


@pytest.mark.parametrize("num_sites", [4, 6])
def test_dimensions(num_sites):
    """Dimensions represent the shape of the individual components of the combined system.
    We expected that the density matrix of each individual component is [2, 2], so if we have
    4 individual components we expect a dimension = [[2,2,2,2],[2,2,2,2]].
    """
    hamiltonian = spinless_kitaev_chain(num_sites, 1, 1, 0.5)
    assert get_ground_matrix(hamiltonian).dims == [[2] * num_sites, [2] * num_sites]


def are_proportional(x1, x2, tol=1e-6):
    x1, x2 = np.around(x1, 9), np.around(x2, 9)
    non_zero_values_x1 = x1[x1 != 0]
    non_zero_values_x2 = x2[x2 != 0]
    proportion = non_zero_values_x1 / non_zero_values_x2
    return abs(np.max(proportion) - np.min(proportion)) < tol


def equal_number_of_zeros(x1, x2):
    x1, x2 = np.around(x1, 9), np.around(x2, 9)
    return len(x1[x1 == 0]) == len(x2[x2 == 0])


def is_unitary(x):
    return np.around(np.linalg.norm(x), 9) == 1.0


def get_eigenvalues_eigenvectors(hamiltonian):
    eigenvalues, eigenvectors = eigsh(get_sparse_operator(hamiltonian), which="SA")
    return eigenvalues, eigenvectors
