import numpy as np
from openfermion import get_ground_state, get_sparse_operator
from qutip.qobj import Qobj


def get_ground_matrix(hamiltonian):
    """Given the energies of the system (hamiltonian) we get a matrix
    representation of the ground_state.

    Args:
        hamiltonian (FermionOperator): Operator that represents all possible
        energies in the system.

    Returns:
        Qobj: Density matrix that represents the ground state.
    """
    ground_qutip = get_ground_eigenvector(hamiltonian)
    return ground_qutip * ground_qutip.dag()


def get_ground_eigenvector(hamiltonian):
    _, ground_vector = get_ground_state_func(hamiltonian)
    return array_to_qutip_object(ground_vector)


def get_ground_eigenvalue(hamiltonian):
    ground_value, _ = get_ground_state_func(hamiltonian)
    return ground_value


def get_ground_state_func(hamiltonian):
    return get_ground_state(get_sparse_operator(hamiltonian))


def array_to_qutip_object(array):
    qutip_obj = Qobj(array)
    qutip_obj.tidyup()
    dimensions = redefine_dimensions(qutip_obj.shape[0])
    qutip_obj.dims = dimensions
    return qutip_obj


def redefine_dimensions(shape):
    """
    Dimensions keep track of the shape of the individual components
    of a combined system.

    A single spin is a 2-level quantum system represented as a vector with shape=(2, 1) and dimension=(2, 1).
    To represent combined system we use the tensor product, so to represent a such a vector
    the shape=(16,1) and dimension=[[2,2,2,2],[1,1,1,1]]
    """
    num_sites = np.log(shape) / np.log(2)
    return [[2] * int(num_sites), [1] * int(num_sites)]
