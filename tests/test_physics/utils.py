import numpy as np
from scipy import sparse
from sympy.physics.quantum import TensorProduct
from sympy.matrices import Matrix, eye


def four_chain_hamiltonian(w, mu, delta, theta, periodicity=True):
    """Given the formula of the Hamiltonian:

    \mathcal{H} = \sum_{j}
            - \omega(\hat{a}_j^\dagger \hat{a}_{j+1}+ \hat{a}_{j+1}^\dagger \hat{a}_j)
            - \mu\big(\hat{a}_j^\dagger \hat{a}_j - \frac{1}{2}\big)
            + |\Delta|e^{i\theta}\hat{a}_j \hat{a}_{j+1}
            + |\Delta|e^{-i\theta}\hat{a}_{j+1}^\dagger \hat{a}_j^\dagger

    We have built a hamiltonian for j=4, in order to test an specific case
    for the hamiltonian from the package.
    """
    s_cond = abs(delta) * np.exp(complex(0, 1) * theta)
    hamiltonian = hopping(w) + chemical(mu) + superconductivity(s_cond)

    if periodicity:
        hamiltonian += -w * (a4_creation * a1_destruction + a1_creation * a4_destruction)
        hamiltonian += s_cond * (a4_destruction * a1_destruction)
        hamiltonian += s_cond.conjugate() * (a1_creation * a4_creation)

    hamiltonian = np.array(hamiltonian).astype(complex)
    sparse_matrix = sparse.csr_matrix(hamiltonian)
    return sparse_matrix


def hopping(w):
    return -w * (
        a1_creation * a2_destruction
        + a2_creation * a1_destruction
        + a2_creation * a3_destruction
        + a3_creation * a2_destruction
        + a3_creation * a4_destruction
        + a4_creation * a3_destruction
    )


def chemical(mu):
    return -mu * (
        a1_creation * a1_destruction
        + a2_creation * a2_destruction
        + a3_creation * a3_destruction
        + a4_creation * a4_destruction
        - 2 * identity(16)
    )


def superconductivity(constant):
    destruction = constant * (
        a1_destruction * a2_destruction + a2_destruction * a3_destruction + a3_destruction * a4_destruction
    )
    creation = constant.conjugate() * (
        a2_creation * a1_creation + a3_creation * a2_creation + a4_creation * a3_creation
    )
    return destruction + creation


def identity(dimensions=2):
    return eye(dimensions)


def sigma_z():
    return Matrix([[1, 0], [0, -1]])


def destruction_matrix():
    return Matrix([[0, 1], [0, 0]])


def creation_matrix():
    return Matrix([[0, 0], [1, 0]])


a1_creation = TensorProduct(creation_matrix(), identity(), identity(), identity())
a1_destruction = TensorProduct(destruction_matrix(), identity(), identity(), identity())

a2_creation = TensorProduct(sigma_z(), creation_matrix(), identity(), identity())
a2_destruction = TensorProduct(sigma_z(), destruction_matrix(), identity(), identity())

a3_creation = TensorProduct(sigma_z(), sigma_z(), creation_matrix(), identity())
a3_destruction = TensorProduct(sigma_z(), sigma_z(), destruction_matrix(), identity())

a4_creation = TensorProduct(sigma_z(), sigma_z(), sigma_z(), creation_matrix())
a4_destruction = TensorProduct(sigma_z(), sigma_z(), sigma_z(), destruction_matrix())
