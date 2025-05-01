import numpy as np
from openfermion.ops import FermionOperator
from openfermion import number_operator


def spinless_kitaev_chain(n_sites, hopping, chemical, delta, theta=0, periodicity=True):
    """
    Create a chain of spinless two-level-systems.

    This function represents the following hamiltonian:
    \mathcal{H} = \sum_{j}
                - \omega(\hat{a}_j^\dagger \hat{a}_{j+1}+ \hat{a}_{j+1}^\dagger \hat{a}_j)
                - \mu\big(\hat{a}_j^\dagger \hat{a}_j - \frac{1}{2}\big)
                + |\Delta|e^{i\theta}\hat{a}_j \hat{a}_{j+1}
                + |\Delta|e^{-i\theta}\hat{a}_{j+1}^\dagger \hat{a}_j^\dagger

    Args:
        n_sites (int): Number of sites availables in your chain.
        hopping (float): Energy required to jump to the nearest site.
        chemical (float): Energy required to create a particle.
        delta (float): Energy required to create a Cooper Pair.
        theta (float): Superconducting phase between 0 <= theta < 2pi:
        periodicity (bool, optional): Close chain (True) Open chain (False). Defaults to True.

    Returns:
        FermionOperator: Hamiltonian representation of the Kitaev-chain.
    """
    # Shifting indexing from (0, n-1) to (1, n).
    # Reason: FermionOperator((1, 0)) sparse matrix is larger than FermionOperator((0, 0)).
    first_site = 0
    last_site = n_sites - 1

    hamiltonian = FermionOperator()
    for site in range(first_site, last_site):
        hamiltonian += hopping_term(site, -hopping)
        hamiltonian += chemical_term(last_site, site, -chemical)
        hamiltonian += superconducting_gap_term(site, delta, theta)

    hamiltonian += chemical_term(last_site, last_site, -chemical)
    if periodicity:
        hamiltonian += FermionOperator(((last_site, 1), (first_site, 0)), -hopping)
        hamiltonian += FermionOperator(((first_site, 1), (last_site, 0)), -hopping)

        hamiltonian += FermionOperator(((last_site, 0), (first_site, 0)), abs(delta) * np.exp(theta * complex(0, 1)))
        hamiltonian += FermionOperator(((first_site, 1), (last_site, 1)), abs(delta) * np.exp(theta * complex(0, -1)))
    return hamiltonian


def hopping_term(site, coefficient):
    hopping_term = FermionOperator(((site, 1), (site + 1, 0)), coefficient)
    hopping_term += FermionOperator(((site + 1, 1), (site, 0)), coefficient.conjugate())
    return hopping_term


def chemical_term(n_site, site, coefficient):
    chemical_term = number_operator(n_site, site, coefficient)
    chemical_term -= FermionOperator((), coefficient * 0.5)
    return chemical_term


def superconducting_gap_term(site, delta_coefficient, theta_coefficient):
    constant = abs(delta_coefficient) * np.exp(theta_coefficient * complex(0, 1))
    superconducting_term = FermionOperator(((site, 0), (site + 1, 0)), constant)
    superconducting_term += FermionOperator(((site + 1, 1), (site, 1)), constant.conjugate())
    return superconducting_term
