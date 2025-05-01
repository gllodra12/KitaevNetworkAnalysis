import numpy as np

import argparse
import pathlib

from kitaev_network_analysis.physics.hamiltonian import spinless_kitaev_chain
from kitaev_network_analysis.physics.ground_state import get_ground_eigenvector


REPOSITORY_ROOT = pathlib.Path.cwd()
STORE_DATA = REPOSITORY_ROOT / "data" / "fidelity"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_sites", help="Choose an integer. Number of sites of your chain.", type=int)
    parser.add_argument(
        "-w",
        "--hopping",
        help="Choose a float. Energy required to jump to the nearest site (default 1)",
        type=float,
        default=1,
    )
    parser.add_argument(
        "-d",
        "--delta",
        help="Choose a float. Energy required to create a Cooper Pair (default 0.5)",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-t",
        "--theta",
        help="Choose a number in radians. Superconductivity parameter (default 0)",
        type=float,
        default=0,
    )
    parser.add_argument("-pT", "--periodicity_True", help="Close chain", action="store_true", dest="periodicity")
    parser.add_argument("-pF", "--periodicity_False", help="Open chain", action="store_false", dest="periodicity")
    parser.add_argument(
        "-s",
        "--step_distance",
        help="Choose a float. Distance between points in the μ critical region",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()
    return args


def ground_state_transition(n_sites, hopping, delta, theta, periodicity, step_distance):
    """Computes the fidelity between a ground state at μ=x and the following
    ground states at μ=x+1, x+2, etc.

    Fidelity quantifies between 0 and 1, the similarity between both ground states.
    Fidelity = 1 --> both ground states are identical.
    Fidelity = 0 --> both ground states are orthogonal.

    Args:
        n_sites (int): Number of sites in the chain.
        hopping (float): Energy required to jump to the nearest site.
        delta (float): Energy required to create a Cooper Pair.
        theta (float]): Superconducting phase between 0 <= theta < 2pi
        periodicity (bool): Close chain (True) Open chain (False). Defaults to True.
        step_distance (float): Distance between point in the μ critical region

    Returns:
        (list, list): Values of μ. Fidelity between the first value of μ at the μ at this idx in the list.
    """
    # y-values
    fidelities = list()

    # x-values
    chemical_potentials = get_chemical_range(hopping, step_distance, periodicity)

    # Computing fidelities
    mu = chemical_potentials[0] if periodicity else chemical_potentials[len(chemical_potentials) // 2]
    hamiltonian = spinless_kitaev_chain(n_sites, hopping, mu, delta, theta, periodicity)
    first_ground_state = get_ground_eigenvector(hamiltonian)

    for mu in chemical_potentials:
        hamiltonian = spinless_kitaev_chain(n_sites, hopping, mu, delta, theta, periodicity)
        next_ground_state = get_ground_eigenvector(hamiltonian)
        inner_product = first_ground_state.overlap(next_ground_state)
        overlap = np.linalg.norm(inner_product)
        fidelities.append(overlap)

    return chemical_potentials, fidelities


def get_chemical_range(hopping, step_distance, periodicity):
    if periodicity:
        min_mu = 2 * hopping - 1
        max_mu = 2 * hopping + 1
        critical_mu = 2 * hopping
        window_length = 0.7

        lower_range = np.linspace(min_mu, min_mu + window_length, int(10 * window_length), endpoint=False)
        upper_range = np.linspace(max_mu - window_length, max_mu, int(10 * window_length))
        critical_zone = np.arange(min_mu + window_length, max_mu - window_length, step_distance)

        # Delete critical mu.
        if critical_mu in critical_zone:
            index = np.argwhere(critical_zone == critical_mu)
            critical_zone = np.delete(critical_zone, index)

        mu_values = np.r_[lower_range, critical_zone, upper_range]

    # μ values for chain with open boundary conditions.
    else:
        max_mu = 2 * hopping + 1
        min_mu = -max_mu
        steps = 2 * int(max_mu / step_distance)

        mu_values = np.linspace(min_mu, max_mu, steps)
        mu_values = np.r_[mu_values[: len(mu_values) // 2], 0, mu_values[len(mu_values) // 2 :]]

    return mu_values


def save_values(x_values, y_values):
    args = parse_args()
    parameters = map(str, (args.n_sites, args.hopping, args.delta, args.theta, args.periodicity, args.step_distance))
    foldername = "_".join(parameters)

    path = STORE_DATA / foldername
    print("path", path)
    if not path.exists():
        path.mkdir()

    np.save(path / "x_values", np.array(x_values))
    np.save(path / "y_values", np.array(y_values))


if __name__ == "__main__":
    args = parse_args()
    chemical_potentials, fidelities = ground_state_transition(
        args.n_sites, args.hopping, args.delta, args.theta, args.periodicity, args.step_distance
    )
    save_values(chemical_potentials, fidelities)
