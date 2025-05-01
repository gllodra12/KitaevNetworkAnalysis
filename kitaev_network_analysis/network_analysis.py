import numpy as np

import argparse
import pathlib

from kitaev_network_analysis.physics.correlations import coherence, concurrence, mutual_information
from kitaev_network_analysis.physics.hamiltonian import spinless_kitaev_chain
from kitaev_network_analysis.physics.partial_trace import partial_traces
from kitaev_network_analysis.network import complete_graph, metrics
from kitaev_network_analysis.fidelity import get_chemical_range

REPOSITORY_ROOT = pathlib.Path.cwd()
STORE_DATA = REPOSITORY_ROOT / "data" / "network_analysis"

METRICS_MAP = {
    "density": metrics.density,
    "disparity": metrics.disparity,
    "betweeness": metrics.betweeness_centrality,
    "clustering": metrics.average_clustering,
    "shortest_path": metrics.average_shortest_path,
}
CORRELATIONS_MAP = {"mutual": mutual_information, "coherence": coherence, "concurrence": concurrence}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metric", help="Available metrics: {}".format(list(METRICS_MAP.keys())), choices=METRICS_MAP.keys()
    )
    parser.add_argument(
        "correlation",
        help="Available correlations: {}".format(list(CORRELATIONS_MAP.keys())),
        choices=CORRELATIONS_MAP.keys(),
    )
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
    parser.add_argument("-t", "--theta", help="Superconducting phase between 0 <= theta < 2pi", type=float, default=0)
    parser.add_argument("-pT", "--periodicity_True", help="Close chain", action="store_true", dest="periodicity")
    parser.add_argument("-pF", "--periodicity_False", help="Open chain", action="store_false", dest="periodicity")
    parser.add_argument(
        "-s",
        "--steps",
        help="Choose a float. Distance between points in the $\mu$ critical region",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "-u",
        "--chemical_potentials",
        help="This optional parameter is useful when you want to increase the resolution of certain regions of the phase transition by selecting several mu values.",
        nargs="+",
        type=float,
    )
    args = parser.parse_args()
    return args


def phase_transition(metric, correlation, n_sites, hopping, delta, theta, periodicity, steps):
    """Measures the same metric for different values of μ, providing
    the necessary information to see if the metric detects the phase transition.

    See the function measure() for more information about the parameters.
    """
    chemical_potentials = get_chemical_range(hopping, steps, periodicity)
    results = list()
    for mu in chemical_potentials:
        results.append(measure(metric, correlation, n_sites, hopping, mu, delta, theta, periodicity))
    return chemical_potentials, results


def measure(metric, correlation, n_sites, hopping, chemical, delta, theta, periodicity):
    """Given a hamiltonian configuration and a correlation function,
    it builds a network and measures a certain metric.

    Args:
        metric (func): Complex network metric (Ex: density, clustering, ...)
        correlation (func): Relation between particles (Ex: Mutual information)
        n_sites (int): Number of sites in the chain.
        hopping (float): Energy required to jump to the nearest site
        chemical (float): Energy required to create a particle.
        delta (float): Energy required to create a Cooper pair.

    Returns:
        float: Outputs the value of the metric.
    """
    hamiltonian = spinless_kitaev_chain(n_sites, hopping, chemical, delta, theta, periodicity)

    # Partial trace is calculated using the ground state of the hamiltonian
    ptraces = partial_traces(hamiltonian, periodicity)
    edges, weights = graph_elements(ptraces, correlation)

    if periodicity:
        edges, weights = complete_graph(n_sites, edges, weights)

    return metric(edges, weights)


def graph_elements(ptraces, correlation):
    """Compute the relation (weight) of two particles
    given a correlation function.

    The information of each particle is codified in the partial trace.

    Args:
        ptraces (dict): Key: Edges (source, target). Value: partial trace
        correlation (func): Relation between particles.

    Returns:
        (list, list): Elements of the graph: edges and weights
    """
    edges, weights = list(), list()
    for edge, ptrace in ptraces.items():
        edges.append(edge)
        weights.append(correlation(ptrace))
    return edges, weights


def save_values(x, y, hamiltonian, metric, correlation):
    path = STORE_DATA / correlation / metric / hamiltonian
    print("path", path)
    if not path.exists():
        path.mkdir(parents=True)

    np.save(path / "x_values", np.array(x))
    np.save(path / "y_values", np.array(y))


if __name__ == "__main__":
    args = parse_args()

    # Variables
    metric = METRICS_MAP[args.metric]
    correlation = CORRELATIONS_MAP[args.correlation]
    n_sites = args.n_sites
    hopping = args.hopping
    delta = args.delta
    theta = args.theta
    periodicity = args.periodicity
    steps = args.steps
    chemical_potentials = args.chemical_potentials

    # By default theta is 0
    parameters = "_".join(
        map(
            str,
            (
                args.n_sites,
                args.hopping,
                args.delta,
                args.theta,
                args.periodicity,
                args.steps,
                args.metric,
                args.correlation,
            ),
        )
    )

    # Phase transition
    if chemical_potentials is None:
        chemical_potentials = get_chemical_range(hopping, steps, periodicity)

    results = list()
    for mu in chemical_potentials:
        potential = "mu={number:.{digits}f}".format(number=mu, digits=3)

        # Build your network:
        hamiltonian = spinless_kitaev_chain(n_sites, hopping, mu, delta, theta, periodicity)
        ptraces = partial_traces(hamiltonian, periodicity)
        edges, weights = graph_elements(ptraces, correlation)
        if periodicity:
            edges, weights = complete_graph(n_sites, edges, weights)

        # Measure a network metric:
        measurement = metric(edges, weights)
        results.append(measurement)

    # Save phase transition results
    mu_range = "mu={0:.1f},{1:.1f}_".format(chemical_potentials[0], chemical_potentials[-1])
    h_config = mu_range + parameters
    save_values(chemical_potentials, results, h_config, args.metric, args.correlation)
