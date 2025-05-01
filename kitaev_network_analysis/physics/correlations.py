import qutip.entropy as qt


def mutual_information(ptrace):
    """Given two particles A and B, it quantifies the amount of information
    obtained from B by observing A.

        $$ \mathcal{I}_{ij} = Tr{(\rho_{ij}\log_2 \rho_{ij}
                - \rho_{i}\log_2 \rho_{i} - \rho_{j}\log_2  \rho_{j})} $$

    Args:
        ptrace (Qutip Obj): Matrix representation of a subsystem.

    Returns:
        Float: Correlation between two particles.
    """
    return qt.entropy_mutual(ptrace, 1, 0, base=2)


def concurrence(ptrace):
    """It measures the entanglement between two particles.

        $$ C_{ij} = \max \Big(0, \sqrt{\lambda_1^{ij}} - \sqrt{\lambda_2^{ij}} - \sqrt{\lambda_3^{ij}} -\sqrt{\lambda_4^{ij}}\Big) $$

    where $\lambda^{ij}$ are the eigenvalues of the matrix $\rho_{ij}(\sigma_y \otimes \sigma_y)\rho_{ij}^*(\sigma_y \otimes \sigma_y)$.
    Args:
        ptrace (Qutip Obj): Matrix representation of a subsystem.

    Returns:
        Float: Entanglement between two particles.
    """
    return qt.concurrence(ptrace)


def coherence(ptrace):
    """Sum over all no-diagonal terms:
    \mathcal{C}_{ij} = \sum_{i \neq j}|\rho_{ij}|

    Since our matrix is symmetric we can sum over the upper triangular matrix
    and then multiply by 2.

    Args:
        ptrace (Qutip Obj): Matrix representation of a subsystem.

    Returns:
        Float: It measures if the subsystem is in superposition.
    """
    matrix = abs(ptrace.data)
    coherence = matrix.sum() - matrix.diagonal().sum()
    return coherence.real
