import numpy as np
import pytest
from qutip import Qobj, tensor, sigmay
from scipy.linalg import eig

from phase_transition_cna.physics import coherence, concurrence, mutual_information


@pytest.fixture
def partial_trace():
    return Qobj(
        np.array([[0.021, 0, 0, 0.074], [0, 0.257, 0.346, 0], [0, 0.346, 0.465, 0], [0.074, 0, 0, 0.257]]),
        dims=[[2, 2], [2, 2]],
    )


def test_mutual_information(partial_trace):
    # Get the eigenvalues.
    p_ij, _ = eig(partial_trace)
    p_i, _ = eig(partial_trace.ptrace(0))
    p_j, _ = eig(partial_trace.ptrace(1))

    # Computing the trace with the eigenvalues.
    trace_ij = np.sum(p_ij * np.log2(p_ij))
    trace_i = np.sum(p_i * np.log2(p_i))
    trace_j = np.sum(p_j * np.log2(p_j))

    # Trace is a linear function: Tr(A+B)=Tr(A)+Tr(B)
    result = np.real(trace_ij - trace_i - trace_j)  # Remove small imaginary errors produce by computer precision.
    assert mutual_information(partial_trace) == pytest.approx(result, 1e-9)


def test_concurrence(partial_trace):
    matrix = partial_trace * spinflip_op() * partial_trace.conj() * spinflip_op()
    eigenvalues = sorted(eig(matrix)[0], reverse=True)
    assert concurrence(partial_trace) == max(0, np.sqrt(eigenvalues[0]) - np.sum(np.sqrt(eigenvalues[1:])))


def test_coherence(partial_trace):
    assert coherence(partial_trace) == pytest.approx(2 * (0.346 + 0.074), 1e-9)


def spinflip_op():
    return tensor(sigmay(), sigmay())
