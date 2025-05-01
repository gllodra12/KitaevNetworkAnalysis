import pytest
import numpy as np

from phase_transition_cna.physics import spinless_kitaev_chain
from phase_transition_cna.physics import partial_traces


@pytest.mark.parametrize(
    "n_sites, periodicity, number_ptraces", [(4, True, 2), (4, False, 6), (6, True, 3), (6, False, 15)]
)
def test_number_ptraces(n_sites, periodicity, number_ptraces):
    hamiltonian = spinless_kitaev_chain(n_sites, 1, 1, 0.5, 0, periodicity)
    ptraces = partial_traces(hamiltonian, periodicity)
    assert len(ptraces) == number_ptraces


@pytest.mark.parametrize(
    "periodicity, pair, result",
    [
        (
            True,
            (0, 1),
            np.array([[0.167, 0, 0, 0.289], [0, 0.167, 0.167, 0], [0, 0.167, 0.167, 0], [0.289, 0, 0, 0.5]]),
        ),
        (
            False,
            (0, 1),
            np.array([[0.021, 0, 0, 0.074], [0, 0.257, 0.346, 0], [0, 0.346, 0.465, 0], [0.074, 0, 0, 0.257]]),
        ),
        (
            False,
            (0, 2),
            np.array([[0.021, 0, 0, 0.099], [0, 0.257, 0.257, 0], [0, 0.257, 0.257, 0], [0.099, 0, 0, 0.465]]),
        ),
        (
            False,
            (1, 2),
            np.array([[0.021, 0, 0, 0.074], [0, 0.465, 0.346, 0], [0, 0.346, 0.257, 0], [0.074, 0, 0, 0.257]]),
        ),
    ],
)
def test_ptrace(periodicity, pair, result):
    """The hamiltonian for: spinless_kitaev_chain(3, 1, 1, 0.5, 0, True)
    H = 0.167*(|001><001|+|010><001|+|100><001|+
               |001><010|+|010><010|+|100><010|+
               |001><100|+|010><100|+|100><100|)
      + 0.289*(|111><001|+|111><010|+|111><100|+
               |001><111|+|010><111|+|100><111|)
      + 0.5  *(|111><111|)

    |abc> = |001> --> site 0 is 0, site 1 is 0, site 2 is 1.

    We can trace out all the sites except the pair we interested using the following formula:

        $$ rho_01 = Trace_over_2(rho_012) = sum_{ijklmn}(|a_i b_j><a_k b_l|*<c_m|c_n>) $$

    So the expected result would be:

    H = 0.167*(|00><00|+|01><01|+|10><01|+|01><10|+|10><10|)
      + 0.289*(|11><00|+|00><11|)
      + 0.5  *(|11><11|)

    For rho_12 the formula would be:
        $$ rho_12 = Trace_over_0(rho_012) = sum_{ijklmn}(|b_i c_j><b_k c_l|*<a_m|a_n>) $$

    The hamiltonian for: spinless_kitaev_chain(3, 1, 1, 0.5, 0, False) is:
    H = 0.021|000><000|
      + 0.074(|011><000|+|110><000|+|000><110|+|000><011|)
      + 0.099(|000><101|+|101><000|)
      + 0.257(|011><011|+|110><011|+|011><110|+|110><110|)
      + 0.346(|101><011|+|011><101|+|110><101|+|101><110|)
      + 0.465|101><101|
    """
    hamiltonian = spinless_kitaev_chain(3, 1, 1, 0.5, 0, periodicity)
    ptrace = partial_traces(hamiltonian, periodicity)[pair].data
    assert (np.around(ptrace.todense(), 3) == result).all()
