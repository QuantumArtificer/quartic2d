import numpy as np
import pytest

from quartic2d._numerics import Interpolator1D, composite_gauss_nodes_weights


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip"])
def test_interpolators_support_complex_data_and_zero_exterior(method):
    x = np.linspace(0.0, 2.0, 21)
    y = np.exp(1j * x)
    f = Interpolator1D(x, y, method)

    values = f(np.array([0.3, 1.1]))
    assert values.shape == (2,)
    assert np.iscomplexobj(values)
    assert f(-1.0) == 0.0
    assert f(3.0) == 0.0


def test_composite_gauss_nodes_are_inside_intervals():
    x = np.array([0.0, 0.5, 2.0])
    for order in (4, 8):
        nodes, weights = composite_gauss_nodes_weights(x, order)
        assert nodes.size == (x.size - 1) * order
        assert weights.size == nodes.size
        assert np.all(nodes > x[0])
        assert np.all(nodes < x[-1])
        assert np.all(weights > 0.0)


def test_not_a_knot_cubic_is_available_for_q_space_boundary_conditions():
    x = np.linspace(0.0, 7.0, 39)
    y = np.exp(-x * x / 4.0)
    probe = np.linspace(0.0, 0.5, 101)
    exact = np.exp(-probe * probe / 4.0)

    natural = Interpolator1D(x, y, "cubic")
    q_space = Interpolator1D(x, y, "cubic", cubic_bc_type="not-a-knot")

    natural_error = np.linalg.norm(natural(probe) - exact)
    q_space_error = np.linalg.norm(q_space(probe) - exact)
    assert q_space_error < natural_error
