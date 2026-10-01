"""Independent least-squares reference and raw derivative design checks."""

import numpy as np
import pytest
from scipy.linalg import lstsq
from tests.contracts.test_generic_policy_coordinates import network, profile

from mooneural.training.generic_derivative_warm_start import make_derivative_warm_start
from mooneural.training.generic_policy_coordinates import TanhCoordinateMap


def test_svd_fit_and_design_match_independent_value_and_derivative_losses():
    coordinates = profile()
    mapping = TanhCoordinateMap(coordinates, 4)
    rng = np.random.default_rng(331)
    parameters = rng.normal(size=mapping.parameter_dim) * .2
    inputs = rng.normal(size=(20, 3))
    targets, derivatives = rng.normal(size=(20, 2)), rng.normal(size=(20, 2, 3))
    scales = np.array([.25, 4., 2.])
    fit = make_derivative_warm_start(coordinates, 4, 20, 1)
    result = fit(parameters, inputs, targets, derivatives, scales)
    fitted, design, target, coefficients, _singular, normal, valid = [item.numpy() for item in result]
    assert valid and normal < 1e-10
    expected, _residuals, _rank, _singular = lstsq(design, target, cond=1e-12, lapack_driver="gelsd")
    np.testing.assert_allclose(coefficients, expected, rtol=1e-8, atol=1e-9)
    raw = mapping.to_raw(fitted).numpy()
    values = network(raw, inputs)
    jacobian = np.stack([(network(raw, inputs + 1e-6 * basis) - network(raw, inputs - 1e-6 * basis)) / 2e-6 for basis in np.eye(3)], axis=-1)
    loss = np.mean((values - targets)**2) / scales[0] + np.mean((jacobian[:, :, :1] - derivatives[:, :, :1])**2) / scales[1] + np.mean((jacobian[:, :, 1:] - derivatives[:, :, 1:])**2) / scales[2]
    np.testing.assert_allclose(np.sum((design @ coefficients - target)**2), loss, rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(fitted[:sum(mapping.sizes[:4])], parameters[:sum(mapping.sizes[:4])])


def test_exactly_representable_value_and_derivative_targets_recover_predictions():
    coordinates = profile()
    mapping = TanhCoordinateMap(coordinates, 4)
    rng = np.random.default_rng(119)
    parameters = rng.normal(size=mapping.parameter_dim) * .2
    inputs = rng.normal(size=(40, 3)) * .1 + coordinates.input_center
    raw = mapping.to_raw(parameters).numpy()
    targets = network(raw, inputs)
    derivatives = np.stack([(network(raw, inputs + 1e-7 * basis) - network(raw, inputs - 1e-7 * basis)) / 2e-7 for basis in np.eye(3)], axis=-1)
    result = make_derivative_warm_start(coordinates, 4, 40, 1)(parameters, inputs, targets, derivatives, np.ones(3))
    assert bool(result[-1])
    np.testing.assert_allclose(network(mapping.to_raw(result[0]).numpy(), inputs), targets, atol=2e-9, rtol=1e-7)


@pytest.mark.parametrize("count,split", [(0, 1), (3, 0), (3, 3), (True, 1)])
def test_invalid_construction_shapes_refused(count, split):
    with pytest.raises(ValueError):
        make_derivative_warm_start(profile(), 4, count, split)
