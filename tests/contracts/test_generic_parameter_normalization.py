"""Manufactured host contracts; runnable with unittest without pytest conftest."""

import sys
import unittest

import numpy as np

from mooneural.training.generic_parameter_normalization import (
    ParameterTransformPoint,
    ReductionAxes,
    apply_metric_scale,
    composed_parameter_pullback,
    divide_jet,
    inverse_mse_scale,
    local_parameter_pullback,
    squared_error_jet,
)
from mooneural.training.generic_training_contracts import (
    ImmutableJSONMapping,
    MetricCoordinates,
    NormalizationMetadata,
    ScaleSpec,
)


def transform_point():
    raw = np.array([.2, -.3])
    first = np.exp(raw[0])
    second = raw[0]+2*raw[1]
    physical = np.array([first, second, first*second])
    jacobian = np.array([[first, 0.], [1., 2.], [first*(second+1), 2*first]])
    return ParameterTransformPoint(("u0", "u1"), ("theta0", "theta1", "theta2"), raw, physical, jacobian,
        ImmutableJSONMapping({"recipe": "manufactured exp/coupled product; no economic support"}))


def registry(factor):
    return NormalizationMetadata("manufactured.task",
        ScaleSpec("raw-v1", "raw", 1.), ScaleSpec("D-v1", "training", factor),
        ScaleSpec("selection-v1", "selection", .5), ScaleSpec("terminal-v1", "terminal", .2))


class ParameterNormalizationContractTests(unittest.TestCase):
    def test_transform_full_rectangular_jacobian_and_binding(self):
        transform = transform_point()
        partial = np.array([[[2., 3., 4.], [1., 0., -2.]]])
        actual = local_parameter_pullback(partial, transform)
        expected = np.stack([sum(partial[..., index]*transform.jacobian[index, axis] for index in range(3))
                             for axis in range(2)], axis=-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(actual.shape, (1, 2, 2))
        self.assertEqual(transform.binding_hash(), transform_point().binding_hash())
        with self.assertRaises(ValueError):
            transform.jacobian[0, 0] = 9.
        with self.assertRaises(TypeError):
            transform.recipe["recipe"] = "changed"

    def test_local_fixed_location_differs_from_moving_successor(self):
        transform = transform_point()
        first, second, third = transform.physical
        current_state, innovation = .4, -.7
        successor = second*current_state+third*innovation
        local_theta = np.array([[successor**2, successor, 1.]])
        state_partial = np.array([[2*first*successor+second]])
        state_motion = np.array([[0., current_state, innovation]])
        local = local_parameter_pullback(local_theta, transform)
        total = composed_parameter_pullback(local_theta, state_partial, state_motion, transform)
        expected_difference = (2*first*successor+second)*(current_state*transform.jacobian[1]+innovation*transform.jacobian[2])
        np.testing.assert_allclose(total-local, expected_difference[None, :], rtol=1e-13, atol=1e-14)
        self.assertFalse(np.allclose(local, total))

    def test_composed_global_current_next_direct_and_raw_transform(self):
        transform = transform_point()
        first, second, third = transform.physical
        current_state, innovation = .4, -.7
        successor = second*current_state+third*innovation
        current_theta = np.array([current_state**2, current_state, 1.])
        next_theta = np.array([successor**2, successor, 1.])+(2*first*successor+second)*np.array([0., current_state, innovation])
        arguments = np.stack([current_theta, next_theta])
        actual = composed_parameter_pullback(np.array([[2*first, 0., 0.]]), np.array([[1., 3.]]), arguments, transform)
        first_raw, second_raw, third_raw = transform.jacobian
        expected = (2*first*first_raw+current_state**2*first_raw+current_state*second_raw+third_raw
            +3*(successor**2*first_raw+successor*second_raw+third_raw
                 +(2*first*successor+second)*(current_state*second_raw+innovation*third_raw)))
        np.testing.assert_allclose(actual[0], expected, rtol=1e-13, atol=1e-14)

    def test_derivative_shape_and_finiteness_refusals(self):
        transform = transform_point()
        with self.assertRaises(ValueError):
            local_parameter_pullback(np.ones((2, 2)), transform)
        with self.assertRaises(ValueError):
            local_parameter_pullback(np.full((2, 3), np.nan), transform)
        with self.assertRaises(ValueError):
            composed_parameter_pullback(np.ones((2, 1, 3)), np.ones((1, 1, 2)), np.ones((2, 2, 3)), transform)
        with self.assertRaises(ValueError):
            ParameterTransformPoint(("u",), ("theta",), np.ones(1), np.ones(1), np.ones((1, 2)), {"recipe": "test"})

    def test_parameter_dependent_output_normalization(self):
        raw_parameter = .3
        value = np.array([[1+raw_parameter**2, 2+raw_parameter]])
        derivative = np.array([[[2*raw_parameter], [1.]]])
        denominator = np.array([[2+raw_parameter, np.exp(raw_parameter)]])
        denominator_partial = np.array([[[1.], [np.exp(raw_parameter)]]])
        normalized, partial = divide_jet(value, derivative, denominator, denominator_partial)
        expected = np.array([[((2*raw_parameter)*(2+raw_parameter)-(1+raw_parameter**2))/(2+raw_parameter)**2,
                              -(1+raw_parameter)*np.exp(-raw_parameter)]])
        np.testing.assert_allclose(partial[..., 0], expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(normalized, value/denominator, rtol=1e-14, atol=1e-14)
        self.assertFalse(np.allclose(partial, derivative/denominator[..., None]))
        with self.assertRaises(ValueError):
            divide_jet(value, derivative, np.array(2.), denominator_partial)
        with self.assertRaises(ValueError):
            divide_jet(value, derivative, -denominator, denominator_partial)

    def test_explicit_axis_mean_vs_sum_and_derivative_axis(self):
        residual = (np.arange(24, dtype=np.float64).reshape(2, 3, 4)-10)/5
        partial = np.stack((np.ones_like(residual), residual*3), axis=-1)
        names = ("row", "output", "component")
        summed = squared_error_jet(residual, partial, ReductionAxes(names, ("row", "output"), ("component",)))
        averaged = squared_error_jet(residual, partial, ReductionAxes(names, names, ()))
        self.assertAlmostEqual(summed.value, 4*averaged.value)
        np.testing.assert_allclose(summed.derivative, 4*averaged.derivative, rtol=1e-14, atol=1e-14)
        expected = sum(residual[row, output, component]**2 for row in range(2) for output in range(3)
                       for component in range(4))/6
        self.assertAlmostEqual(summed.value, expected)
        self.assertEqual(summed.derivative.shape, (2,))
        np.testing.assert_allclose(summed.derivative, [2*np.sum(residual)/6, 6*summed.value], rtol=1e-14, atol=1e-14)

    def test_reductions_require_every_axis_once_and_nonempty_tensors(self):
        for mean_axes, sum_axes in ((("row",), ()), (("row", "component"), ("component",)), (("row", "bad"), ())):
            with self.assertRaises(ValueError):
                ReductionAxes(("row", "component"), mean_axes, sum_axes)
        reduction = ReductionAxes(("row", "component"), ("row",), ("component",))
        with self.assertRaises(ValueError):
            squared_error_jet(np.zeros((0, 3)), np.zeros((0, 3, 2)), reduction)
        with self.assertRaises(ValueError):
            squared_error_jet(np.ones((2, 3)), np.ones((2, 3)), reduction)

    def test_mse_denominator_quotient_role_coordinates_and_once_only(self):
        parameter = .5
        error = np.array([[1+parameter, 2-parameter]])
        partial = np.array([[[1.], [-1.]]])
        metric = squared_error_jet(error, partial, ReductionAxes(("row", "output"), ("row", "output"), ()))
        denominator = np.array(2+parameter**2)
        factor, factor_partial = inverse_mse_scale(denominator, np.array([2*parameter]))
        metadata = registry(factor)
        normalized = apply_metric_scale(metric, metadata, "training", factor_partial)
        self.assertIsInstance(normalized.coordinates, MetricCoordinates)
        self.assertEqual(normalized.coordinates.training, metric.value/denominator)
        expected = (metric.derivative*denominator-metric.value*2*parameter)/denominator**2
        np.testing.assert_allclose(normalized.derivative, expected, rtol=1e-14, atol=1e-14)
        terminal = apply_metric_scale(metric, metadata, "terminal", np.zeros(1))
        self.assertEqual(terminal.coordinates.terminal, .2*metric.value)
        self.assertNotEqual(terminal.coordinates.terminal, .2*normalized.coordinates.training)
        with self.assertRaises(TypeError):
            apply_metric_scale(normalized, metadata, "terminal", np.zeros(1))
        with self.assertRaises(ValueError):
            metadata.validate_application("training", ("D-v1", "D-v1"))

    def test_constant_denominator_and_role_derivative_refusals(self):
        factor, partial = inverse_mse_scale(np.array(4.), np.zeros(2))
        self.assertEqual(factor, .25)
        np.testing.assert_array_equal(partial, np.zeros(2))
        metric = squared_error_jet(np.ones((2,)), np.ones((2, 2)), ReductionAxes(("row",), ("row",), ()))
        with self.assertRaises(ValueError):
            apply_metric_scale(metric, registry(factor), "training", np.zeros(1))
        with self.assertRaises(ValueError):
            inverse_mse_scale(np.array(0.), np.zeros(2))

    def test_imports_do_not_load_economic_model_or_tensorflow(self):
        # A full suite may already have imported TensorFlow in another test.
        # Test this module's import behavior in a genuinely fresh interpreter.
        import subprocess
        code = ("import sys; import mooneural.training.generic_parameter_normalization; "
                "assert not any(n.split('.')[0] in ('tensorflow', 'dsge_hmc', 'common_utils') for n in sys.modules)")
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
