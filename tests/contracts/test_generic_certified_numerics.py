"""Independent convex references and actual shared-transaction regressions."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from scipy.optimize import brentq, minimize, nnls

from mooneural.training.generic_certified_numerics import (
    certified_cagrad_impl,
    make_certified_method_function,
    make_certified_projected_adam_function,
    make_certified_projection_function,
    numerical_binding,
)
from mooneural.training.generic_execution_boundary import jacobi_schedule, subset_masks
from mooneural.training.generic_permanent_pass_executor import PermanentPassExecutor
from tests.support.fixtures import (
    fake_control,
    fixture,
    roles,
)
from mooneural.training.generic_training_contracts import CheckpointState, canonical_json

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/generic_numerical_repair"


@contextmanager
def fixture_arrays(prefix):
    path = FIXTURES / "operands.npz"
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        yield {key.removeprefix(prefix): arrays[key] for key in arrays.files if key.startswith(prefix)}


def projection(direction, rows, analytic=None):
    direction, rows = np.asarray(direction, dtype=float), np.asarray(rows, dtype=float)
    function = make_certified_projection_function(direction.size, rows.shape[0])
    result = function(direction, rows, subset_masks(rows.shape[0]), jacobi_schedule(), tf.constant(1e-12, tf.float64))
    assert bool(result[-1]), [item.numpy() for item in result]
    normalized = rows / np.where(np.linalg.norm(rows, axis=1, keepdims=True) > 0,
                                 np.linalg.norm(rows, axis=1, keepdims=True), 1)
    if analytic is not None:
        reference = np.asarray(analytic)
    elif rows.shape[0]:
        multiplier, _residual = nnls(normalized.T, -direction, maxiter=10000)
        reference = direction + normalized.T @ multiplier
    else:
        reference = direction
    selected = result[0].numpy()
    np.testing.assert_allclose(selected, reference, rtol=2e-9, atol=2e-10)
    assert np.min(normalized @ selected, initial=0) >= -1e-10
    assert np.linalg.norm(selected - direction) <= np.linalg.norm(direction) + 1e-10
    np.testing.assert_allclose(selected @ (selected - direction), 0, atol=2e-9 * (1 + direction @ direction))
    assert float(result[2]) <= 1e-12
    return selected


@pytest.mark.parametrize("case", range(3))
def test_captured_projection_and_both_adam_projections(case):
    with fixture_arrays(f"sgu{case}_") as saved:
        direction, rows = saved["adam_input_4"], saved["adam_input_5"]
        selected = projection(direction, rows)
        if case == 2:
            assert np.linalg.norm(selected) == pytest.approx(27.120119139, rel=1e-8)
            assert np.linalg.norm(saved["projection_0_0"]) == 0
        update = make_certified_projected_adam_function(direction.size, rows.shape[0])
        result = update(*[saved[f"adam_input_{index}"] for index in range(7)])
        assert bool(result[-1])
        assert int(result[3]) == int(saved["adam_input_3"]) + 1
        np.testing.assert_allclose(result[4], selected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(-result[6], projection(-result[5].numpy(), rows), rtol=1e-9, atol=1e-12)
        assert np.min(result[7]) >= -1e-10
        assert np.max(result[8]) <= 1e-10


@pytest.mark.parametrize("direction,rows", [
    ([1., -2., 3.], np.empty((0, 3))),
    ([-1., -2., 3.], [[1., 0., 0.], [0., 1., 0.]]),
    ([-1., -2., 3.], [[1., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),
    ([1., -2., 3.], [[1., 0., 0.], [-1., 0., 0.]]),
    ([-1., 2., 3.], [[1e-12, 0., 0.], [0., 1e12, 0.]]),
    ([0., 0., 0.], [[0., 0., 0.]]),
    ([-1., -1.], [[1., 0.], [0., 1.]]),
])
def test_projection_degenerate_analytic_cones(direction, rows):
    projection(direction, rows)


@pytest.mark.parametrize("separation", (1e-3, 2.00001e-6, 1.99999e-6, 1e-8))
def test_nearly_parallel_rank_cutoff_neighbors(separation):
    projection([1., -2., 3.], [[1., 0., 0.], [-1., separation, 0.]], analytic=[0., 0., 3.])


def cagrad(rows):
    rows = np.asarray(rows, dtype=float)
    function = tf.function(certified_cagrad_impl, autograph=False, input_signature=[tf.TensorSpec(rows.shape, tf.float64)])
    coefficients, direction, diagnostics, valid = function(rows)
    assert bool(valid), diagnostics
    weights, multiplier, _norm, value, gap = [item.numpy() for item in diagnostics]
    base = rows.mean(axis=0)
    epsilon = float(np.float32(1e-8))
    coefficient = .5 * np.sqrt(base @ base + epsilon) + epsilon
    weighted = weights @ rows
    gradient = rows @ base + coefficient * (rows @ weighted) / np.sqrt(weighted @ weighted + epsilon)
    scale = max(1., abs(float(value)))
    assert weights.min() >= 0 and weights.sum() == pytest.approx(1., abs=1e-12)
    assert float(gap) <= 1e-10 * scale
    assert weights @ gradient - gradient.min() <= 1e-10 * scale
    np.testing.assert_allclose(direction, (base + multiplier * weighted) / 1.25, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(coefficients.numpy() @ rows, direction, rtol=1e-10, atol=1e-10)

    def objective(alpha):
        combined = alpha @ rows
        return (base @ combined + coefficient * np.sqrt(combined @ combined + epsilon)) / scale

    reference = minimize(objective, np.ones(len(rows)) / len(rows), method="SLSQP",
                         bounds=[(0, 1)] * len(rows), constraints={"type": "eq", "fun": lambda alpha: alpha.sum() - 1},
                         options={"ftol": 1e-13, "maxiter": 1000})
    assert objective(weights) <= reference.fun + 1e-9
    return weights, gradient


@pytest.mark.parametrize("case", range(2))
def test_captured_cagrad_unridged_optimum(case):
    with fixture_arrays(f"sgu{case}_") as saved:
        rows = saved["partition_input_1"][[1, 5, 6]]
    weights, _gradient = cagrad(rows)
    base = rows.mean(axis=0)
    epsilon = float(np.float32(1e-8))
    coefficient = .5 * np.sqrt(base @ base + epsilon) + epsilon

    def derivative(weight):
        combined = weight * rows[1] + (1 - weight) * rows[2]
        return (rows[1] - rows[2]) @ (base + coefficient * combined / np.sqrt(combined @ combined + epsilon))

    reference = brentq(derivative, 0., 1., xtol=1e-15)
    np.testing.assert_allclose(weights, [0, reference, 1 - reference], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("rows", [
    np.eye(3), np.zeros((3, 4)), np.ones((3, 4)),
    [[1., 0.], [-1., 0.], [0., 0.]],
    [[1., 0.], [2., 0.], [3., 0.]],
    [[1e4, 1.], [1e-5, -1e-5], [2e-3, 1e-3]],
    [[1e-8, 1e-8], [1e-8, -1e-8]],
    [[1.]], [[1.], [-1.], [2.]],
    np.random.default_rng(901).normal(size=(4, 8)),
])
def test_cagrad_degenerate_and_mixed_scales(rows):
    cagrad(rows)


def test_zero_gradient_momentum_and_nonfinite_rollback():
    update = make_certified_projected_adam_function(3, 1)
    args = [np.ones(3), np.array([.1, -.2, .3]), np.ones(3), np.int64(3), np.zeros(3), np.array([[1., 0., 0.]]), 1e-3]
    result = update(*args)
    assert bool(result[-1])
    np.testing.assert_array_equal(result[4], 0.)
    assert np.linalg.norm(result[6]) > 0
    for index in (0, 1, 2, 4, 5):
        damaged = list(args)
        damaged[index] = np.full_like(damaged[index], np.nan)
        rejected = update(*damaged)
        assert not bool(rejected[-1])
        for returned, original in zip(rejected[:4], damaged[:4], strict=True):
            np.testing.assert_array_equal(returned, original)


def repaired_executor():
    legacy, initial, batch = fixture()
    executor = PermanentPassExecutor(legacy.adapter, legacy.spec, roles(), threshold=.04,
                                    method_factory=make_certified_method_function, method_binding=numerical_binding(),
                                    update_factory=make_certified_projected_adam_function, update_binding=numerical_binding())
    return executor, initial, batch


def test_shared_transaction_dispatch_resume_and_legacy_binding_refusal():
    executor, initial, batch = repaired_executor()
    state = executor.initialize(initial, fake_control(initial, executor.adapter.registry.task_ids, [.2] * 7))
    first, event = executor.step(state, batch)
    assert event["method"] == "cagrad" and all(item["accepted"] for item in event["method_events"])
    direct, _event = executor.step(first, batch)
    restarted, _initial, _batch = repaired_executor()
    recovered = CheckpointState.from_dict(json.loads(canonical_json(first.to_dict())))
    resumed, _event = restarted.step(recovered, batch)
    assert canonical_json(resumed.to_dict()) == canonical_json(direct.to_dict())
    assert executor.kernel_cache["method_factory"] is make_certified_method_function
    assert executor.kernel_cache["update_factory"] is make_certified_projected_adam_function
    legacy, _initial, _batch = fixture()
    with pytest.raises(ValueError, match="binding mismatch"):
        legacy._validate_numerical_state(recovered)


@pytest.mark.parametrize("factory,binding", [(make_certified_method_function, None), (None, numerical_binding()), (make_certified_method_function, {})])
def test_method_factory_requires_explicit_binding(factory, binding):
    with pytest.raises(ValueError):
        PermanentPassExecutor(None, None, None, threshold=.04, method_factory=factory, method_binding=binding)


@pytest.mark.parametrize("count", (1, 3, 6))
def test_preserved_rotemberg_cagrad_operands(count):
    with fixture_arrays(f"rotemberg{count}_") as operands:
        weights, _gradient = cagrad(operands["rows"])
        result = certified_cagrad_impl(operands["rows"])
        assert weights.size == count
        np.testing.assert_allclose(result[1], operands["direction"], rtol=2e-6, atol=1e-8)


def test_preserved_rotemberg_projector_operands():
    with fixture_arrays("rotemberg_projection_") as operands:
        projection(operands["direction"], operands["protected_rows"])


@pytest.mark.parametrize("scale", (1e-8, 1., 1e8))
def test_cagrad_common_scale_and_nonfinite_rejection(scale):
    rows = np.random.default_rng(912).normal(size=(3, 4)) * scale
    cagrad(rows)
    rows[0, 0] = np.nan
    assert not bool(certified_cagrad_impl(rows)[-1])


def test_cagrad_canary_small_barycentric_weight():
    with fixture_arrays("canary64_") as operands:
        weights, gradient = cagrad(operands["rows"])
        assert weights[0] == 0
        assert 1e-6 < weights[1] < 1e-4
        assert abs(gradient[1] - gradient[2]) < 1e-8
