"""Host construction/identity checks; native numerical qualification is separate."""

import inspect

import pytest
import tensorflow as tf

from mooneural.training.generic_execution_boundary import (
    DEFAULT_ADAM,
    make_projected_adam_function,
)
from mooneural.training.generic_native_projection import (
    make_native_projected_adam_function,
    make_native_projection_function,
    native_update_binding,
)
from mooneural.training.generic_xla_kernels import project_cone_impl
from mooneural.training.adam import packed_adam_impl


@pytest.mark.parametrize("count", range(7))
def test_update_backend_defined_factory_signatures_without_tracing(count):
    update = make_native_projected_adam_function(8, count)
    captures = inspect.getclosurevars(update.python_function).nonlocals
    projector, packed = captures["projector_impl"], captures["packed_impl"]
    assert update._jit_compile is False
    assert projector._jit_compile is False
    assert packed._jit_compile is True
    assert update.input_signature[5] == tf.TensorSpec([count, 8], tf.float64, "protected_rows")
    assert projector.input_signature[2] == tf.TensorSpec(
        [(1 << count) - 1, count], tf.bool, "subset_masks"
    )
    assert captures["config"] == DEFAULT_ADAM
    assert all(function.experimental_get_tracing_count() == 0 for function in (update, projector, packed))


def test_update_backend_default_factory_keeps_original_callables_and_signature():
    update = make_projected_adam_function(8, 4)
    captures = inspect.getclosurevars(update.python_function).nonlocals
    assert update._jit_compile is True
    assert captures["projector_impl"] is project_cone_impl
    assert captures["packed_impl"] is packed_adam_impl
    assert str(inspect.signature(make_projected_adam_function)).startswith("(parameter_dim, constraint_count, config=")
    assert update.experimental_get_tracing_count() == 0


@pytest.mark.parametrize("dimension,count", ((0, 4), (True, 4), (8, -1), (8, 7), (8, True)))
def test_update_backend_native_factory_refuses_invalid_dimensions(dimension, count):
    with pytest.raises(ValueError):
        make_native_projection_function(dimension, count)


def test_update_backend_profile_returns_independent_canonical_payloads():
    original = native_update_binding()
    modified = native_update_binding()
    modified["projection"]["rcond"] = 0.5
    assert original == native_update_binding() != modified
    assert original["legacy_guard"]["qualification"] == "diagnostic_only_pending_source_failure_domain_coverage"
