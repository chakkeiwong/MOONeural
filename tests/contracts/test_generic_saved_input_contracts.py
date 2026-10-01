"""Contracts for immutable host-owned saved-input bundles."""

import numpy as np
import pytest

from mooneural.training.generic_saved_input_contracts import (
    FrozenTensorBundle,
    SourceHashBinding,
    TensorFieldSpec,
    array_fingerprint,
)

SPECS = (
    TensorFieldSpec("rows", 2, (2,)),
    TensorFieldSpec("fixed", 1, (2,), row_axis=None),
)


def make_bundle() -> FrozenTensorBundle:
    return FrozenTensorBundle.from_mapping(
        SPECS,
        {
            "rows": np.arange(6, dtype=np.float64).reshape(3, 2),
            "fixed": np.array([10.0, 20.0], dtype=np.float64),
        },
    )


def test_bundle_preserves_order_copies_arrays_and_is_read_only():
    source = np.arange(6, dtype=np.float64).reshape(3, 2)
    bundle = FrozenTensorBundle.from_mapping(
        SPECS,
        {"rows": source, "fixed": np.array([10.0, 20.0])},
    )

    source[0, 0] = 99.0
    assert bundle.names == ("rows", "fixed")
    assert bundle.row_count == 3
    assert bundle.mapping["rows"][0, 0] == 0.0
    assert not bundle.mapping["rows"].flags.writeable
    with pytest.raises(ValueError):
        bundle.mapping["rows"][0, 0] = 1.0
    with pytest.raises(TypeError):
        bundle.mapping["new"] = np.zeros(1)


def test_fingerprints_are_stable_and_content_sensitive():
    bundle = make_bundle()
    assert bundle.fingerprint == bundle.fingerprint
    changed = np.array(bundle.mapping["rows"], copy=True)
    changed[0, 0] += 1.0
    assert bundle.fingerprint != array_fingerprint(
        bundle.names, (changed, bundle.mapping["fixed"])
    )


@pytest.mark.parametrize(
    "bad_rows",
    [
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((3, 2), dtype=np.float32),
        np.array([[np.nan, 0.0]], dtype=np.float64),
        np.zeros((0, 2), dtype=np.float64),
    ],
)
def test_field_validation_refuses_bad_shape_dtype_finiteness_or_empty_rows(bad_rows):
    with pytest.raises((TypeError, ValueError)):
        FrozenTensorBundle.from_mapping(
            SPECS,
            {"rows": bad_rows, "fixed": np.array([10.0, 20.0])},
        )


def test_bundle_refuses_missing_extra_reordered_and_mismatched_rows():
    with pytest.raises(ValueError, match="field order"):
        FrozenTensorBundle.from_mapping(
            SPECS,
            {"fixed": np.array([10.0, 20.0]), "rows": np.zeros((3, 2))},
        )
    with pytest.raises(ValueError, match="row count"):
        FrozenTensorBundle.from_mapping(
            (TensorFieldSpec("left", 2, (2,)), TensorFieldSpec("right", 2, (2,))),
            {"left": np.zeros((3, 2)), "right": np.zeros((4, 2))},
        )


def test_source_hash_binding_normalizes_and_reports_provenance():
    binding = SourceHashBinding(
        {"worker.py": "A" * 64},
        "B" * 64,
        "rotemberg.public.explicit",
    )
    assert binding.source_hashes["worker.py"] == "a" * 64
    assert binding.input_hash == "b" * 64
    assert binding.metadata()["target_id"] == "rotemberg.public.explicit"
