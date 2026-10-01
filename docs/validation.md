# Extraction validation — October 1, 2026

The standalone installed wheel has 2,595 distinct passing tests,
10 explicit skips and one documented upstream expected failure.
Coverage combines the initial full sweep, repair of every failed or unexecuted
case, and a full regression of every test file transitively importing either
path-repaired runner. Reruns are counted once. See [validation.json](validation.json)
for suite counts/times and the exact known-failure measurements.

Tests ran on Python 3.11 / TensorFlow 2.21.0, CPU-only, four pinned host cores.
Both former source packages (`dsge_hmc` and `common_utils`) were blocked during
package tests. Production module imports came from an installed wheel, not an
editable source checkout. Importing the package base is inert. Static and
runtime import checks cover all 71 production Python files.

MacroFinance's real bond/equity pricing adapter passed
14 tests, with 2 inapplicable
bond-only equity cases skipped. That separate process also blocks DSGE imports.
It checks six distinct task gradients, finite differences, compiled/eager parity,
accepted updates, explicit rejection, full-state rollback and checkpoint replay.
The standalone quadratic example committed one constrained update.

## Known inherited numerical limitation

The symmetric zero-bias coverage warm-start fixture is rejected by the unchanged
normal-equation certificate under TensorFlow 2.21. The untouched upstream code
at `0ec4a4d776e0985dd50efe40e9bed308733c46a6` independently reproduces this: normalized residual
`3.26551844086e-10` exceeds its `1e-10` cutoff.
The biased fit is valid (residual `1.37332846944e-13`).

The numerical code and cutoff are preserved. All value, derivative and curvature
assertions run before a dedicated exception records this single strict expected
failure. Other assertion failures remain failures. A future fix makes this
marker fail as an unexpected pass and requires review. This is a conservative
rejection, not permission to use an invalid warm start.

## Scope and exclusions

The upstream checkout-specific imports, source paths, stale mocks and fresh-import
test assumptions were adapted. Saved DSGE runs, model-specific drivers and
external official-code benchmark campaigns remain upstream; every exclusion
is recorded in [extraction.json](extraction.json). Optional upstream numerical
integration checks retain their explicit opt-in skips.

The 50 generic modules, two MOO/Adam kernels, 16 multiobjective utilities and
artifact helpers preserve the numerical algorithms. The complete non-namespace
production diff is [production-path-changes.patch](production-path-changes.patch):
neutral program names, installed-source/client-evidence separation, and safe
retained-snapshot lookup. Test-only historical oracles and static operands are
kept under `tests/support` and `tests/fixtures`.

The current host dispatcher and NumPy finite guard are **not a fully XLA compiled
training update**. Neither this extraction nor its tests establishes trained
pricing accuracy, GPU performance, model identification or scientific admission.
The client retains those qualification responsibilities.
