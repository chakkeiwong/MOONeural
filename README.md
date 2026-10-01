# MOONeural

Multi-objective neural solver training, extracted from DSGE's generic training
stack into a standalone Python package. No `dsge_hmc`, `common_utils`, economic
model, filter or HMC package is required.

MOONeural solves problems with **separate task constraints**. For task errors
`e_j(w)`, a typical adapter supplies `L_j = mean(e_j**2)` and its own gradient.
Training uses `L_j / D_j`; acceptance uses independently specified limits `T_j`.
A small combined objective does not certify that every constraint is satisfied.

## Included

- Generic task, role, normalization and checkpoint contracts.
- Permanent-pass task rotation and multi-objective directions: CAGrad, PCGrad,
  MGDA, GradNorm and an active-only normalized-sum comparator.
- Constraint projection before Adam and projection of its actual displacement.
- Finite trial screening, transactional rejection/rollback and checkpoint replay.
- Functional model adapters, derivative/coverage initialization, conditional
  expectations, control banks, scheduled roles and independent role bridges.
- Replication/campaign controllers, durable artifact storage and statistical
  evaluation helpers.
- TensorFlow multiobjective utilities, including additional reference methods
  whose availability is distinct from the permanent-pass method population.

The extraction includes 50 generic training modules, two compiled MOO/Adam
kernels, 16 TensorFlow multiobjective utility modules and artifact helpers. Eight
DSGE-specific campaign drivers/probes remain in the original project; see the
complete inventory in [docs/extraction.json](docs/extraction.json).

## Install

```sh
python -m pip install .
```

Python 3.11+ and TensorFlow 2.16+ are declared dependencies. The extraction is
tested on Python 3.11/TensorFlow 2.21; that does not certify every dependency
version or GPU backend. The application owns device selection, thread limits
and memory-growth configuration. Importing `mooneural` has no runtime side
effects. This package has no TensorFlow Probability dependency.

## Use with a model

```python
from mooneural.training.generic_permanent_pass_executor import PermanentPassExecutor
from mooneural.training.generic_postproposal import GuardedPostproposal
from mooneural.training.generic_training_contracts import TaskDefinition, TaskRegistry

tasks = TaskRegistry(tuple(TaskDefinition(name, i) for i, name in enumerate(
    ("price", "price_state", "price_parameter", "euler", "euler_state", "euler_parameter")
)))
```

Your adapter provides the task registry, normalizations, optional hard checks,
and an evaluation factory. The factory returns raw losses, each raw gradient
row, normalized losses/rows, hard-check maxima, connectivity and validity. The
executor owns optimization and checkpoint transactions. Models supply their
own equations, parameters, batches, independent control/selection/terminal
data and uncertainty treatment. A working small example is
[examples/quadratic.py](examples/quadratic.py).

See [docs/architecture.md](docs/architecture.md) for interfaces, update semantics
and extraction/migration boundaries.

## Compilation and limits

The numerical evaluation, direction and projected-Adam kernels support XLA.
The current executor dispatches on the host, and the optional finite guard
contains NumPy/Python operations. **The full training update/loop is not fully
XLA compiled.** Native projection is a separate explicit backend; it must not
be represented as full XLA readiness.

Permanent membership is a rotation rule, not a scientific accuracy certificate.
The basic finite guard checks active loss decrease and preserves feasible
protected limits; it does not guarantee finite nonincrease of every infeasible
task. A zero direction, rejection or exhausted budget is not successful
constraint satisfaction. Model-specific limits and numerical tolerances need
calibration and independent verification by the application.

## Tests

```sh
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  python -m pytest -q -p no:cacheprovider
```

Install the test extra with `python -m pip install ".[test]"`.

Tests block imports from DSGE and `common_utils`, and include upstream numerical
comparators as explicitly labeled test-only oracles. CPU test configuration is
set before TensorFlow import. No GPU result is implied. The installed-wheel
and MacroFinance integration results are recorded in [docs/validation.md](docs/validation.md).

See [NOTICE](NOTICE) for attribution and license status.
