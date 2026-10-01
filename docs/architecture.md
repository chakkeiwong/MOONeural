# Reusable training boundary

The client owns its equations and scientific claims. MOONeural owns task
contracts, update/control state, constrained numerical primitives, guarded
transactions and repeatable artifact handling.

`training.generic_permanent_pass_executor.PermanentPassExecutor` accepts an
`evaluation_factory(adapter, ExecutionSpec)` plus an explicit source binding.
This is the extension point used by MacroFinance. Its evaluator has signature
`evaluate(parameters, *batch, update_index)` and returns:

1. Raw task loss vector and separate weight-gradient matrix.
2. Normalized task loss vector and gradient matrix, using the same `1/D_j`.
3. Hard-check maxima, task connectivity flags and finite/valid status.

The registry and normalization metadata declare a stable task order. Each
proposed policy is a packed vector; a functional adapter must evaluate that
vector without modifying the saved parent model. `generic_model_task_executor`
provides a higher-level adapter boundary where its contracts fit the model.

## Update semantics

At declared independent control boundaries, passing tasks become permanent
constraints. Updates rotate over up to three unresolved tasks. All other tasks
are protected. A combiner uses only active gradients, then the shared kernel
projects its direction, applies Adam and projects the actual displacement.
Method failure/fallback state and PCGrad update seeds are checkpointed.

`generic_postproposal.GuardedPostproposal` optionally evaluates the finite
candidate on the same frozen batch. Its basic profile tries declared fractions,
requires strict active loss decrease and keeps already-feasible protected tasks
below their raw limits. Rejection restores the complete parent transaction.
The host controller and this guard are not a compiled whole-update interface.

`D_j` changes numerical scale; `T_j` changes the declared feasible set. They must
not be conflated. Aggregate means can hide a failing derivative coordinate;
clients must supply their own coordinate/country/horizon checks and independent
accuracy criteria. The library's synthetic thresholds are not economic defaults.

## Other modules

| Area | Modules |
|---|---|
| Contracts and control | `generic_training_contracts`, `generic_permanent_pass`, `generic_role_banks`, `generic_scheduled_role_banks` |
| Numerical methods | `generic_xla_kernels`, `moo`, `adam`, `generic_certified_numerics`, `generic_native_projection` |
| Finite checks and diagnostics | `generic_postproposal`, `generic_displacement_blend`, component/curvature/relative-progress and residual helpers |
| Model and objective interfaces | `generic_model_task_executor`, finite/coverage runners, normalization, policy-coordinate and derivative helpers |
| Reproducible experiments | replication runners/statistics, `generic_training_campaign`, protocol/program helpers, `artifacts` |
| Broader MOO utilities | `multiobjective` |

Program/protocol helpers are optional host-side orchestration. Their plan and
state files belong to the client; a fresh installation ships no claimed phase
passes or historical DSGE campaign state. The default program ID is
`mooneural-training`, with plan `docs/plans/mooneural-training-program.md`.

## Provenance and migration

`docs/extraction.json` records the original revision, every extracted file hash,
namespace/path changes, test adaptations and omitted model/campaign drivers.
Numerical algorithms are preserved. The generic `sgu_xla_moo` and
`sgu_xla_optimizer` modules are named `moo` and `adam` here. Shared utilities
formerly under `common_utils.tf_multiobjective` live in `mooneural.multiobjective`.

Historical `dsge_hmc.*` serialized schema names are retained as format
identifiers; they are not imports or runtime dependencies. Source-bound executor
identities do change on extraction. Existing checkpoints must be deliberately
rebound and verified by the client; do not silently bypass the source/identity
checks to resume an old campaign. MacroFinance preserves its old evidence and
uses a fresh source-bound adapter for this dependency transition.

Eight modules with generic names are nevertheless DSGE-specific campaign
entry points or provider bindings. They remain upstream. Their reusable task,
bank, statistical and checkpoint engines are included. Economics-specific
model adapters and source empirical artifacts are client responsibilities.

## Installed packages and artifacts

The optional coverage and fixed-objective runners bind library source relative
to the installed package (`SOURCE_ROOT`), while relative client evidence is
anchored to the working directory at module import (`ROOT`). Launch those
helpers from the client project root or supply absolute artifact references.
Generated references are absolute; library snapshots keep package-relative
file names. An external callback's live file cannot substitute for a missing
retained snapshot during continuation. This preserves recovery when the wheel
and experiment directory are separate. No files are written into the installed
package by the training runners.
