# ``RLXVector``

Batched / parallel logical environments for data collection.

@Metadata {
    @TechnologyRoot
}

## Overview

| Type | Style | Notes |
|------|--------|------|
| ``SyncVectorEnv`` | Synchronous `throws` | Sequential slots; simple tests & small `N` |
| ``AsyncVectorEnv`` | `async throws` actor | Concurrent slots; `maxConcurrency: 1` for serial tests; cancel on `close` |

Shared concepts:

- ``AutoresetMode`` — `.disabled`, `.nextStep`, `.sameStep` (default **`.sameStep`**)
- ``VectorResetResult`` / ``VectorStepResult`` — per-slot arrays (index order preserved)
- ``VectorEnvironmentError`` — closed / cancelled / batch size mismatch
- Info keys `final_observation` / `final_info` on same-step autoreset

Seeding: `reset(seed:)` uses ``Seed/child(index:)`` per slot.

Single-env ``Environment`` APIs remain **synchronous**; parallelism lives here (`design.md` §21).

## Topics

### Vector envs

- ``SyncVectorEnv``
- ``AsyncVectorEnv``

### Policy and results

- ``AutoresetMode``
- ``VectorResetResult``
- ``VectorStepResult``
- ``VectorEnvironmentError``

## See also

- Repository `Documentation/DeveloperGuide.md` (vector collection pattern)
- `design.md` §16 (vectorized & parallel), §13.4 (vector seeding)
