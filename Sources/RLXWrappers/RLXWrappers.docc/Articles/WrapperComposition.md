# Wrapper composition

## Overview

Wrappers should:

1. Store `inner`
2. Forward spaces, `spec`, `reset` / `step` / `close` unless intentionally overridden
3. Preserve or deliberately transform spaces (transforming wrappers may change associated types)

### Order of stacking (practical)

From **outermost** (first to receive calls) to **innermost** (dynamics):

1. ``PassiveEnvChecker`` — validate values at the boundary you care about
2. ``RecordEpisodeStatistics`` — sees true episode ends after limits
3. ``TimeLimit`` — truncates long episodes
4. ``OrderEnforcing`` — rejects illegal call sequences
5. **Your env** or transform wrappers (`ClipAction`, etc.)

Exact order can vary; document stacks in factories (see CartPole registration in `RLXEnvs`).

### Type erasure

At registry or heterogeneous boundaries, box with ``AnyEnvironment``. Transform wrappers that
change observation/action types need careful placement below erasure.
