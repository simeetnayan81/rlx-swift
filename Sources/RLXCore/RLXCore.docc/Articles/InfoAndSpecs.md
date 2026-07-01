# Info, specs, and metadata

## Overview

### Info

``Info`` is a structured bag on reset/step for diagnostics that are not part of the core
transition tuple. Values are ``InfoValue`` (bool, int, double, string, nested info, `MLXArray`).

Prefer **stable, scoped keys**. Compile-time constants live in `InfoKeys` (`RLXWrappers`), including:

| Key | Set by | Meaning |
|-----|--------|---------|
| `TimeLimit.truncated` | TimeLimit | Truncation due to step limit (flag remains primary) |
| `final_observation` / `final_info` | Vector autoreset | Terminal obs/info when live obs is next episode |
| `episode` (`r`, `l`, optional `t`) | RecordEpisodeStatistics | Completed episode metrics |

Env-specific keys should be scoped (e.g. nested under an env id) to avoid collisions.

### EnvSpec

``EnvSpec`` is **immutable kind metadata** (id, max episode steps, thresholds, version) — not
instance state. Registries attach specs to factories.

### Related design sections

`design.md` §14 (info & specs).
