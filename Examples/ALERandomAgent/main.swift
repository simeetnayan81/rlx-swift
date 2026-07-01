// ALERandomAgent — random policy on ALE when linked + ROM path given.
//
//   ALE_ROOT=... ALE_ROM_PATH=/path/to/pong.bin swift run ALERandomAgent
//
// Without ALE_ROOT, prints how to enable the adapter and exits 0 (so default CI is happy).

import Foundation
import RLXALE
import RLXCore
import RLXWrappers

guard RLXALE.isALELinked else {
    print("""
    ALERandomAgent: ALE C++ is not linked into this build (stub shim).

    To enable:
      1. Build & install Farama ALE (CMake, SDL off is fine)
      2. export ALE_ROOT=/path/to/ale/install
      3. export ALE_ROM_PATH=/path/to/game.bin
      4. swift build --product ALERandomAgent   # rebuild so Package.swift sees ALE_ROOT
      5. swift run ALERandomAgent

    Design: docs/ale-adapter-design.md
    """)
    exit(0)
}

guard let rom = ProcessInfo.processInfo.environment["ALE_ROM_PATH"], !rom.isEmpty else {
    fputs("ALERandomAgent: set ALE_ROM_PATH to a ROM file\n", stderr)
    exit(2)
}

let raw = try ALEEnvironment(
    config: ALEConfig(
        romPath: rom,
        observationType: .grayscale,
        frameSkip: 4,
        repeatActionProbability: 0,
        livesPolicy: .gameOverOnly,
        seed: 42
    )
)
let env = OrderEnforcing(raw)

var rng = SplitMix64(seed: 7)
_ = try env.reset(seed: 42 as UInt64?, options: nil)

var total: Float = 0
var steps = 0
let maxSteps = 2_000
while steps < maxSteps {
    let action = env.actionSpace.sample(using: &rng)
    let step = try env.step(action)
    total += step.reward
    steps += 1
    if step.done { break }
}

print("ALERandomAgent: steps=\(steps) return=\(total) linked=true rom=\(rom)")
try env.close()
