// RandomAgentDemo — minimal interaction loop (design.md P8 / PR-15).
//
// Samples actions from the action space (no learned policy). Demonstrates reset/step,
// optional PassiveEnvChecker + OrderEnforcing + TimeLimit stack, and episode returns.
// Uses DummyEnv so `swift run RandomAgentDemo` works on Linux CPU and macOS without Metal.

import Foundation
import RLXCore
import RLXEnvs
import RLXWrappers

let episodes = 5
let maxStepsHint = 20
var rng = SplitMix64(seed: 42)

// Recommended dev stack: value checks + call-order + time limit (design.md §15, §20.2).
let env = PassiveEnvChecker(
    OrderEnforcing(
        TimeLimit(DummyEnv(episodeLength: 10), maxEpisodeSteps: maxStepsHint)
    )
)

print("RandomAgentDemo — rlx-swift \(RLXCore.version)")
print("Env: PassiveEnvChecker(OrderEnforcing(TimeLimit(DummyEnv)))")
print("Episodes: \(episodes)\n")

var returns: [Float] = []

for ep in 0..<episodes {
    let reset = try env.reset(seed: UInt64(ep) as UInt64?, options: nil)
    _ = reset.observation
    var episodeReturn: Float = 0
    var steps = 0
    var done = false
    while !done {
        let action = env.actionSpace.sample(using: &rng)
        let step = try env.step(action)
        episodeReturn += step.reward
        steps += 1
        done = step.done
    }
    returns.append(episodeReturn)
    print(String(format: "  episode %d: return=%.1f steps=%d", ep, episodeReturn, steps))
}

try env.close()
let mean = returns.reduce(0, +) / Float(returns.count)
print(String(format: "\nMean return over %d episodes: %.2f", episodes, mean))
print("OK")
