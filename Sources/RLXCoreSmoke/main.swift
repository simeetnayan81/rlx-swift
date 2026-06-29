// Tier-2 smoke: links RLXCore + exercises PR-02/PR-03 types without MLXArray eval
// (avoids Metal/runtime resource issues on CLI and Linux CPU toolchains).
// Tier-1 XCTest (incl. MLX key equality) runs via xcodebuild on macOS.

import Foundation
import RLXCore

enum SmokeFailure: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        switch self {
        case .message(let text): return text
        }
    }
}

func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw SmokeFailure.message(message) }
}

do {
    // Identity
    try expect(!RLXCore.version.isEmpty, "RLXCore.version must be non-empty")
    try expect(RLXCore.version.contains("0.1.0"), "RLXCore.version should contain 0.1.0, got \(RLXCore.version)")
    let _: () -> Any = { RLXCore.mlxSmokeArray() as Any }

    // Info: empty, subscript, remove, set, merge, nested, equality
    var info = Info()
    try expect(info.isEmpty, "fresh Info should be empty")
    try expect(info.count == 0, "fresh Info count 0")
    try expect(info["missing"] == nil, "missing key is nil")

    info["TimeLimit.truncated"] = .bool(true)
    info["steps"] = .int(3)
    info["score"] = .double(1.5)
    info["label"] = .string("ok")
    info["episode"] = .nested(Info(["r": .double(1.0), "l": .int(2)]))
    try expect(info.count == 5, "five keys after inserts")
    try expect(info["TimeLimit.truncated"] == .bool(true), "bool key")
    try expect(info["steps"] == .int(3), "int key")
    try expect(info["score"] == .double(1.5), "double key")
    try expect(info["label"] == .string("ok"), "string key")
    guard case .nested(let episode)? = info["episode"] else {
        throw SmokeFailure.message("expected nested episode")
    }
    try expect(episode["r"] == .double(1.0), "nested double")
    try expect(episode["l"] == .int(2), "nested int")

    let removed = info.removeValue(forKey: "score")
    try expect(removed == .double(1.5), "remove returns prior value")
    try expect(info["score"] == nil, "removed key is nil")

    _ = info.set("extra", .bool(false))
    try expect(info["extra"] == .bool(false), "set helper")

    var base = Info(["a": .int(1), "b": .string("keep")])
    base.merge(Info(["a": .int(2), "c": .bool(true)]))
    try expect(base["a"] == .int(2), "merge overwrite")
    try expect(base["b"] == .string("keep"), "merge keep")
    try expect(base["c"] == .bool(true), "merge insert")

    let merged = Info(["x": .int(1)]).merging(Info(["x": .int(9)]))
    try expect(merged["x"] == .int(9), "merging overwrite")

    let infoCopy = Info([
        "TimeLimit.truncated": .bool(true),
        "steps": .int(3),
        "label": .string("ok"),
        "episode": .nested(Info(["r": .double(1.0), "l": .int(2)])),
        "extra": .bool(false),
    ])
    try expect(info == infoCopy, "Info equality")
    try expect(InfoValue.bool(true) != .int(1), "InfoValue kind inequality")

    // ResetResult / StepResult
    let reset = ResetResult(observation: 0, info: info)
    try expect(reset.observation == 0, "reset observation")
    try expect(!reset.info.isEmpty, "reset info preserved")
    let reset2 = ResetResult(observation: 0, info: info)
    try expect(reset == reset2, "ResetResult equality")
    try expect(reset != ResetResult(observation: 1), "ResetResult inequality")

    let step = StepResult(
        observation: 1,
        reward: 0.5,
        terminated: false,
        truncated: true,
        info: Info(["note": .string("limit")])
    )
    try expect(step.done, "truncated step is done")
    try expect(!step.terminated && step.truncated, "truncation flags")
    try expect(step.reward == Float(0.5), "reward is Float")
    try expect(step.info["note"] == .string("limit"), "step info")
    try expect(
        StepResult(observation: 0, reward: 0, terminated: false, truncated: false).done == false,
        "continuing not done"
    )
    try expect(
        StepResult(observation: 0, reward: 0, terminated: true, truncated: false).done,
        "terminated is done"
    )
    try expect(
        StepResult(observation: 0, reward: 0, terminated: true, truncated: true).done,
        "both flags is done"
    )

    let step2 = StepResult(
        observation: 1,
        reward: 0.5,
        terminated: false,
        truncated: true,
        info: Info(["note": .string("limit")])
    )
    try expect(step == step2, "StepResult equality")
    try expect(
        step != StepResult(observation: 1, reward: 0.5, terminated: true, truncated: false),
        "StepResult inequality on flags"
    )

    // EnvironmentError equality + descriptions
    try expect(EnvironmentError.notReset == .notReset, "notReset eq")
    try expect(EnvironmentError.episodeEnded == .episodeEnded, "episodeEnded eq")
    try expect(EnvironmentError.closed == .closed, "closed eq")
    try expect(EnvironmentError.renderNotSupported == .renderNotSupported, "render eq")
    try expect(EnvironmentError.closed != .notReset, "closed neq notReset")
    try expect(
        EnvironmentError.invalidAction("oob") == .invalidAction("oob"),
        "invalidAction eq"
    )
    try expect(
        EnvironmentError.invalidAction("a") != .invalidAction("b"),
        "invalidAction neq"
    )
    try expect(
        EnvironmentError.invalidObservation("s") == .invalidObservation("s"),
        "invalidObservation eq"
    )
    try expect(
        EnvironmentError.configuration("c") == .configuration("c"),
        "configuration eq"
    )
    try expect(
        EnvironmentError.invalidAction("x") != .configuration("x"),
        "different cases unequal"
    )
    struct E1: Error {}
    struct E2: Error {}
    try expect(
        EnvironmentError.underlying(E1()) == .underlying(E2()),
        "underlying is case-only equal"
    )
    try expect(
        EnvironmentError.underlying(E1()) != .closed,
        "underlying neq closed"
    )
    try expect(EnvironmentError.notReset.description.contains("notReset"), "desc notReset")
    try expect(EnvironmentError.invalidAction("x").description.contains("x"), "desc action")

    // RenderMode / RenderFrame (non-tensor cases)
    try expect(RenderMode.allCases.count == 4, "four render modes")
    try expect(RenderMode.none.rawValue == "none", "none raw value")
    try expect(RenderMode.human.rawValue == "human", "human raw value")
    try expect(RenderMode.rgbArray.rawValue == "rgbArray", "rgbArray raw value")
    try expect(RenderMode.ansi.rawValue == "ansi", "ansi raw value")
    try expect(RenderFrame.ansi("x") == .ansi("x"), "ansi frame eq")
    try expect(RenderFrame.ansi("x") != .ansi("y"), "ansi frame neq")
    try expect(RenderFrame.humanDisplayed == .humanDisplayed, "humanDisplayed eq")
    try expect(RenderFrame.ansi("x") != .humanDisplayed, "ansi neq humanDisplayed")

    // Codable round-trip for RenderMode (Foundation JSON, no MLX)
    let encoded = try JSONEncoder().encode(RenderMode.human)
    let decoded = try JSONDecoder().decode(RenderMode.self, from: encoded)
    try expect(decoded == .human, "RenderMode Codable")

    // PR-03: Seed + SplitMix64 (pure Swift; no MLX eval)
    let seed = Seed(42)
    try expect(seed.rawValue == 42 && seed.uint64 == 42, "Seed rawValue")
    try expect(Seed(rawValue: 42) == seed, "Seed equality")
    let base1 = Seed(1)
    try expect(base1.child(index: 0).rawValue == 0x910a2dec89025cc1, "Seed.child(0) golden")
    try expect(base1.child(index: 0) == Seed(1).child(index: 0), "Seed.child determinism")
    try expect(base1.child(index: 0) != base1.child(index: 1), "Seed.child distinct")
    try expect(base1.child(index: 1) != base1, "child not parent")
    try expect(seed.child(index: 0) == Seed(42).child(index: 0), "Seed(42).child determinism")

    var sm0 = SplitMix64(seed: 0xDEAD_BEEF)
    var sm1 = SplitMix64(seed: Seed(0xDEAD_BEEF))
    try expect(sm0.next() == sm1.next(), "SplitMix64 reproducibility")
    var smGold = SplitMix64(seed: 0xDEAD_BEEF)
    try expect(smGold.next() == 0x4adfb90f68c9eb9b, "SplitMix64 golden first")
    // MLX-backed PRNG.key / split need Metal metallib — covered by tier-1 XCTest only.

    // PR-04: DiscreteSpace Swift RNG path (no MLX key / eval)
    let discrete = DiscreteSpace(n: 4, start: 1)
    try expect(discrete.contains(1) && discrete.contains(4), "discrete contains ends")
    try expect(!discrete.contains(0) && !discrete.contains(5), "discrete rejects OOB")
    var spaceRng = SplitMix64(seed: 1)
    let act = discrete.sample(using: &spaceRng)
    try expect(discrete.contains(act), "discrete sample in range")
    try expect(discrete.shape == nil && discrete.dtype == nil, "discrete non-tensor metadata")

    
    // PR-05: MultiDiscrete + Dict (Swift RNG / RNGBox only)
    let multi = MultiDiscreteSpace(nvec: [2, 3])
    try expect(multi.contains([0, 2]), "multi contains")
    try expect(!multi.contains([0, 3]), "multi OOB")
    var mr = SplitMix64(seed: 2)
    let mv = multi.sample(using: &mr)
    try expect(multi.contains(mv), "multi sample")
    let dict = DictSpace([("a", AnySpace(DiscreteSpace(n: 2))), ("b", AnySpace(DiscreteSpace(n: 2)))])
    let dbox = RNGBox(seed: 3)
    let dv = dict.sample(box: dbox)
    try expect(dict.contains(dv), "dict sample contains")
    // SpaceFlatten / MultiBinary MLX paths need metallib — tier-1 XCTest only.

    
    // PR-06: EnvSpec (no live env on CLI — unit tests cover Environment / AnyEnvironment)
    let espec = EnvSpec(id: "Smoke-v0", maxEpisodeSteps: 10, version: 1)
    try expect(espec.id == "Smoke-v0" && espec.maxEpisodeSteps == 10, "EnvSpec fields")
    try expect(espec.nondeterministic == false, "EnvSpec default deterministic")

    print("RLXCoreSmoke: all checks passed (rlx-swift \(RLXCore.version), RLXCore+MLX linked; PR-02..PR-06 OK)")
    exit(0)
} catch {
    let message = "RLXCoreSmoke FAILED: \(error)\n"
    if let data = message.data(using: .utf8) {
        try? FileHandle.standardError.write(contentsOf: data)
    } else {
        print(message, terminator: "")
    }
    exit(1)
}
