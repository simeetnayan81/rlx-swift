import Foundation
import MLX
import RLXCore
import XCTest

/// Unit tests for PR-02 core result types, `Info` key access, and error equality.
///
/// Prefer `xcodebuild` (Metal shaders) for tier-1:
///   DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
///     ./scripts/xcodebuild-test.sh
final class CoreResultTypesTests: XCTestCase {

    override func setUp() {
        super.setUp()
        _ = Device.withDefaultDevice(.cpu) { () -> Void in }
    }

    // MARK: - Info key access

    func testInfoDefaultIsEmpty() {
        let info = Info()
        XCTAssertTrue(info.isEmpty)
        XCTAssertEqual(info.count, 0)
        XCTAssertNil(info["missing"])
        XCTAssertTrue(Array(info.keys).isEmpty)
    }

    func testInfoSubscriptGetSetAndRemove() {
        var info = Info()
        info["TimeLimit.truncated"] = .bool(true)
        info["score"] = .double(1.5)
        info["label"] = .string("ok")
        info["steps"] = .int(3)

        XCTAssertEqual(info["TimeLimit.truncated"], .bool(true))
        XCTAssertEqual(info["score"], .double(1.5))
        XCTAssertEqual(info["label"], .string("ok"))
        XCTAssertEqual(info["steps"], .int(3))
        XCTAssertEqual(info.count, 4)
        XCTAssertFalse(info.isEmpty)

        let removed = info.removeValue(forKey: "score")
        XCTAssertEqual(removed, .double(1.5))
        XCTAssertNil(info["score"])
        XCTAssertEqual(info.count, 3)

        info["label"] = nil
        XCTAssertNil(info["label"])
        XCTAssertEqual(info.count, 2)
    }

    func testInfoSetHelper() {
        var info = Info()
        info.set("a", .int(1))
        info.set("b", .bool(false))
        XCTAssertEqual(info["a"], .int(1))
        XCTAssertEqual(info["b"], .bool(false))
    }

    func testInfoNestedValueAndEquality() {
        var episode = Info()
        episode["r"] = .double(12.0)
        episode["l"] = .int(8)

        var info = Info()
        info["episode"] = .nested(episode)

        guard case .nested(let nested)? = info["episode"] else {
            return XCTFail("expected nested episode info")
        }
        XCTAssertEqual(nested["r"], .double(12.0))
        XCTAssertEqual(nested["l"], .int(8))

        var info2 = Info()
        info2["episode"] = .nested(episode)
        XCTAssertEqual(info, info2)

        info2["episode"] = .nested(Info(["r": .double(0)]))
        XCTAssertNotEqual(info, info2)
    }

    func testInfoDeepNestedEquality() {
        let inner = Info(["t": .double(0.01)])
        let mid = Info(["episode": .nested(inner), "flag": .bool(true)])
        let a = Info(["wrap": .nested(mid)])
        let b = Info(["wrap": .nested(Info([
            "episode": .nested(Info(["t": .double(0.01)])),
            "flag": .bool(true),
        ]))])
        XCTAssertEqual(a, b)
    }

    func testInfoMergeOverwritesOnConflict() {
        var base = Info(["a": .int(1), "b": .string("keep")])
        let other = Info(["a": .int(2), "c": .bool(true)])
        base.merge(other)
        XCTAssertEqual(base["a"], .int(2))
        XCTAssertEqual(base["b"], .string("keep"))
        XCTAssertEqual(base["c"], .bool(true))

        let merged = Info(["x": .int(1)]).merging(Info(["x": .int(9), "y": .int(2)]))
        XCTAssertEqual(merged["x"], .int(9))
        XCTAssertEqual(merged["y"], .int(2))
        // Original unchanged
        XCTAssertEqual(Info(["x": .int(1)])["x"], .int(1))
    }

    func testInfoValueCaseInequalityAcrossKinds() {
        XCTAssertNotEqual(InfoValue.bool(true), .int(1))
        XCTAssertNotEqual(InfoValue.int(1), .double(1.0))
        XCTAssertNotEqual(InfoValue.string("1"), .int(1))
        XCTAssertNotEqual(InfoValue.nested(Info()), .bool(false))
    }

    func testInfoArrayValueEqualityOnCPU() {
        Device.withDefaultDevice(.cpu) {
            let a = MLXArray([1.0, 2.0, 3.0] as [Float])
            let b = MLXArray([1.0, 2.0, 3.0] as [Float])
            let c = MLXArray([1.0, 2.0, 4.0] as [Float])
            let d = MLXArray([1.0, 2.0] as [Float]) // different shape
            eval(a, b, c, d)

            let infoA = Info(["final_observation": .array(a)])
            let infoB = Info(["final_observation": .array(b)])
            let infoC = Info(["final_observation": .array(c)])
            let infoD = Info(["final_observation": .array(d)])

            XCTAssertEqual(infoA, infoB)
            XCTAssertNotEqual(infoA, infoC)
            XCTAssertNotEqual(infoA, infoD)
        }
    }

    func testInfoArrayInt32EqualityOnCPU() {
        Device.withDefaultDevice(.cpu) {
            let a = MLXArray([1, 2, 3] as [Int32])
            let b = MLXArray([1, 2, 3] as [Int32])
            let c = MLXArray([1, 2, 9] as [Int32])
            eval(a, b, c)
            XCTAssertEqual(InfoValue.array(a), .array(b))
            XCTAssertNotEqual(InfoValue.array(a), .array(c))
        }
    }

    func testInfoDictionaryInitializer() {
        let info = Info([
            "TimeLimit.truncated": .bool(true),
            "note": .string("limit"),
        ])
        XCTAssertEqual(info.count, 2)
        XCTAssertEqual(info["TimeLimit.truncated"], .bool(true))
        XCTAssertTrue(Set(info.keys).isSuperset(of: ["TimeLimit.truncated", "note"]))
    }

    // MARK: - ResetResult / StepResult

    func testResetResultDefaultsEmptyInfo() {
        let result = ResetResult(observation: 42)
        XCTAssertEqual(result.observation, 42)
        XCTAssertTrue(result.info.isEmpty)
    }

    func testResetResultEquality() {
        let a = ResetResult(observation: "obs", info: Info(["k": .int(1)]))
        let b = ResetResult(observation: "obs", info: Info(["k": .int(1)]))
        let c = ResetResult(observation: "other")
        let d = ResetResult(observation: "obs", info: Info(["k": .int(2)]))
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
        XCTAssertNotEqual(a, d)
    }

    func testResetResultMutableFields() {
        var result = ResetResult(observation: 0)
        result.observation = 7
        result.info["seeded"] = .bool(true)
        XCTAssertEqual(result.observation, 7)
        XCTAssertEqual(result.info["seeded"], .bool(true))
    }

    func testStepResultFieldsAndDone() {
        let continuing = StepResult(
            observation: [0.0],
            reward: 1.0,
            terminated: false,
            truncated: false
        )
        XCTAssertFalse(continuing.done)
        XCTAssertFalse(continuing.terminated)
        XCTAssertFalse(continuing.truncated)
        XCTAssertEqual(continuing.reward, 1.0 as Float)
        XCTAssertTrue(continuing.info.isEmpty)

        let terminated = StepResult(
            observation: [1.0],
            reward: 0,
            terminated: true,
            truncated: false,
            info: Info(["reason": .string("goal")])
        )
        XCTAssertTrue(terminated.done)
        XCTAssertTrue(terminated.terminated)
        XCTAssertEqual(terminated.info["reason"], .string("goal"))

        let truncated = StepResult(
            observation: [2.0],
            reward: -0.1,
            terminated: false,
            truncated: true
        )
        XCTAssertTrue(truncated.done)
        XCTAssertTrue(truncated.truncated)

        // Both flags discouraged but still "done"
        let both = StepResult(
            observation: 0,
            reward: 0,
            terminated: true,
            truncated: true
        )
        XCTAssertTrue(both.done)
    }

    func testStepResultEquality() {
        let a = StepResult(observation: 1, reward: 0.5, terminated: false, truncated: true)
        let b = StepResult(observation: 1, reward: 0.5, terminated: false, truncated: true)
        let c = StepResult(observation: 1, reward: 0.5, terminated: true, truncated: false)
        let d = StepResult(observation: 1, reward: 0.25, terminated: false, truncated: true)
        let e = StepResult(
            observation: 1,
            reward: 0.5,
            terminated: false,
            truncated: true,
            info: Info(["x": .int(1)])
        )
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
        XCTAssertNotEqual(a, d)
        XCTAssertNotEqual(a, e)
    }

    func testStepResultRewardIsFloat32Semantics() {
        // Reward is locked to Float (binary32), not Double — compile-time type lock + value check.
        let step = StepResult(observation: 0, reward: Float(0.1) + Float(0.2), terminated: false, truncated: false)
        let reward: Float = step.reward
        XCTAssertEqual(reward, Float(0.1) + Float(0.2))
        let explicit = StepResult(observation: 0, reward: Float(1.25), terminated: false, truncated: false)
        XCTAssertEqual(explicit.reward, 1.25 as Float)
    }

    // MARK: - EnvironmentError equality

    func testEnvironmentErrorEqualityForSimpleCases() {
        XCTAssertEqual(EnvironmentError.notReset, .notReset)
        XCTAssertEqual(EnvironmentError.episodeEnded, .episodeEnded)
        XCTAssertEqual(EnvironmentError.closed, .closed)
        XCTAssertEqual(EnvironmentError.renderNotSupported, .renderNotSupported)

        XCTAssertNotEqual(EnvironmentError.notReset, .closed)
        XCTAssertNotEqual(EnvironmentError.episodeEnded, .notReset)
        XCTAssertNotEqual(EnvironmentError.renderNotSupported, .closed)
    }

    func testEnvironmentErrorEqualityForAssociatedStrings() {
        XCTAssertEqual(
            EnvironmentError.invalidAction("oob"),
            .invalidAction("oob")
        )
        XCTAssertNotEqual(
            EnvironmentError.invalidAction("oob"),
            .invalidAction("other")
        )
        XCTAssertEqual(
            EnvironmentError.invalidObservation("shape"),
            .invalidObservation("shape")
        )
        XCTAssertNotEqual(
            EnvironmentError.invalidObservation("shape"),
            .invalidObservation("dtype")
        )
        XCTAssertEqual(
            EnvironmentError.configuration("bad dt"),
            .configuration("bad dt")
        )
        XCTAssertNotEqual(
            EnvironmentError.configuration("a"),
            .configuration("b")
        )
        // Different cases with same payload string are still unequal
        XCTAssertNotEqual(
            EnvironmentError.invalidAction("x"),
            .configuration("x")
        )
    }

    func testEnvironmentErrorUnderlyingEqualityIsCaseOnly() {
        struct E1: Error {}
        struct E2: Error {}
        // Nested errors are not compared; both `.underlying` are equal by design.
        XCTAssertEqual(
            EnvironmentError.underlying(E1()),
            .underlying(E2())
        )
        XCTAssertNotEqual(
            EnvironmentError.underlying(E1()),
            .closed
        )
    }

    func testEnvironmentErrorDescriptionIsStable() {
        XCTAssertTrue(EnvironmentError.notReset.description.contains("notReset"))
        XCTAssertTrue(EnvironmentError.episodeEnded.description.contains("episodeEnded"))
        XCTAssertTrue(EnvironmentError.invalidAction("x").description.contains("x"))
        XCTAssertTrue(EnvironmentError.invalidObservation("y").description.contains("y"))
        XCTAssertTrue(EnvironmentError.configuration("z").description.contains("z"))
        XCTAssertTrue(EnvironmentError.closed.description.contains("closed"))
        XCTAssertTrue(EnvironmentError.renderNotSupported.description.contains("renderNotSupported"))
        struct E: Error {}
        XCTAssertTrue(EnvironmentError.underlying(E()).description.contains("underlying"))
    }

    func testEnvironmentErrorIsThrownAndCaught() {
        func fail() throws {
            throw EnvironmentError.notReset
        }
        XCTAssertThrowsError(try fail()) { error in
            XCTAssertEqual(error as? EnvironmentError, .notReset)
        }
    }

    // MARK: - RenderMode / RenderFrame

    func testRenderModeCasesAndCodable() throws {
        XCTAssertEqual(RenderMode.allCases.count, 4)
        XCTAssertEqual(Set(RenderMode.allCases), [.none, .human, .rgbArray, .ansi])
        XCTAssertEqual(RenderMode.none.rawValue, "none")
        XCTAssertEqual(RenderMode.human.rawValue, "human")
        XCTAssertEqual(RenderMode.rgbArray.rawValue, "rgbArray")
        XCTAssertEqual(RenderMode.ansi.rawValue, "ansi")

        for mode in RenderMode.allCases {
            let encoded = try JSONEncoder().encode(mode)
            let decoded = try JSONDecoder().decode(RenderMode.self, from: encoded)
            XCTAssertEqual(decoded, mode)
        }
    }

    func testRenderFrameEquality() {
        XCTAssertEqual(RenderFrame.ansi("hi"), .ansi("hi"))
        XCTAssertNotEqual(RenderFrame.ansi("hi"), .ansi("bye"))
        XCTAssertEqual(RenderFrame.humanDisplayed, .humanDisplayed)
        XCTAssertNotEqual(RenderFrame.ansi("hi"), .humanDisplayed)

        Device.withDefaultDevice(.cpu) {
            let pixels = MLXArray.zeros([2, 2, 3], type: UInt8.self)
            let pixels2 = MLXArray.zeros([2, 2, 3], type: UInt8.self)
            let other = MLXArray.ones([2, 2, 3], type: UInt8.self)
            eval(pixels, pixels2, other)
            XCTAssertEqual(RenderFrame.rgb(pixels), .rgb(pixels2))
            XCTAssertNotEqual(RenderFrame.rgb(pixels), .rgb(other))
            XCTAssertNotEqual(RenderFrame.rgb(pixels), .ansi("x"))
        }
    }
}
