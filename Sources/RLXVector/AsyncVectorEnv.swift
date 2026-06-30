// AsyncVectorEnv — concurrent vector of AnyEnvironment (design.md §16.3, PR-14).

import RLXCore
import RLXWrappers

/// Steps `numEnvs` independent ``AnyEnvironment`` instances using Swift concurrency.
///
/// - **Public API:** `async throws` (`reset` / `step` / `close`). Actor-isolated so only one
///   batch operation runs at a time; sub-envs within a batch may run in parallel.
/// - **Ordering:** results are always assembled in fixed slot index order `0..<numEnvs`,
///   regardless of completion order of worker tasks.
/// - **Autoreset:** same policy as ``SyncVectorEnv`` (see ``AutoresetMode``; default `.sameStep`).
/// - **Seeding:** `reset(seed:)` uses `Seed(seed).child(index: i)` per slot when `seed != nil`.
/// - **Concurrency:** `maxConcurrency` limits in-flight slot work (`1` = serial / deterministic tests).
/// - **Cancellation:** ``close()`` cancels any outstanding reset/step work, then closes sub-envs.
public actor AsyncVectorEnv {
    public let numEnvs: Int
    public let autoresetMode: AutoresetMode
    /// Maximum number of sub-env tasks in flight for one batch (`1...numEnvs`).
    public let maxConcurrency: Int

    private var envs: [AnyEnvironment]
    /// For ``AutoresetMode/nextStep``: slot must reset before the next step action.
    private var pendingReset: [Bool]
    private var isClosed = false
    /// Cancels the in-flight batch task (if any) from ``close()``.
    private var cancelInFlight: (@Sendable () -> Void)?

    /// - Parameters:
    ///   - numEnvs: Number of parallel logical environments (must be > 0).
    ///   - autoresetMode: Episode boundary policy (immutable).
    ///   - maxConcurrency: Cap on concurrent slot work; defaults to `numEnvs`. Use `1` for serial tests.
    ///   - makeEnv: Factory invoked `numEnvs` times (fresh instance each call). Must be `@Sendable`.
    public init(
        numEnvs: Int,
        autoresetMode: AutoresetMode = .sameStep,
        maxConcurrency: Int? = nil,
        makeEnv: @Sendable () -> AnyEnvironment
    ) {
        precondition(numEnvs > 0, "numEnvs must be > 0")
        let limit = maxConcurrency ?? numEnvs
        precondition(limit > 0 && limit <= numEnvs, "maxConcurrency must be in 1...numEnvs")
        self.numEnvs = numEnvs
        self.autoresetMode = autoresetMode
        self.maxConcurrency = limit
        self.envs = (0..<numEnvs).map { _ in makeEnv() }
        self.pendingReset = Array(repeating: false, count: numEnvs)
    }

    /// Convenience: factory takes a slot index (for varied config).
    public init(
        numEnvs: Int,
        autoresetMode: AutoresetMode = .sameStep,
        maxConcurrency: Int? = nil,
        makeEnv: @Sendable (Int) -> AnyEnvironment
    ) {
        precondition(numEnvs > 0, "numEnvs must be > 0")
        let limit = maxConcurrency ?? numEnvs
        precondition(limit > 0 && limit <= numEnvs, "maxConcurrency must be in 1...numEnvs")
        self.numEnvs = numEnvs
        self.autoresetMode = autoresetMode
        self.maxConcurrency = limit
        self.envs = (0..<numEnvs).map { makeEnv($0) }
        self.pendingReset = Array(repeating: false, count: numEnvs)
    }

    public var singleObservationSpace: AnySpace { envs[0].observationSpace }
    public var singleActionSpace: AnySpace { envs[0].actionSpace }

    /// Reset all sub-environments (child seeds when `seed != nil`).
    public func reset(
        seed: UInt64? = nil,
        options: (any ResetOptions)? = nil
    ) async throws -> VectorResetResult {
        try ensureOpen()
        let envsSnapshot = envs
        let n = numEnvs
        let limit = maxConcurrency
        // ResetOptions is not formally Sendable; exclusive to this batch via actor + unchecked box.
        let optionsBox = UncheckedBox(options)

        let pairs: [ResetPair] = try await runCancellable {
            try await Self.mapSlots(count: n, maxConcurrency: limit) { i in
                try Task.checkCancellation()
                let slotSeed: UInt64?
                if let seed {
                    slotSeed = Seed(seed).child(index: i).rawValue
                } else {
                    slotSeed = nil
                }
                let r = try envsSnapshot[i].reset(seed: slotSeed, options: optionsBox.value)
                return ResetPair(observation: r.observation, info: r.info)
            }
        }

        pendingReset = Array(repeating: false, count: numEnvs)
        return VectorResetResult(
            observations: pairs.map(\.observation),
            infos: pairs.map(\.info)
        )
    }

    /// Step all environments with one action per slot (length must equal ``numEnvs``).
    public func step(_ actions: [Any]) async throws -> VectorStepResult {
        try ensureOpen()
        guard actions.count == numEnvs else {
            throw VectorEnvironmentError.batchSizeMismatch(expected: numEnvs, actual: actions.count)
        }

        let envsSnapshot = envs
        let pendingSnapshot = pendingReset
        let n = numEnvs
        let limit = maxConcurrency
        let autoreset = autoresetMode
        // `Any` actions are not Sendable; each index is read once by its slot task only.
        let actionBoxes = actions.map { UncheckedBox($0) }

        let slotResults: [SlotStepOutcome] = try await runCancellable {
            try await Self.mapSlots(count: n, maxConcurrency: limit) { i in
                try Task.checkCancellation()
                return try Self.stepSlot(
                    env: envsSnapshot[i],
                    action: actionBoxes[i].value,
                    autoresetMode: autoreset,
                    needsPendingReset: pendingSnapshot[i]
                )
            }
        }

        var observations: [Any] = []
        var rewards: [Float] = []
        var terminateds: [Bool] = []
        var truncateds: [Bool] = []
        var infos: [Info] = []
        observations.reserveCapacity(n)
        rewards.reserveCapacity(n)
        terminateds.reserveCapacity(n)
        truncateds.reserveCapacity(n)
        infos.reserveCapacity(n)

        var newPending = pendingReset
        for (i, slot) in slotResults.enumerated() {
            observations.append(slot.observation)
            rewards.append(slot.reward)
            terminateds.append(slot.terminated)
            truncateds.append(slot.truncated)
            infos.append(slot.info)
            newPending[i] = slot.pendingReset
        }
        pendingReset = newPending

        return VectorStepResult(
            observations: observations,
            rewards: rewards,
            terminateds: terminateds,
            truncateds: truncateds,
            infos: infos
        )
    }

    /// Cancel in-flight work (if any), then close all sub-environments.
    ///
    /// Idempotent: subsequent calls are no-ops after the first successful close sequence.
    public func close() async throws {
        guard !isClosed else { return }
        isClosed = true
        cancelInFlight?()
        cancelInFlight = nil
        // Yield so a cancelled batch can observe cancellation before we close sub-envs.
        await Task.yield()
        var firstError: Error?
        for env in envs {
            do {
                try env.close()
            } catch {
                if firstError == nil { firstError = error }
            }
        }
        if let firstError { throw firstError }
    }

    // MARK: - Internals

    private func ensureOpen() throws {
        if isClosed { throw VectorEnvironmentError.closed }
    }

    /// Run work in a cancellable child task tracked for ``close()``.
    private func runCancellable<T: Sendable>(_ body: @Sendable @escaping () async throws -> T) async throws -> T {
        try ensureOpen()
        let task = Task<T, Error> { try await body() }
        cancelInFlight = { task.cancel() }
        defer { cancelInFlight = nil }
        do {
            let value = try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            if isClosed { throw VectorEnvironmentError.cancelled }
            return value
        } catch is CancellationError {
            throw VectorEnvironmentError.cancelled
        }
    }

    /// Map indices `0..<count` with at most `maxConcurrency` tasks in flight; preserve order.
    private static func mapSlots<T: Sendable>(
        count: Int,
        maxConcurrency: Int,
        operation: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T] {
        if count == 0 { return [] }
        var results = [T?](repeating: nil, count: count)
        try await withThrowingTaskGroup(of: (Int, T).self) { group in
            var next = 0
            var inFlightCount = 0

            func enqueue(_ index: Int) {
                group.addTask {
                    let value = try await operation(index)
                    return (index, value)
                }
            }

            while next < count && inFlightCount < maxConcurrency {
                enqueue(next)
                next += 1
                inFlightCount += 1
            }

            while let (index, value) = try await group.next() {
                results[index] = value
                inFlightCount -= 1
                if next < count {
                    enqueue(next)
                    next += 1
                    inFlightCount += 1
                }
            }
        }
        return results.map { $0! }
    }

    /// `@unchecked Sendable` carrier for type-erased payloads (actions / options / obs).
    private struct UncheckedBox<T>: @unchecked Sendable {
        let value: T
        init(_ value: T) { self.value = value }
    }

    private struct ResetPair: @unchecked Sendable {
        let observation: Any
        let info: Info
    }

    private struct SlotStepOutcome: @unchecked Sendable {
        let observation: Any
        let reward: Float
        let terminated: Bool
        let truncated: Bool
        let info: Info
        let pendingReset: Bool
    }

    /// Single-slot step + autoreset (mirrors ``SyncVectorEnv`` semantics).
    private static func stepSlot(
        env: AnyEnvironment,
        action: Any,
        autoresetMode: AutoresetMode,
        needsPendingReset: Bool
    ) throws -> SlotStepOutcome {
        var pendingAfter = false
        if autoresetMode == .nextStep, needsPendingReset {
            _ = try env.reset(seed: nil, options: nil)
        }

        let step = try env.step(action)
        var obs = step.observation
        var info = step.info
        let terminated = step.terminated
        let truncated = step.truncated
        let done = terminated || truncated

        if done {
            switch autoresetMode {
            case .disabled:
                break
            case .nextStep:
                pendingAfter = true
            case .sameStep:
                FinalObservationInfo.embed(step.observation, into: &info)
                info[InfoKeys.finalInfo] = .nested(step.info)
                let r = try env.reset(seed: nil, options: nil)
                obs = r.observation
            }
        }

        return SlotStepOutcome(
            observation: obs,
            reward: step.reward,
            terminated: terminated,
            truncated: truncated,
            info: info,
            pendingReset: pendingAfter
        )
    }
}
