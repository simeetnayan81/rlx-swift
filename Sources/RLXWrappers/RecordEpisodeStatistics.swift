// RecordEpisodeStatistics — episode return/length in info (design.md §14.1, §15.2, PR-08).

import Foundation
import RLXCore

/// Accumulates reward and length since ``reset``; on episode end (`done`), writes a nested
/// ``InfoKeys/episode`` bag with ``InfoKeys/episodeReturn`` (`r`) and ``InfoKeys/episodeLength`` (`l`).
///
/// When ``recordTime`` is true, also writes ``InfoKeys/episodeTime`` (`t`) as wall-clock seconds.
public final class RecordEpisodeStatistics<Inner: Environment>: Environment, EnvironmentWrapper {
    public typealias Observation = Inner.Observation
    public typealias Action = Inner.Action
    public typealias ObservationSpace = Inner.ObservationSpace
    public typealias ActionSpace = Inner.ActionSpace

    public let inner: Inner
    /// When true, include wall-clock seconds under ``InfoKeys/episodeTime``.
    public let recordTime: Bool

    private var episodeReturn: Float = 0
    private var episodeLength: Int = 0
    private var episodeStart: Date?

    public init(_ inner: Inner, recordTime: Bool = false) {
        self.inner = inner
        self.recordTime = recordTime
    }

    public var observationSpace: ObservationSpace { inner.observationSpace }
    public var actionSpace: ActionSpace { inner.actionSpace }
    public var spec: EnvSpec? { inner.spec }

    public var unwrapped: AnyEnvironment {
        AnyEnvironment(inner)
    }

    public func reset(
        seed: UInt64?,
        options: (any ResetOptions)?
    ) throws -> ResetResult<Observation> {
        let result = try inner.reset(seed: seed, options: options)
        episodeReturn = 0
        episodeLength = 0
        if recordTime {
            episodeStart = Date()
        } else {
            episodeStart = nil
        }
        return result
    }

    public func step(_ action: Action) throws -> StepResult<Observation> {
        var result = try inner.step(action)
        episodeReturn += result.reward
        episodeLength += 1
        if result.done {
            var metrics = Info()
            metrics[InfoKeys.episodeReturn] = .double(Double(episodeReturn))
            metrics[InfoKeys.episodeLength] = .int(episodeLength)
            if recordTime, let start = episodeStart {
                metrics[InfoKeys.episodeTime] = .double(Date().timeIntervalSince(start))
            }
            result.info[InfoKeys.episode] = .nested(metrics)
        }
        return result
    }

    public func close() throws {
        try inner.close()
    }
}
