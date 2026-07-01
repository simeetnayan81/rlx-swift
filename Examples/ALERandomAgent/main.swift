// ALERandomAgent — play any Atari ROM with a random policy; verify screen is rendered.
//
//   ./scripts/build-ale.sh ~/.local/ale
//   export ALE_ROOT=$HOME/.local/ale
//   export ALE_ROM_PATH=/path/to/pong.bin   # any Atari .bin
//   swift build --product ALERandomAgent && swift run ALERandomAgent

import Foundation
import MLX
import RLXALE
import RLXCore
import RLXWrappers

func main() throws {
    // CLI builds lack the Metal default library; force CPU before any MLX use.
    // (TaskLocal default is GPU and would otherwise crash on first stream access.)
    Device.setDefault(device: .cpu)

    guard RLXALE.isALELinked else {
        print("""
        ALERandomAgent: ALE C++ is not linked (stub build).

        1. ./scripts/build-ale.sh ~/.local/ale
        2. export ALE_ROOT=$HOME/.local/ale
        3. export ALE_ROM_PATH=/path/to/any_game.bin
        4. swift build --product ALERandomAgent
        5. swift run ALERandomAgent
        """)
        exit(0)
    }

    let romPath: String
    if let path = ProcessInfo.processInfo.environment["ALE_ROM_PATH"], !path.isEmpty {
        romPath = path
    } else if let dir = ProcessInfo.processInfo.environment["ALE_ROM_DIR"], !dir.isEmpty {
        let game = ProcessInfo.processInfo.environment["ALE_GAME"] ?? "pong"
        romPath = try ALEConfig.resolveROMPath(game: game, directory: dir)
    } else {
        fputs("Set ALE_ROM_PATH or ALE_ROM_DIR (+ optional ALE_GAME)\n", stderr)
        exit(2)
    }

    let wantRGB = ProcessInfo.processInfo.environment["ALE_RGB"] == "1"
    let maxSteps = Int(ProcessInfo.processInfo.environment["ALE_MAX_STEPS"] ?? "500") ?? 500
    let outFrame = ProcessInfo.processInfo.environment["ALE_FRAME_OUT"]
        ?? FileManager.default.temporaryDirectory.appendingPathComponent("rlx_ale_frame.ppm").path

    try Device.withDefaultDevice(.cpu) {
        let ale = try ALEEnvironment(
            config: ALEConfig(
                romPath: romPath,
                observationType: wantRGB ? .rgb : .grayscale,
                frameSkip: 4,
                seed: 42,
                displayScreen: ProcessInfo.processInfo.environment["ALE_DISPLAY"] == "1"
            )
        )
        let env = OrderEnforcing(ale)

        print("ALE linked: true")
        print("ROM: \(romPath)")
        print("spec: \(env.spec?.id ?? "?")")
        print("screen: \(ale.screenHeight)x\(ale.screenWidth) actions=\(ale.minimalActionCount)")

        let reset = try env.reset(seed: 42 as UInt64?, options: nil)
        let resetPixels = try ale.copyGrayscaleFrame()
        let resetStats = pixelStats(resetPixels)
        print("reset frame: mlx_shape=\(reset.observation.shape) \(resetStats)")

        let rgb = try ale.copyRGBFrame()
        try writePPM(path: outFrame, width: ale.screenWidth, height: ale.screenHeight, rgb: rgb)
        print("wrote RGB frame: \(outFrame)")

        guard resetStats.nonzero > 0, resetStats.unique >= 2 else {
            fputs(
                "FAIL: screen blank — rendering not working (nonzero=\(resetStats.nonzero) unique=\(resetStats.unique))\n",
                stderr
            )
            exit(1)
        }
        print(
            "RENDER_OK: nonzero=\(resetStats.nonzero) unique=\(resetStats.unique) "
                + "min=\(resetStats.minV) max=\(resetStats.maxV) mean=\(String(format: "%.2f", resetStats.mean))"
        )

        var rng = SplitMix64(seed: 7)
        var total: Float = 0
        var steps = 0
        while steps < maxSteps {
            let action = env.actionSpace.sample(using: &rng)
            let step = try env.step(action)
            total += step.reward
            steps += 1
            if step.done {
                print("episode done at step=\(steps) terminated=\(step.terminated)")
                break
            }
        }

        let after = try ale.copyGrayscaleFrame()
        let afterStats = pixelStats(after)
        let changed = zip(resetPixels, after).reduce(0) { $0 + ($1.0 != $1.1 ? 1 : 0) }
        print("after steps=\(steps) return=\(total) \(afterStats) changed_vs_reset=\(changed)")

        if afterStats.nonzero == 0 {
            fputs("FAIL: post-step screen blank\n", stderr)
            exit(1)
        }
        if changed == 0 && steps > 10 {
            print("WARN: frame identical after \(steps) steps (possible for some no-op policies)")
        }

        print("ALERandomAgent: OK")
        try env.close()
    }
}

struct PixelStats {
    var nonzero: Int
    var unique: Int
    var mean: Double
    var minV: UInt8
    var maxV: UInt8
}

func pixelStats(_ pixels: [UInt8]) -> PixelStats {
    var hist = [Int](repeating: 0, count: 256)
    var sum = 0
    var nonzero = 0
    var minV: UInt8 = 255
    var maxV: UInt8 = 0
    for p in pixels {
        hist[Int(p)] += 1
        sum += Int(p)
        if p != 0 { nonzero += 1 }
        minV = min(minV, p)
        maxV = max(maxV, p)
    }
    let unique = hist.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }
    let mean = pixels.isEmpty ? 0 : Double(sum) / Double(pixels.count)
    return PixelStats(nonzero: nonzero, unique: unique, mean: mean, minV: minV, maxV: maxV)
}

func writePPM(path: String, width: Int, height: Int, rgb: [UInt8]) throws {
    var data = Data()
    data.append(contentsOf: "P6\n\(width) \(height)\n255\n".utf8)
    data.append(contentsOf: rgb.prefix(width * height * 3))
    try data.write(to: URL(fileURLWithPath: path))
}

do {
    try main()
} catch {
    fputs("ALERandomAgent error: \(error)\n", stderr)
    exit(1)
}
