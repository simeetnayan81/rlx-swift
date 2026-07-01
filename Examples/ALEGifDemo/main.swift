// ALEGifDemo — run any Atari ROM via ALEEnvironment and dump RGB frames for a GIF.
//
//   export ALE_ROOT=... ALE_ROM_PATH=.../pong.bin
//   ./scripts/make-ale-gif.sh
//
// Frames: /tmp/rlx_ale_gif/frame_XXXX.ppm → /tmp/rlx_ale_pong.gif (via ffmpeg)

import Foundation
import MLX
import RLXALE
import RLXCore

Device.setDefault(device: .cpu)

guard RLXALE.isALELinked else {
    fputs("ALE not linked. Set ALE_ROOT and rebuild (see docs/ale-adapter-design.md).\n", stderr)
    exit(2)
}

let rom: String
if let p = ProcessInfo.processInfo.environment["ALE_ROM_PATH"], !p.isEmpty {
    rom = p
} else if let dir = ProcessInfo.processInfo.environment["ALE_ROM_DIR"], !dir.isEmpty {
    let game = ProcessInfo.processInfo.environment["ALE_GAME"] ?? "pong"
    rom = try ALEConfig.resolveROMPath(game: game, directory: dir)
} else {
    fputs("Set ALE_ROM_PATH or ALE_ROM_DIR\n", stderr)
    exit(2)
}

let steps = Int(ProcessInfo.processInfo.environment["ALE_GIF_STEPS"] ?? "120") ?? 120
let frameEvery = max(1, Int(ProcessInfo.processInfo.environment["ALE_GIF_EVERY"] ?? "2") ?? 2)
let outDir = ProcessInfo.processInfo.environment["ALE_GIF_DIR"]
    ?? FileManager.default.temporaryDirectory.appendingPathComponent("rlx_ale_gif").path

try? FileManager.default.removeItem(atPath: outDir)
try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

try Device.withDefaultDevice(.cpu) {
    let env = try ALEEnvironment(
        config: ALEConfig(
            romPath: rom,
            observationType: .rgb,
            frameSkip: 4,
            seed: 0
        )
    )
    _ = try env.reset(seed: 0 as UInt64?, options: nil)

    var rng = SplitMix64(seed: 1)
    var saved = 0
    var i = 0
    while i < steps {
        if i % frameEvery == 0 {
            let rgb = try env.copyRGBFrame()
            let path = (outDir as NSString).appendingPathComponent(String(format: "frame_%04d.ppm", saved))
            try writePPM(path: path, width: env.screenWidth, height: env.screenHeight, rgb: rgb)
            saved += 1
        }
        let action = env.actionSpace.sample(using: &rng)
        let step = try env.step(action)
        i += 1
        if step.done {
            _ = try env.reset(seed: nil, options: nil)
        }
    }
    try env.close()
    print("ALEGifDemo: wrote \(saved) frames to \(outDir)")
    print("ALEGifDemo: game=\(env.spec?.id ?? "?") size=\(env.screenWidth)x\(env.screenHeight)")
}

func writePPM(path: String, width: Int, height: Int, rgb: [UInt8]) throws {
    var data = Data()
    data.append(contentsOf: "P6\n\(width) \(height)\n255\n".utf8)
    data.append(contentsOf: rgb.prefix(width * height * 3))
    try data.write(to: URL(fileURLWithPath: path))
}
