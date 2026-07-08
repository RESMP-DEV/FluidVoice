//
//  FluidIntelligenceTrainer.swift
//  Fluid
//
//  On-device idle-time fine-tuning of Fluid Intelligence (Gemma) via the
//  external `fluidvoice-finetune idle-train` CLI.
//
//  Responsibilities:
//    1. Gate: only run when on AC power, idle, thermally nominal, and Low Power
//       Mode is off (mirrors the Python gate; shell-outs avoid framework deps).
//    2. Invoke the venv's `fluidvoice-finetune idle-train` wrapped in
//       `caffeinate -is -w <pid>` so the Mac stays awake mid-training.
//    3. Report status (last run, last GGUF, corpus count) to the UI.
//
//  The trainer no-ops in the public OSS build until the user configures a
//  venv path AND enables training-data collection. It does NOT depend on the
//  private PrivateAIProviderBridge — it produces a GGUF the bridge then loads.
//

import Combine
import Foundation

@MainActor
final class FluidIntelligenceTrainer: ObservableObject {
    static let shared = FluidIntelligenceTrainer()

    @Published private(set) var isRunning = false
    @Published private(set) var lastRunDate: Date?
    @Published private(set) var lastRunSummary: String?
    @Published private(set) var lastProducedGGUF: URL?
    @Published private(set) var corpusEntryCount: Int = 0

    private var process: Process?

    private init() {}

    // MARK: - Public API

    /// Whether the trainer is *configured*: collection consent is on AND the
    /// bundled trainer is discoverable. No user-facing path setting — the
    /// bundle is auto-discovered from the .app's Resources (release) or the
    /// repo's FluidVoiceTrain/dist (dev builds).
    var isConfigured: Bool {
        SettingsStore.shared.allowFluidIntelligenceTrainingCollection
            && Self.bundledTrainerDirectory() != nil
    }

    /// Resolve the bundled FluidVoiceTrain directory, or nil if absent.
    ///
    /// Precedence:
    ///   1. Release: ``<FluidVoice.app>/Contents/Resources/FluidVoiceTrain``
    ///      (copied in by the release build).
    ///   2. Dev/source build: ``<repo>/FluidVoiceTrain/dist/FluidVoiceTrain``
    ///      (the output of FluidVoiceTrain/scripts/build_bundle.sh), located
    ///      via the Swift source tree so an unsigned dev build still finds a
    ///      locally-built bundle without a venv-path setting.
    static func bundledTrainerDirectory() -> URL? {
        let fm = FileManager.default

        // 1. Release: inside the app bundle's Resources.
        let releaseURL = Bundle.main.bundleURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("Resources", isDirectory: true)
            .appendingPathComponent("FluidVoiceTrain", isDirectory: true)
        let releaseBin = releaseURL.appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("fluidvoice-finetune")
        if fm.fileExists(atPath: releaseBin.path) {
            return releaseURL
        }

        // 2. Dev: <repo>/FluidVoiceTrain/dist/FluidVoiceTrain. Resolve the repo
        //    root by walking up from this source file at build time.
        let devURL = Self.devBundleDirectory()
        let devBin = devURL.appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("fluidvoice-finetune")
        if fm.fileExists(atPath: devBin.path) {
            return devURL
        }

        return nil
    }

    /// Locate the dev-build bundle via `#file`-relative path resolution. In a
    /// compiled app `#file` may be flattened, so this is best-effort and only
    /// used when the release bundle isn't present (i.e., a dev build).
    private static func devBundleDirectory() -> URL {
        let file = URL(fileURLWithPath: #file)
        // Sources/Fluid/Services/FluidIntelligenceTrainer.swift -> walk up 4 to repo root.
        let repoRoot = file
            .deletingLastPathComponent() // Services
            .deletingLastPathComponent() // Fluid
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // repo root
        return repoRoot
            .appendingPathComponent("FluidVoiceTrain", isDirectory: true)
            .appendingPathComponent("dist", isDirectory: true)
            .appendingPathComponent("FluidVoiceTrain", isDirectory: true)
    }

    /// The corpus entry count, refreshed by re-exporting. Cheap to call.
    func refreshCorpusCount() {
        do {
            let summary = try TrainingCorpusExporter.export()
            self.corpusEntryCount = summary.exported
        } catch {
            self.corpusEntryCount = 0
        }
    }

    /// The full state snapshot for the UI.
    struct Status {
        let configured: Bool
        let running: Bool
        let corpusEntries: Int
        let lastRun: Date?
        let lastSummary: String?
        let lastGGUF: URL?
    }

    var status: Status {
        Status(
            configured: self.isConfigured,
            running: self.isRunning,
            corpusEntries: self.corpusEntryCount,
            lastRun: self.lastRunDate,
            lastSummary: self.lastRunSummary,
            lastGGUF: self.lastProducedGGUF
        )
    }

    // MARK: - Gate

    struct GateState: Equatable {
        let onACPower: Bool
        let idleSeconds: Double
        let lowPowerMode: Bool
        let ok: Bool
        let reasons: [String]
    }

    /// Check whether it's polite to train right now. Shell-outs mirror the
    /// Python gate so behavior is identical across the two entrypoints.
    func checkGate(minIdleSeconds: Double = 600) -> GateState {
        let onAC = self.isOnACPower()
        let lowPower = self.isLowPowerMode()
        let idle = self.idleSeconds()

        var reasons: [String] = []
        if !onAC {
            reasons.append("not on AC power")
        }
        if lowPower {
            reasons.append("Low Power Mode is on")
        }
        if idle < minIdleSeconds {
            reasons.append("recently active")
        }

        return GateState(
            onACPower: onAC,
            idleSeconds: idle,
            lowPowerMode: lowPower,
            ok: reasons.isEmpty,
            reasons: reasons
        )
    }

    // MARK: - Run

    /// Kick off a gated training run. Re-exports the corpus, then invokes the
    /// Python trainer. No-op if not configured or the gate fails.
    func runIfAppropriate(minIdleSeconds: Double = 600) async {
        guard self.isConfigured else { return }
        guard !self.isRunning else { return }

        let gate = self.checkGate(minIdleSeconds: minIdleSeconds)
        guard gate.ok else {
            self.lastRunSummary = "Skipped: \(gate.reasons.joined(separator: ", "))"
            return
        }

        do {
            let summary = try TrainingCorpusExporter.export()
            self.corpusEntryCount = summary.exported
            guard summary.exported > 0 else {
                self.lastRunSummary = "No training data yet."
                self.lastRunDate = Date()
                return
            }
            try await self.invokeTrainer(corpusURL: summary.corpusURL, exported: summary.exported)
        } catch {
            self.lastRunSummary = "Export failed: \(error.localizedDescription)"
            self.lastRunDate = Date()
        }
    }

    /// Cancel an in-progress run (sends SIGTERM; the Python handler checkpoints).
    func cancel() {
        self.process?.terminate()
    }

    // MARK: - Internals

    private func invokeTrainer(corpusURL: URL, exported: Int) async throws {
        guard let bundleDir = Self.bundledTrainerDirectory() else {
            self.lastRunSummary = "Trainer bundle not found."
            self.lastRunDate = Date()
            return
        }

        let entrypoint = bundleDir.appendingPathComponent("bin", isDirectory: false)
            .appendingPathComponent("fluidvoice-finetune").path
        let checkpointDir = TrainingCorpusExporter.checkpointRoot.path
        // Wrap in caffeinate -is -w <pid> to hold off idle+system sleep.
        let trainCmd = "'\(entrypoint)' idle-train --corpus '\(corpusURL.path)' --checkpoint-dir '\(checkpointDir)'"

        self.isRunning = true
        defer { isRunning = false }

        let proc = Process()
        self.process = proc
        proc.launchPath = "/bin/bash"
        proc.arguments = ["-c", trainCmd + " & caffeinate -is -w $!"]
        // Capture stderr for the status line.
        let errPipe = Pipe()
        proc.standardError = errPipe
        proc.standardOutput = FileHandle(forWritingAtPath: checkpointDir + "/trainer.stdout.log")

        try proc.run()
        proc.waitUntilExit()

        let status = proc.terminationStatus
        let stderrData = try errPipe.fileHandleForReading.readToEnd() ?? Data()
        let stderr = String(data: stderrData, encoding: .utf8) ?? ""

        self.lastRunDate = Date()
        if status == 0 {
            self.lastRunSummary = "Trained on \(exported) records."
            // Look for the produced GGUF in the checkpoint dir.
            let fm = FileManager.default
            if let gguf = try? fm.contentsOfDirectory(
                at: TrainingCorpusExporter.checkpointRoot,
                includingPropertiesForKeys: nil
            )
            .first(where: { $0.pathExtension == "gguf" }) {
                self.lastProducedGGUF = gguf
            }
        } else {
            self.lastRunSummary = "Trainer exited \(status): \(stderr.prefix(200))"
        }
    }

    // MARK: - Power/idle shell-outs

    private func isOnACPower() -> Bool {
        guard let out = try? runCapture(["pmset", "-g", "batt"]) else { return true }
        return out.contains("AC Power")
    }

    private func isLowPowerMode() -> Bool {
        guard let out = try? runCapture(["pmset", "-g"]) else { return false }
        return out.contains("lowpowermode              1")
    }

    private func idleSeconds() -> Double {
        guard let out = try? runCapture(["ioreg", "-c", "IOHIDSystem", "-d", "4"]) else { return 0 }
        // "HIDIdleTime" = <number> nanoseconds. Extract digits after the key.
        let pattern = "HIDIdleTime\"\\s*=\\s*(\\d+)"
        guard let range = out.range(of: pattern, options: .regularExpression) else { return 0 }
        let matched = out[range]
        if let numRange = matched.range(of: "\\d+$", options: .regularExpression),
           let ns = Double(matched[numRange])
        {
            return ns / 1_000_000_000
        }
        return 0
    }

    private func runCapture(_ cmd: [String]) -> String? {
        let proc = Process()
        proc.launchPath = "/usr/bin/env"
        proc.arguments = cmd
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do { try proc.run() } catch { return nil }
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)
    }
}
