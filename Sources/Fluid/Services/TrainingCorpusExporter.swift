//
//  TrainingCorpusExporter.swift
//  Fluid
//
//  Exports FluidVoice dictation history into a local training corpus in the
//  shape the `fluidvoice-finetune` preprocessor expects (the same schema as
//  Aqua Voice's harvested corpus + FluidVoice's own AudioManifestRow):
//
//      ~/Library/Application Support/FluidVoice/FluidIntelligenceTraining/corpus/
//        manifest.jsonl   {audio, raw, corrected, duration, timestamp, has_correction, session_id}
//        audio/*.wav
//
//  Only entries with a genuine correction signal are emitted (wasAIProcessed &&
//  rawText != processedText), since the model learns raw→corrected enhancement.
//  Audio is copied (not moved) so the existing history store is untouched.
//

import Foundation

enum TrainingCorpusExporter {
    /// The corpus root: ``…/FluidVoice/FluidIntelligenceTraining/corpus/``.
    static var corpusRoot: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("FluidVoice", isDirectory: true)
            .appendingPathComponent("FluidIntelligenceTraining", isDirectory: true)
            .appendingPathComponent("corpus", isDirectory: true)
    }

    /// The sibling checkpoint dir the idle trainer writes adapters + GGUF into.
    static var checkpointRoot: URL {
        corpusRoot.deletingLastPathComponent().appendingPathComponent("checkpoints", isDirectory: true)
    }

    private static let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    struct ExportSummary {
        let totalCandidates: Int
        let exported: Int
        let skippedNoAudio: Int
        let skippedNoCorrection: Int
        let corpusURL: URL
    }

    /// Rebuild the corpus directory from the current transcription history.
    ///
    /// - Parameter includeAll: when true, emit every entry with audio (useful
    ///   for Parakeet/ASR training). Default false = only genuine-correction
    ///   entries (for Gemma enhancement training).
    /// - Returns: a summary describing what was exported.
    static func export(includeAll: Bool = false) throws -> ExportSummary {
        let store = DictationAudioHistoryStore.shared
        let entries = TranscriptionHistoryStore.shared.entries
        let corpusURL = self.corpusRoot
        let audioDir = corpusURL.appendingPathComponent("audio", isDirectory: true)
        try FileManager.default.createDirectory(at: audioDir, withIntermediateDirectories: true)

        // Clear previous corpus manifest + audio (cheap incremental re-export).
        try? FileManager.default.removeItem(at: corpusURL.appendingPathComponent("manifest.jsonl"))
        try? FileManager.default.removeItem(at: audioDir)
        try FileManager.default.createDirectory(at: audioDir, withIntermediateDirectories: true)

        // Load any user-captured corrections (the richest signal). Maps entry ID
        // → user-corrected text. Empty unless collection consent is on AND the
        // user edited a dictation.
        let corrections = CorrectionCaptureService.loadCorrections()
        CorrectionCaptureService.prune(keeping: Set(entries.map { $0.id }))

        var manifestLines: [String] = []
        var exported = 0
        var skippedNoAudio = 0
        var skippedNoCorrection = 0

        for entry in entries.sorted(by: { $0.timestamp < $1.timestamp }) {
            // The training target is, in priority order: a user-captured manual
            // correction (strongest), then the AI-enhanced processedText, then
            // nothing (skip — no correction signal).
            let processed = entry.processedText.trimmingCharacters(in: .whitespacesAndNewlines)
            let userCorrected = corrections[entry.id]?.trimmingCharacters(in: .whitespacesAndNewlines)
            let target = userCorrected ?? processed

            // Must have a genuine correction signal (unless includeAll).
            let isGenuine = entry.rawText.trimmingCharacters(in: .whitespacesAndNewlines)
                != target
                && !target.isEmpty
            if !includeAll && !isGenuine {
                skippedNoCorrection += 1
                continue
            }

            guard store.audioFileExists(for: entry), let audioURL = store.audioFileURL(for: entry),
                  let audio = entry.audio
            else {
                skippedNoAudio += 1
                continue
            }

            let exportName = "\(Int(entry.timestamp.timeIntervalSince1970))_\(entry.id.uuidString.prefix(8)).wav"
            let relPath = "audio/\(exportName)"
            let dest = audioDir.appendingPathComponent(exportName)
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: audioURL, to: dest)

            // Shape matches the Aqua manifest + fluidvoice_finetune.data.aqua reader.
            // `corrected` is the best target we have; `user_corrected` is emitted
            // alongside when a manual edit was captured, for provenance/filtering.
            var row: [String: Any] = [
                "audio": relPath,
                "raw": entry.rawText,
                "corrected": target,
                "duration": Double(audio.durationMilliseconds) / 1000.0,
                "timestamp": self.isoFormatter.string(from: entry.timestamp),
                "session_id": entry.id.uuidString,
                "has_correction": isGenuine,
                "model": audio.model ?? "",
            ]
            if let userCorrected {
                row["user_corrected"] = userCorrected
            }
            if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]),
               let line = String(data: data, encoding: .utf8)
            {
                manifestLines.append(line)
                exported += 1
            }
        }

        let manifestURL = corpusURL.appendingPathComponent("manifest.jsonl")
        try (manifestLines.joined(separator: "\n") + "\n")
            .write(to: manifestURL, atomically: true, encoding: .utf8)

        return ExportSummary(
            totalCandidates: entries.count,
            exported: exported,
            skippedNoAudio: skippedNoAudio,
            skippedNoCorrection: skippedNoCorrection,
            corpusURL: corpusURL
        )
    }
}
