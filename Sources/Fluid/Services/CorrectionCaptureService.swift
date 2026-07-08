//
//  CorrectionCaptureService.swift
//  Fluid
//
//  Captures the user's *manually-corrected* dictation text — the richest
//  training signal. After FluidVoice types a transcript and the user edits it
//  (detected by ``PostTranscriptionEditTracker``), this service re-reads the
//  focused field's value via Accessibility and records the corrected text,
//  keyed by the dictation entry ID.
//
//  Privacy envelope: this only runs when the user has explicitly enabled
//  "Fluid Intelligence Training Data" collection
//  (``SettingsStore.allowFluidIntelligenceTrainingCollection``). That toggle's
//  disclosure states audio + transcripts are saved locally to train a personal
//  model — capturing the user's edit is squarely inside that consent. No text
//  leaves the device; ``TrainingCorpusExporter`` reads these records to build
//  the local corpus.
//
//  This closes the gap left by ``PostTranscriptionEditTracker``, which by
//  design records only low-cardinality metadata and discards text content.
//

import ApplicationServices
import Foundation

actor CorrectionCaptureService {
    static let shared = CorrectionCaptureService()

    private init() {}

    // MARK: - Pending observation

    /// A dictation awaiting a possible user edit. Held only under consent.
    private struct Pending: Sendable {
        let entryID: UUID
        let typedText: String
        let appPID: pid_t
        let completedAt: Date
        /// Seconds after `completedAt` during which an edit counts. Matches
        /// PostTranscriptionEditTracker's X-seconds window logic.
        let windowSeconds: Int
    }

    private var pending: Pending?

    /// A captured user correction, keyed by entry ID. Read by TrainingCorpusExporter.
    private struct Correction: Codable {
        let entryID: UUID
        let originalTyped: String
        let userCorrected: String
        let capturedAt: Date
    }

    private static var correctionsURL: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = base.appendingPathComponent("FluidVoice", isDirectory: true)
            .appendingPathComponent("FluidIntelligenceTraining", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("corrections.json")
    }

    // MARK: - Public API

    /// Register a freshly-typed dictation for edit observation. No-op unless
    /// training-data collection is enabled.
    func registerTyped(
        entryID: UUID,
        typedText: String,
        appPID: pid_t,
        windowSeconds: Int
    ) {
        guard SettingsStore.shared.allowFluidIntelligenceTrainingCollection else {
            self.pending = nil
            return
        }
        guard windowSeconds > 0, !typedText.isEmpty else {
            self.pending = nil
            return
        }
        self.pending = Pending(
            entryID: entryID,
            typedText: typedText,
            appPID: appPID,
            completedAt: Date(),
            windowSeconds: windowSeconds
        )
    }

    /// Called when an edit is detected (backspace / cmd-A within the window).
    /// Schedules an AX re-read of the focused field after the user has had time
    /// to finish editing, then records the corrected text.
    func handleEditDetected() async {
        guard let p = self.pending else { return }
        self.pending = nil // single-fire per dictation

        // Give the user a short moment to finish the edit before re-reading.
        // (The window already elapsed; this is a small grace period for the
        // edit keystrokes to land.)
        try? await Task.sleep(nanoseconds: 1_500_000_000) // 1.5s

        guard let corrected = await Self.readFocusedFieldValue(appPID: p.appPID) else { return }
        let trimmed = corrected.trimmingCharacters(in: .whitespacesAndNewlines)
        let original = p.typedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != original else { return }

        // Record the correction.
        await Self.appendCorrection(Correction(
            entryID: p.entryID,
            originalTyped: p.typedText,
            userCorrected: corrected,
            capturedAt: Date()
        ))
    }

    /// Read all captured corrections (used by TrainingCorpusExporter).
    nonisolated static func loadCorrections() -> [UUID: String] {
        guard let data = try? Data(contentsOf: correctionsURL) else { return [:] }
        guard let array = try? JSONDecoder().decode([Correction].self, from: data) else { return [:] }
        // Last write wins per entry ID.
        var out: [UUID: String] = [:]
        for c in array {
            out[c.entryID] = c.userCorrected
        }
        return out
    }

    /// Drop corrections for entries no longer in history (housekeeping).
    nonisolated static func prune(keeping entryIDs: Set<UUID>) {
        guard let data = try? Data(contentsOf: correctionsURL),
              let array = try? JSONDecoder().decode([Correction].self, from: data) else { return }
        let kept = array.filter { entryIDs.contains($0.entryID) }
        if kept.count != array.count {
            try? Self.write(kept)
        }
    }

    // MARK: - AX re-read

    /// Read the `.value` of the focused UI element in the target app. Returns
    /// nil if AX is unavailable or the focused element has no textual value.
    /// This uses the same Accessibility permission FluidVoice already requires
    /// for typing.
    nonisolated static func readFocusedFieldValue(appPID: pid_t) async -> String? {
        let app = AXUIElementCreateApplication(appPID)
        var focused: CFTypeRef?
        let focusResult = AXUIElementCopyAttributeValue(app, kAXFocusedUIElementAttribute as CFString, &focused)
        guard focusResult == .success else { return nil }

        // AXUIElement is a CoreFoundation opaque type; CFTypeRef bridging to it
        // is the documented pattern. Use unsafeBitCast (the values are the same
        // pointer) to avoid both a force-cast (SwiftLint `force_cast`) and the
        // "always succeeds" warning a conditional downcast produces for CF types.
        guard let raw = focused else { return nil }
        let focusedElement = unsafeBitCast(raw, to: AXUIElement.self)

        var valueRef: CFTypeRef?
        let valueResult = AXUIElementCopyAttributeValue(focusedElement, kAXValueAttribute as CFString, &valueRef)
        guard valueResult == .success, let value = valueRef else { return nil }

        // kAXValue is typically a string for text fields; guard the cast.
        return value as? String
    }

    // MARK: - Persistence

    private static func appendCorrection(_ correction: Correction) async {
        let existing = (try? Data(contentsOf: self.correctionsURL))
            .flatMap { try? JSONDecoder().decode([Correction].self, from: $0) } ?? []
        var all = existing
        // Replace any prior correction for the same entry.
        all.removeAll { $0.entryID == correction.entryID }
        all.append(correction)
        self.write(all)
    }

    private static func write(_ corrections: [Correction]) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted]
        if let data = try? encoder.encode(corrections) {
            try? data.write(to: self.correctionsURL, options: .atomic)
        }
    }
}
