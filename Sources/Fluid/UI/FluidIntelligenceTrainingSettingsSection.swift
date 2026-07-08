//
//  FluidIntelligenceTrainingSettingsSection.swift
//  Fluid
//
//  On-device Fluid Intelligence training settings UI (corpus status, venv
//  configuration, Train Now / Cancel). Extracted from SettingsView to keep
//  that file under SwiftLint's type_body_length limit.
//

import SwiftUI

extension SettingsView {
    func fluidIntelligenceTrainingSection() -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("On-device Training")
                .font(self.theme.typography.bodyStrong)
                .foregroundStyle(self.settingsTitleText)

            // Trainer status: corpus count, last run, last GGUF.
            let status = FluidIntelligenceTrainer.shared.status
            VStack(alignment: .leading, spacing: 3) {
                Text("Corpus: \(status.corpusEntries) samples")
                Text("Last run: \(status.lastRun.map { Self.shortDateFormatter.string(from: $0) } ?? "never")\(status.lastSummary.map { " — \($0)" } ?? "")")
                    .lineLimit(2)
                if let gguf = status.lastGGUF {
                    Text("Last model: \(gguf.lastPathComponent)")
                        .font(self.theme.typography.bodySmall)
                        .foregroundStyle(self.settingsSecondaryText)
                }
            }
            .font(self.theme.typography.bodySmall)
            .foregroundStyle(self.settingsSecondaryText)

            HStack(spacing: 8) {
                Button("Train Now") {
                    Task { @MainActor in
                        await FluidIntelligenceTrainer.shared.runIfAppropriate(minIdleSeconds: 0)
                    }
                }
                .controlSize(.small)
                .disabled(!FluidIntelligenceTrainer.shared.isConfigured || FluidIntelligenceTrainer.shared.isRunning)

                if FluidIntelligenceTrainer.shared.isRunning {
                    Button("Cancel") { FluidIntelligenceTrainer.shared.cancel() }
                        .controlSize(.small)
                }
            }
            .padding(.top, 2)

            if !FluidIntelligenceTrainer.shared.isConfigured {
                Text("On-device training is included with the Fluid Intelligence build. Nothing to configure.")
                    .font(self.theme.typography.caption)
                    .foregroundStyle(self.settingsSecondaryText)
            }
        }
        .padding(.top, 2)
    }

    static let shortDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter
    }()
}
