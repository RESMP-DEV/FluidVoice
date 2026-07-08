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

            // venv path configuration (required for the trainer to run).
            HStack(spacing: 8) {
                Text("Python venv")
                    .font(self.theme.typography.bodySmall)
                    .frame(width: 110, alignment: .leading)
                TextField("/path/to/venv", text: Binding(
                    get: { SettingsStore.shared.fluidIntelligenceTrainerVenvPath ?? "" },
                    set: { SettingsStore.shared.fluidIntelligenceTrainerVenvPath = $0.isEmpty ? nil : $0 }
                ))
                .textFieldStyle(.roundedBorder)
                .controlSize(.small)
            }

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
                Text("Configure a Python venv with the `fluidvoice-finetune` package to enable training.")
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
