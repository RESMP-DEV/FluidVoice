import Foundation

struct PrivateAIModelArtifact: Sendable, Codable, Hashable {
    var identifier: String
    var filename: String
    var downloadURL: URL?
    var sha256: String?
    var byteCount: Int64?
    var version: String?
}

/// Fluid Intelligence on-device enhancement model variant.
///
/// `fluid-1` is a modified **Gemma 4** derivative (confirmed from the GGUF
/// metadata: `general.architecture = "gemma4"`, 35 layers, 131072 context)
/// shipped as a Q4_K_M GGUF at `altic-dev/FluidIntelligence`. The `e2b` variant
/// (from `google/gemma-4-E2B`) is the current default; `e4b` (from
/// `google/gemma-4-E4B`) is a larger, higher quality alternative produced by
/// the `FluidVoiceTrain` pipeline.
///
/// Note: the "E2B/E4B" naming also appeared in Gemma 3n, but the architectures
/// differ — Gemma 4 dropped MatFormer/ALTUP/LAUREL. Do not treat 3n and 4 as
/// interchangeable for fine-tuning.
///
/// The variant only selects *which* GGUF artifact the runtime loads — both are
/// loaded by the same private `PrivateAIProviderBridge`. The artifact literals
/// (download URL / SHA-256 / byte count) for each variant live in that private
/// package; the public OSS build resolves neither. See the seam contract on
/// `PrivateAIProviderFeatureProviding`.
enum FluidIntelligenceModelVariant: String, CaseIterable, Identifiable, Sendable, Codable, Hashable {
    /// Smaller / faster Gemma-4-E2B derivative. Current default; preserves
    /// pre-toggle behavior (`fluid-1-q4_k_m.gguf`).
    case e2b = "fluid-1-e2b"
    /// Larger / higher-quality Gemma-4-E4B derivative. New in this change set;
    /// produced by the `FluidVoiceTrain` pipeline as
    /// `models/fluid-1-e4b-q4_k_m.gguf` and uploaded to `altic-dev/FluidIntelligence`.
    case e4b = "fluid-1-e4b"

    var id: String { self.rawValue }

    /// Human-readable label for the settings picker.
    var displayName: String {
        switch self {
        case .e2b: return "E2B (faster, smaller)"
        case .e4b: return "E4B (higher quality)"
        }
    }

    /// One-line description shown under the picker option.
    var detail: String {
        switch self {
        case .e2b: return "Gemma-4-E2B derivative · ~3.4 GB · Q4_K_M GGUF"
        case .e4b: return "Gemma-4-E4B derivative · larger · Q4_K_M GGUF"
        }
    }

    /// GGUF filename the runtime loads. Must match the artifact produced by
    /// `fluidvoice-finetune` (see `gemma/export_gguf.py`) and the file listed on
    /// the `altic-dev/FluidIntelligence` model card.
    var artifactFilename: String { "\(self.rawValue)-q4_k_m.gguf" }
}

struct PrivateAIModelDownloadProgress: Sendable, Equatable {
    var bytesWritten: Int64
    var totalBytesWritten: Int64
    var totalBytesExpected: Int64?

    init(bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpected: Int64?) {
        self.bytesWritten = bytesWritten
        self.totalBytesWritten = totalBytesWritten
        self.totalBytesExpected = totalBytesExpected
    }

    init(initialExpectedBytes: Int64?) {
        self.init(
            bytesWritten: 0,
            totalBytesWritten: 0,
            totalBytesExpected: initialExpectedBytes.flatMap { $0 > 0 ? $0 : nil }
        )
    }

    var hasWrittenBytes: Bool {
        self.totalBytesWritten > 0
    }

    var fractionCompleted: Double? {
        guard self.hasWrittenBytes else { return nil }
        guard let totalBytesExpected, totalBytesExpected > 0 else { return nil }
        return min(1, max(0, Double(self.totalBytesWritten) / Double(totalBytesExpected)))
    }

    /// All expected bytes have been written. The download itself is done; the
    /// provider is now validating (SHA-256) and moving the file into place.
    var isComplete: Bool {
        guard let totalBytesExpected, totalBytesExpected > 0 else { return false }
        return self.totalBytesWritten >= totalBytesExpected
    }

    func withFallbackExpectedBytes(_ byteCount: Int64?) -> PrivateAIModelDownloadProgress {
        guard self.totalBytesExpected == nil,
              let byteCount,
              byteCount > 0
        else {
            return self
        }

        return PrivateAIModelDownloadProgress(
            bytesWritten: self.bytesWritten,
            totalBytesWritten: self.totalBytesWritten,
            totalBytesExpected: byteCount
        )
    }
}

enum PrivateAIModelDownloadProgressText {
    static func buttonTitle(for progress: PrivateAIModelDownloadProgress?) -> String {
        if progress?.isComplete == true { return "Verifying" }
        guard let fraction = progress?.fractionCompleted else { return "Downloading" }
        return "Downloading \(Int(fraction * 100))%"
    }

    static func statusText(for progress: PrivateAIModelDownloadProgress?) -> String {
        guard let progress else {
            return "Starting download. This can take a few minutes."
        }
        if progress.isComplete {
            return "Verifying download. This can take a moment."
        }
        guard progress.hasWrittenBytes else {
            return "Downloading. This can take a few minutes."
        }
        guard let fraction = progress.fractionCompleted else {
            return "Downloading. This can take a few minutes."
        }
        return "Downloading \(Int(fraction * 100))%. This can take a few minutes."
    }

    static func byteText(for progress: PrivateAIModelDownloadProgress?) -> String? {
        guard let progress else { return nil }

        if progress.isComplete {
            return "\(Self.byteCountText(progress.totalBytesWritten)) downloaded"
        }

        guard progress.hasWrittenBytes else {
            guard let expected = progress.totalBytesExpected, expected > 0 else { return nil }
            return "\(Self.byteCountText(expected)) download"
        }

        let written = Self.byteCountText(progress.totalBytesWritten)
        guard let expected = progress.totalBytesExpected, expected > 0 else {
            return "\(written) downloaded"
        }

        return "\(written) of \(Self.byteCountText(expected))"
    }

    static func detailText(for progress: PrivateAIModelDownloadProgress?) -> String {
        guard let progress else { return "Starting download..." }
        if progress.isComplete {
            return "Verifying download..."
        }
        guard let byteText = Self.byteText(for: progress) else { return "Downloading..." }
        guard progress.hasWrittenBytes else { return byteText }
        guard let fraction = progress.fractionCompleted else { return byteText }
        return "\(byteText) (\(Int(fraction * 100))%)"
    }

    private static func byteCountText(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

typealias PrivateAIModelDownloadProgressHandler = @Sendable (PrivateAIModelDownloadProgress) async -> Void

struct PrivateAIRegisteredModel: Sendable, Codable, Hashable, Identifiable {
    var id: String { self.artifact.identifier }
    var displayName: String
    var detail: String
    var isEnabled: Bool
    var parameterCount: String
    var recommendedMemoryGB: Int?
    var artifact: PrivateAIModelArtifact

    var canDownload: Bool {
        self.artifact.downloadURL != nil && self.artifact.sha256?.isEmpty == false
    }
}

enum PrivateAIRuntimeState: String, Sendable, Codable, Hashable {
    case unavailable
    case missingModel
    case configured
    case loading
    case ready
    case failed
}

struct PrivateAIStatus: Sendable, Codable, Equatable {
    var state: PrivateAIRuntimeState
    var message: String?
}

struct PrivateAIUnavailableError: LocalizedError {
    var errorDescription: String? {
        "Private AI provider is not available in this build."
    }
}

/// Contract for the Fluid Intelligence ("Private AI") provider.
///
/// - Important: **Bridge integration seam.** In the public OSS build the
///   `UnavailablePrivateAIProviderFeature` stub is installed: `isAvailable == false`
///   and the registry is empty, so the UI gates Fluid Intelligence off entirely
///   (see `PrivateFeatures.privateAIProvider`). The real implementation is
///   installed by `PrivateAIProviderBridge.install()` under
///   `#if PRIVATE_AI_PROVIDER` (a flag absent from the committed `project.pbxproj`).
///
///   When linking the bridge, it MUST honor the selected Fluid Intelligence
///   variant: each registered model's `artifact.filename` must equal
///   `SettingsStore.shared.selectedFluidIntelligenceVariant.artifactFilename`
///   for its variant (e.g. `fluid-1-e2b-q4_k_m.gguf` / `fluid-1-e4b-q4_k_m.gguf`),
///   and the artifact's `downloadURL`/`sha256`/`byteCount` must resolve to the
///   matching file under `altic-dev/FluidIntelligence`. The variant flows into
///   the runtime via `PrivateAIIntegrationService.RuntimeConfiguration.modelVariant`.
protocol PrivateAIProviderFeatureProviding: Sendable {
    var isAvailable: Bool { get }
    var providerID: String { get }
    var providerName: String { get }
    var promptSelectionID: String { get }
    var defaultModelID: String { get }
    var selectedModelDefaultsKey: String { get }
    var localModelPathDefaultsKey: String { get }
    var prefixCacheDefaultsKey: String { get }
    var boostDefaultsKey: String { get }
    var modelDirectoryName: String { get }

    func modelIDs() -> [String]
    func model(id: String) -> PrivateAIRegisteredModel?
    func canonicalModelID(for value: String) -> String?
    func isKnownModelID(_ value: String) -> Bool
    func matches(model: String) -> Bool
    func localModelURL(for model: PrivateAIRegisteredModel, directoryURL: URL) -> URL
}

extension PrivateAIProviderFeatureProviding {
    func matches(model: String) -> Bool {
        guard self.isAvailable else { return false }

        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if self.isKnownModelID(normalized) {
            return true
        }

        let normalizedProviderID = self.providerID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !normalizedProviderID.isEmpty && normalized == normalizedProviderID
    }

    func localModelURL(for model: PrivateAIRegisteredModel, directoryURL: URL) -> URL {
        directoryURL.appendingPathComponent(model.artifact.filename)
    }
}

protocol PrivateAIIntegrationProviding: Sendable {
    var configuredModelID: String { get }
    var selectedModel: PrivateAIRegisteredModel { get }
    var configuredLocalModelPath: String? { get }
    var modelDirectoryURL: URL { get }
    var isLocalRuntimeConfigured: Bool { get }

    func expectedLocalModelURL(for model: PrivateAIRegisteredModel) -> URL
    func localModelPath(for model: PrivateAIRegisteredModel) -> String?
    func isModelInstalled(_ model: PrivateAIRegisteredModel) -> Bool
    func prepareModel(
        _ model: PrivateAIRegisteredModel,
        progressHandler: PrivateAIModelDownloadProgressHandler?
    ) async throws -> URL
    func shouldHandleDictation(model: String) -> Bool
    func status(for runtime: PrivateAIIntegrationService.RuntimeConfiguration) async -> PrivateAIStatus
    func loadedModelState() async -> PrivateAIIntegrationService.LoadedModelState?
    func loadModel(_ model: PrivateAIRegisteredModel) async throws -> PrivateAIStatus
    func prewarmDictation() async
    func unloadCachedRuntime(reason: String) async
    func shutdownForTermination() async
    func enhanceDictation(
        _ inputText: String,
        runtime: PrivateAIIntegrationService.RuntimeConfiguration,
        context: PrivateAIIntegrationService.AppContext
    ) async throws -> PrivateAIIntegrationService.EnhancementResult
}

extension PrivateAIIntegrationProviding {
    func prepareModel(_ model: PrivateAIRegisteredModel) async throws -> URL {
        try await self.prepareModel(model, progressHandler: nil)
    }

    /// Best-effort, no-op by default. Backends that support prefix priming
    /// (the local FluidIntelligence bridge) override this to load and prime the
    /// dictation runtime ahead of the first request.
    func prewarmDictation() async {}

    func shutdownForTermination() async {
        await self.unloadCachedRuntime(reason: "termination")
    }
}

enum PrivateAIProviderRegistry {
    nonisolated(unsafe) static var feature: any PrivateAIProviderFeatureProviding = UnavailablePrivateAIProviderFeature()
    nonisolated(unsafe) static var integration: any PrivateAIIntegrationProviding = UnavailablePrivateAIIntegrationProvider()
}

private enum PrivateAIProviderBootstrap {
    static let installOnce: Void = {
        #if PRIVATE_AI_PROVIDER
        PrivateAIProviderBridge.install()
        #endif
    }()

    static func installIfAvailable() {
        _ = self.installOnce
    }
}

enum PrivateAIProviderFeature {
    nonisolated static var shared: any PrivateAIProviderFeatureProviding {
        PrivateAIProviderBootstrap.installIfAvailable()
        return PrivateAIProviderRegistry.feature
    }

    nonisolated static var displayName: String {
        let name = self.shared.providerName.trimmingCharacters(in: .whitespacesAndNewlines)
        return name.isEmpty ? "Private AI Provider" : name
    }

    nonisolated static func verificationFingerprint(for modelID: String) -> String {
        "private-ai-provider|\(modelID)"
    }
}

enum PrivateFeatures {
    static var privateAIProvider: Bool {
        PrivateAIProviderFeature.shared.isAvailable
    }
}

enum PrivateAIModelRegistry {
    nonisolated static var defaultModelID: String {
        PrivateAIProviderFeature.shared.defaultModelID
    }

    nonisolated static var defaultModel: PrivateAIRegisteredModel {
        if let model = model(id: defaultModelID) {
            return model
        }
        return PrivateAIRegisteredModel.unavailable
    }

    nonisolated static func model(id: String) -> PrivateAIRegisteredModel? {
        PrivateAIProviderFeature.shared.model(id: id)
    }

    nonisolated static func canonicalModelID(for value: String) -> String? {
        PrivateAIProviderFeature.shared.canonicalModelID(for: value)
    }

    nonisolated static func modelIDs(includeDisabled _: Bool = false) -> [String] {
        PrivateAIProviderFeature.shared.modelIDs()
    }

    nonisolated static func localModelURL(for model: PrivateAIRegisteredModel, directoryURL: URL) -> URL {
        PrivateAIProviderFeature.shared.localModelURL(for: model, directoryURL: directoryURL)
    }
}

extension PrivateAIRegisteredModel {
    static let unavailable = PrivateAIRegisteredModel(
        displayName: "",
        detail: "",
        isEnabled: false,
        parameterCount: "",
        recommendedMemoryGB: nil,
        artifact: PrivateAIModelArtifact(
            identifier: "",
            filename: "",
            downloadURL: nil,
            sha256: nil,
            byteCount: nil,
            version: nil
        )
    )
}

private struct UnavailablePrivateAIProviderFeature: PrivateAIProviderFeatureProviding {
    let isAvailable = false
    let providerID = "__private_ai_provider__"
    let providerName = ""
    let promptSelectionID = "__PRIVATE_AI_PROVIDER__"
    let defaultModelID = ""
    let selectedModelDefaultsKey = "PrivateAIProviderSelectedModelID"
    let localModelPathDefaultsKey = "PrivateAIProviderLocalModelPath"
    let prefixCacheDefaultsKey = "PrivateAIProviderPrefixKVCacheEnabled"
    let boostDefaultsKey = "PrivateAIProviderBoostEnabled"
    let modelDirectoryName = "PrivateAIProvider"

    func modelIDs() -> [String] { [] }
    func model(id _: String) -> PrivateAIRegisteredModel? { nil }
    func canonicalModelID(for _: String) -> String? { nil }
    func isKnownModelID(_: String) -> Bool { false }
}

private struct UnavailablePrivateAIIntegrationProvider: PrivateAIIntegrationProviding {
    var configuredModelID: String { PrivateAIModelRegistry.defaultModelID }
    var selectedModel: PrivateAIRegisteredModel { PrivateAIModelRegistry.defaultModel }
    var configuredLocalModelPath: String? { nil }
    var modelDirectoryURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("FluidVoice", isDirectory: true)
            .appendingPathComponent(PrivateAIProviderFeature.shared.modelDirectoryName, isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("FluidVoice", isDirectory: true)
            .appendingPathComponent(PrivateAIProviderFeature.shared.modelDirectoryName, isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    var isLocalRuntimeConfigured: Bool { false }

    func expectedLocalModelURL(for model: PrivateAIRegisteredModel) -> URL {
        PrivateAIModelRegistry.localModelURL(for: model, directoryURL: self.modelDirectoryURL)
    }

    func localModelPath(for _: PrivateAIRegisteredModel) -> String? { nil }
    func isModelInstalled(_: PrivateAIRegisteredModel) -> Bool { false }

    func prepareModel(
        _: PrivateAIRegisteredModel,
        progressHandler _: PrivateAIModelDownloadProgressHandler?
    ) async throws -> URL {
        throw PrivateAIUnavailableError()
    }

    func shouldHandleDictation(model _: String) -> Bool { false }

    func status(for _: PrivateAIIntegrationService.RuntimeConfiguration) async -> PrivateAIStatus {
        PrivateAIStatus(
            state: .unavailable,
            message: PrivateAIUnavailableError().errorDescription
        )
    }

    func loadedModelState() async -> PrivateAIIntegrationService.LoadedModelState? { nil }

    func loadModel(_: PrivateAIRegisteredModel) async throws -> PrivateAIStatus {
        throw PrivateAIUnavailableError()
    }

    func unloadCachedRuntime(reason _: String) async {}

    func enhanceDictation(
        _ inputText: String,
        runtime _: PrivateAIIntegrationService.RuntimeConfiguration,
        context _: PrivateAIIntegrationService.AppContext
    ) async throws -> PrivateAIIntegrationService.EnhancementResult {
        PrivateAIIntegrationService.EnhancementResult(
            outputText: inputText,
            backendKind: nil,
            latencyMilliseconds: nil
        )
    }
}
