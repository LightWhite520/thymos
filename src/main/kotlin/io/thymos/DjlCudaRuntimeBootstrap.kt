package io.thymos

import ai.djl.util.Platform
import ai.djl.util.Utils
import java.net.URI
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.logging.Logger

internal data class DjlPytorchPlatform(
    val version: String,
    val classifier: String,
)

internal data class PreparedDjlCudaRuntime(
    val flavor: String,
    val nativeDirectory: Path,
)

internal class DjlCudaRuntimeBootstrap(
    private val osName: String = System.getProperty("os.name"),
    private val explicitFlavor: () -> String? = {
        System.getProperty(PYTORCH_FLAVOR)
            ?: System.getenv(PYTORCH_FLAVOR)
    },
    private val detectDriverFlavor: () -> String? = WindowsCudaNativeAccess::detectDriverFlavor,
    private val platform: () -> DjlPytorchPlatform = {
        Platform.detectPlatform(PYTORCH_ENGINE).let {
            DjlPytorchPlatform(it.version, it.classifier)
        }
    },
    private val cacheRoot: () -> Path = { Utils.getEngineCacheDir(PYTORCH_ENGINE) },
    private val loadIndex: (DjlPytorchPlatform) -> List<String> = ::loadPytorchIndex,
    private val createDirectories: (Path) -> Unit = Files::createDirectories,
    private val setFlavor: (String) -> Unit = { System.setProperty(PYTORCH_FLAVOR, it) },
    private val prependProcessPath: (Path) -> Unit = WindowsCudaNativeAccess::prependProcessPath,
    private val logSelection: (PreparedDjlCudaRuntime) -> Unit = {
        logger.info("Thymos selected DJL CUDA flavor=${it.flavor} nativeDir=${it.nativeDirectory}")
    },
) {
    private var prepared: PreparedDjlCudaRuntime? = null

    @Synchronized
    fun prepare(devicePolicy: ThymosDevicePolicy): PreparedDjlCudaRuntime? {
        prepared?.let { return it }
        if (devicePolicy == ThymosDevicePolicy.CPU || !osName.startsWith("Windows", ignoreCase = true)) {
            return null
        }

        val requestedFlavor = explicitFlavor()?.takeIf(String::isNotBlank)
            ?: detectDriverFlavor()
            ?: return unavailable(
                devicePolicy,
                "Thymos GPU mode requires an available NVIDIA CUDA driver",
            )
        val detectedPlatform = platform()
        val selectedFlavor = DjlCudaRuntimeSelection.selectFlavor(
            indexLines = loadIndex(detectedPlatform),
            classifier = detectedPlatform.classifier,
            requestedFlavor = requestedFlavor,
        ) ?: return unavailable(
            devicePolicy,
            "No DJL CUDA runtime is compatible with $requestedFlavor on ${detectedPlatform.classifier}",
        )
        val nativeDirectory = cacheRoot().resolve(
            "${detectedPlatform.version}-$selectedFlavor-${detectedPlatform.classifier}",
        )
        createDirectories(nativeDirectory)
        setFlavor(selectedFlavor)
        prependProcessPath(nativeDirectory)
        return PreparedDjlCudaRuntime(selectedFlavor, nativeDirectory).also {
            prepared = it
            logSelection(it)
        }
    }

    private fun unavailable(devicePolicy: ThymosDevicePolicy, message: String): PreparedDjlCudaRuntime? {
        if (devicePolicy == ThymosDevicePolicy.GPU) throw IllegalStateException(message)
        return null
    }

    companion object {
        private const val PYTORCH_ENGINE = "pytorch"
        private const val PYTORCH_FLAVOR = "PYTORCH_FLAVOR"
        private val logger = Logger.getLogger(DjlCudaRuntimeBootstrap::class.java.name)

        private fun loadPytorchIndex(platform: DjlPytorchPlatform): List<String> {
            val cacheRoot = Utils.getEngineCacheDir(PYTORCH_ENGINE)
            val indexPath = cacheRoot.resolve("${platform.version}.txt")
            if (Files.isRegularFile(indexPath)) return Files.readAllLines(indexPath)

            Files.createDirectories(cacheRoot)
            val releaseVersion = Regex("^(\\d+\\.\\d+\\.\\d+)")
                .find(platform.version)
                ?.groupValues
                ?.get(1)
                ?: error("Unsupported DJL PyTorch version: ${platform.version}")
            val temporary = Files.createTempFile(cacheRoot, platform.version, ".tmp")
            try {
                URI("https://publish.djl.ai/pytorch/$releaseVersion/files.txt").toURL().openStream().use { input ->
                    Files.copy(input, temporary, StandardCopyOption.REPLACE_EXISTING)
                }
                Files.move(temporary, indexPath, StandardCopyOption.REPLACE_EXISTING)
                return Files.readAllLines(indexPath)
            } finally {
                Files.deleteIfExists(temporary)
            }
        }
    }
}
