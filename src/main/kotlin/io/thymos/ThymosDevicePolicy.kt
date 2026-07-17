package io.thymos

import ai.djl.Device
import ai.djl.engine.Engine

enum class ThymosDevicePolicy {
    AUTO,
    CPU,
    GPU,
    ;

    fun resolve(engineName: String = "PyTorch"): Device =
        resolve(gpuCount = { Engine.getEngine(engineName).gpuCount })

    fun candidates(engineName: String = "PyTorch"): List<Device> =
        candidates(gpuCount = { Engine.getEngine(engineName).gpuCount })

    internal fun resolve(gpuCount: () -> Int): Device = when (this) {
        AUTO -> if (gpuCount() > 0) Device.gpu() else Device.cpu()
        CPU -> Device.cpu()
        GPU -> Device.gpu()
    }

    internal fun candidates(gpuCount: () -> Int): List<Device> = when (this) {
        AUTO -> if (gpuCount() > 0) listOf(Device.gpu(), Device.cpu()) else listOf(Device.cpu())
        CPU -> listOf(Device.cpu())
        GPU -> listOf(Device.gpu())
    }

    companion object {
        fun fromRuntime(): ThymosDevicePolicy =
            parse(
                System.getProperty("thymos.device")
                    ?: System.getenv("THYMOS_DEVICE")
                    ?: System.getenv("OPENEDEN_THYMOS_DEVICE")
                    ?: "auto",
            )

        fun parse(value: String): ThymosDevicePolicy = when (value.trim().lowercase()) {
            "", "auto" -> AUTO
            "cpu" -> CPU
            "gpu", "cuda" -> GPU
            else -> error("Unsupported Thymos device policy: $value")
        }
    }
}
