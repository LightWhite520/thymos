package io.thymos

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DjlQwenTextAffectPredictorTest {
    @Test
    fun `auto device policy prefers gpu when available`() {
        val device = ThymosDevicePolicy.AUTO.resolve(gpuCount = { 1 })

        assertTrue(device.isGpu)
    }

    @Test
    fun `auto device policy keeps cpu fallback after gpu candidate`() {
        val devices = ThymosDevicePolicy.AUTO.candidates(gpuCount = { 1 })

        assertTrue(devices.first().isGpu)
        assertFalse(devices.last().isGpu)
    }

    @Test
    fun `auto device policy falls back to cpu when gpu is unavailable`() {
        val device = ThymosDevicePolicy.AUTO.resolve(gpuCount = { 0 })

        assertFalse(device.isGpu)
    }

    @Test
    fun `device policy can be forced from text`() {
        assertFalse(ThymosDevicePolicy.parse("cpu").resolve(gpuCount = { 4 }).isGpu)
        assertTrue(ThymosDevicePolicy.parse("cuda").resolve(gpuCount = { 0 }).isGpu)
    }

    @Test
    fun `CUDA runtime is prepared before DJL device candidates are resolved`() {
        val events = mutableListOf<String>()

        val devices = DjlQwenTextAffectPredictor.prepareDeviceCandidates(
            devicePolicy = ThymosDevicePolicy.GPU,
            prepareRuntime = {
                events += "prepare:${it.name}"
            },
            resolveCandidates = {
                events += "candidates"
                listOf(ai.djl.Device.gpu())
            },
        )

        assertEquals(listOf("prepare:GPU", "candidates"), events)
        assertTrue(devices.single().isGpu)
    }

    @Test
    fun `trained bundle loads and returns six finite values`() {
        val bundle = runtimeBundlePath()
        if (!Files.isRegularFile(bundle.resolve("model.pt"))) return
        DjlQwenTextAffectPredictor.fromBundle(bundle).use { predictor ->
            val values = predictor.predict("别担心，我只是有点累")
            assertEquals(6, values.size)
            assertTrue(values.all(Float::isFinite))
        }
    }

    private fun runtimeBundlePath(): Path =
        System.getenv("THYMOS_MODEL_PATH")?.takeIf(String::isNotBlank)
            ?.let(Path::of)
            ?: Path.of("../openeden/data/models/thymos-6d")
                .takeIf { Files.isRegularFile(it.resolve("model.pt")) }
            ?: Path.of("model")
}
