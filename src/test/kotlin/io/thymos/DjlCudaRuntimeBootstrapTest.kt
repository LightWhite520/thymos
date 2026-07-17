package io.thymos

import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class DjlCudaRuntimeBootstrapTest {
    private val platform = DjlPytorchPlatform("2.7.1", "win-x86_64")
    private val index = listOf(
        "cu121/win-x86_64/native/lib/torch.dll.gz",
        "cu128/win-x86_64/native/lib/torch.dll.gz",
    )

    @Test
    fun `cpu policy performs no CUDA work`() {
        var driverDetections = 0
        val harness = harness(detectDriverFlavor = {
            driverDetections += 1
            "cu130"
        })

        assertNull(harness.bootstrap.prepare(ThymosDevicePolicy.CPU))
        assertEquals(0, driverDetections)
        assertEquals(emptyList(), harness.mutations)
    }

    @Test
    fun `non Windows host performs no CUDA work`() {
        val harness = harness(osName = "Linux")

        assertNull(harness.bootstrap.prepare(ThymosDevicePolicy.GPU))
        assertEquals(emptyList(), harness.mutations)
    }

    @Test
    fun `explicit flavor wins without probing the driver`() {
        var driverDetections = 0
        val harness = harness(
            explicitFlavor = { "cu121" },
            detectDriverFlavor = {
                driverDetections += 1
                "cu130"
            },
        )

        val prepared = harness.bootstrap.prepare(ThymosDevicePolicy.GPU)

        assertEquals("cu121", prepared?.flavor)
        assertEquals(0, driverDetections)
        assertEquals(
            Path.of("cache", "2.7.1-cu121-win-x86_64"),
            prepared?.nativeDirectory,
        )
    }

    @Test
    fun `automatic mode selects highest compatible indexed flavor`() {
        val harness = harness(detectDriverFlavor = { "cu130" })

        val prepared = harness.bootstrap.prepare(ThymosDevicePolicy.AUTO)

        assertEquals("cu128", prepared?.flavor)
        assertEquals(
            listOf(
                "mkdir:cache\\2.7.1-cu128-win-x86_64",
                "flavor:cu128",
                "path:cache\\2.7.1-cu128-win-x86_64",
            ),
            harness.mutations,
        )
    }

    @Test
    fun `strict GPU policy fails when NVIDIA driver is unavailable`() {
        val harness = harness(detectDriverFlavor = { null })

        val failure = assertFailsWith<IllegalStateException> {
            harness.bootstrap.prepare(ThymosDevicePolicy.GPU)
        }

        assertEquals("Thymos GPU mode requires an available NVIDIA CUDA driver", failure.message)
    }

    @Test
    fun `auto policy leaves CPU fallback available when driver is unavailable`() {
        val harness = harness(detectDriverFlavor = { null })

        assertNull(harness.bootstrap.prepare(ThymosDevicePolicy.AUTO))
        assertEquals(emptyList(), harness.mutations)
    }

    @Test
    fun `successful process preparation is idempotent`() {
        val harness = harness(detectDriverFlavor = { "cu130" })

        val first = harness.bootstrap.prepare(ThymosDevicePolicy.GPU)
        val second = harness.bootstrap.prepare(ThymosDevicePolicy.GPU)

        assertEquals(first, second)
        assertEquals(3, harness.mutations.size)
    }

    private fun harness(
        osName: String = "Windows 11",
        explicitFlavor: () -> String? = { null },
        detectDriverFlavor: () -> String? = { "cu128" },
    ): BootstrapHarness {
        val mutations = mutableListOf<String>()
        val bootstrap = DjlCudaRuntimeBootstrap(
            osName = osName,
            explicitFlavor = explicitFlavor,
            detectDriverFlavor = detectDriverFlavor,
            platform = { platform },
            cacheRoot = { Path.of("cache") },
            loadIndex = { index },
            createDirectories = { mutations += "mkdir:$it" },
            setFlavor = { mutations += "flavor:$it" },
            prependProcessPath = { mutations += "path:$it" },
            logSelection = {},
        )
        return BootstrapHarness(bootstrap, mutations)
    }

    private data class BootstrapHarness(
        val bootstrap: DjlCudaRuntimeBootstrap,
        val mutations: List<String>,
    )
}
