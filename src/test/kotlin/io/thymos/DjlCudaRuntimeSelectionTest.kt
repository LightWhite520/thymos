package io.thymos

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class DjlCudaRuntimeSelectionTest {
    private val index = listOf(
        "cu118/win-x86_64/native/lib/torch.dll.gz",
        "cu121/win-x86_64/native/lib/torch.dll.gz",
        "cu128/win-x86_64/native/lib/torch.dll.gz",
        "cu130/linux-x86_64/native/lib/libtorch.so.gz",
    )

    @Test
    fun `selects highest DJL flavor supported by a newer driver`() {
        assertEquals(
            "cu128",
            DjlCudaRuntimeSelection.selectFlavor(index, "win-x86_64", "cu130"),
        )
    }

    @Test
    fun `selects highest flavor no newer than driver capability`() {
        assertEquals(
            "cu121",
            DjlCudaRuntimeSelection.selectFlavor(index, "win-x86_64", "cu124"),
        )
    }

    @Test
    fun `keeps exact explicitly requested flavor when available`() {
        assertEquals(
            "cu118",
            DjlCudaRuntimeSelection.selectFlavor(index, "win-x86_64", "cu118"),
        )
    }

    @Test
    fun `returns null when index has no compatible Windows flavor`() {
        assertNull(DjlCudaRuntimeSelection.selectFlavor(index, "win-aarch64", "cu130"))
    }

    @Test
    fun `rejects malformed requested flavor`() {
        assertFailsWith<IllegalArgumentException> {
            DjlCudaRuntimeSelection.selectFlavor(index, "win-x86_64", "cuda-12.8")
        }
    }
}
