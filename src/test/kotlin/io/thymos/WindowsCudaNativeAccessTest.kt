package io.thymos

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class WindowsCudaNativeAccessTest {
    @Test
    fun `converts CUDA driver API version to DJL flavor`() {
        assertEquals("cu128", WindowsCudaNativeAccess.toDjlFlavor(12080))
        assertEquals("cu121", WindowsCudaNativeAccess.toDjlFlavor(12010))
        assertEquals("cu130", WindowsCudaNativeAccess.toDjlFlavor(13000))
    }

    @Test
    fun `returns driver flavor after successful native calls`() {
        val flavor = WindowsCudaNativeAccess.detectDriverFlavor(
            initialize = { 0 },
            readVersion = { output ->
                output[0] = 12080
                0
            },
        )

        assertEquals("cu128", flavor)
    }

    @Test
    fun `returns null when CUDA driver initialization fails`() {
        assertNull(
            WindowsCudaNativeAccess.detectDriverFlavor(
                initialize = { 100 },
                readVersion = { 0 },
            ),
        )
    }

    @Test
    fun `returns null when CUDA driver version query fails`() {
        assertNull(
            WindowsCudaNativeAccess.detectDriverFlavor(
                initialize = { 0 },
                readVersion = { 1 },
            ),
        )
    }

    @Test
    fun `prepends native directory once ignoring Windows path case`() {
        assertEquals(
            "C:\\djl\\cu128;C:\\Windows\\System32",
            WindowsCudaNativeAccess.prependPathValue(
                currentPath = "C:\\Windows\\System32",
                nativeDirectory = "C:\\djl\\cu128",
            ),
        )
        assertEquals(
            "C:\\DJL\\CU128;C:\\Windows\\System32",
            WindowsCudaNativeAccess.prependPathValue(
                currentPath = "C:\\DJL\\CU128;C:\\Windows\\System32",
                nativeDirectory = "c:\\djl\\cu128",
            ),
        )
    }
}
