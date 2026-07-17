package io.thymos

import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.WString
import java.nio.file.Path

internal object WindowsCudaNativeAccess {
    fun detectDriverFlavor(): String? = runCatching {
        val driver = Native.load("nvcuda", NvidiaDriverLibrary::class.java)
        detectDriverFlavor(
            initialize = { driver.cuInit(it) },
            readVersion = { driver.cuDriverGetVersion(it) },
        )
    }.getOrNull()

    internal fun detectDriverFlavor(
        initialize: (Int) -> Int,
        readVersion: (IntArray) -> Int,
    ): String? {
        if (initialize(0) != CUDA_SUCCESS) return null
        val version = IntArray(1)
        if (readVersion(version) != CUDA_SUCCESS || version[0] <= 0) return null
        return toDjlFlavor(version[0])
    }

    internal fun toDjlFlavor(driverApiVersion: Int): String {
        require(driverApiVersion > 0) { "CUDA driver API version must be positive" }
        val major = driverApiVersion / 1000
        val minor = (driverApiVersion % 1000) / 10
        return "cu%02d%d".format(major, minor)
    }

    fun prependProcessPath(nativeDirectory: Path) {
        val currentPath = System.getenv("PATH").orEmpty()
        val updatedPath = prependPathValue(currentPath, nativeDirectory.toString())
        if (updatedPath == currentPath) return
        val kernel32 = Native.load("kernel32", Kernel32Environment::class.java)
        check(kernel32.SetEnvironmentVariableW(WString("PATH"), WString(updatedPath))) {
            "Windows rejected the DJL CUDA native PATH update for $nativeDirectory"
        }
    }

    internal fun prependPathValue(currentPath: String, nativeDirectory: String): String {
        val comparableDirectory = nativeDirectory.trimEnd('\\', '/')
        val alreadyPresent = currentPath.split(';').any {
            it.trim().trimEnd('\\', '/').equals(comparableDirectory, ignoreCase = true)
        }
        return if (alreadyPresent) currentPath else {
            listOf(nativeDirectory, currentPath)
                .filter(String::isNotBlank)
                .joinToString(";")
        }
    }

    private interface NvidiaDriverLibrary : Library {
        fun cuInit(flags: Int): Int

        fun cuDriverGetVersion(driverVersion: IntArray): Int
    }

    private interface Kernel32Environment : Library {
        fun SetEnvironmentVariableW(name: WString, value: WString): Boolean
    }

    private const val CUDA_SUCCESS = 0
}
