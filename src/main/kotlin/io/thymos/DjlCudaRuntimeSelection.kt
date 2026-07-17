package io.thymos

internal object DjlCudaRuntimeSelection {
    private val flavorPattern = Regex("^cu(\\d{3})$")

    fun selectFlavor(
        indexLines: List<String>,
        classifier: String,
        requestedFlavor: String,
    ): String? {
        val requestedVersion = flavorPattern.matchEntire(requestedFlavor)
            ?.groupValues
            ?.get(1)
            ?.toInt()
            ?: throw IllegalArgumentException("Unsupported PyTorch CUDA flavor: $requestedFlavor")
        val torchLibrary = Regex(
            "^cu(\\d{3})/${Regex.escape(classifier)}/native/lib/torch\\.dll\\.gz$",
        )
        return indexLines.asSequence()
            .mapNotNull { torchLibrary.matchEntire(it)?.groupValues?.get(1)?.toInt() }
            .filter { it <= requestedVersion }
            .maxOrNull()
            ?.let { "cu$it" }
    }
}
