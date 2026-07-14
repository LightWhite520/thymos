package io.thymos

data class AffectState(
    val valence: Float,
    val arousal: Float,
    val dominance: Float,
    val connectionNeed: Float,
    val openness: Float,
    val confidence: Float,
) {
    companion object {
        val Uncertain = AffectState(0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.0f)

        fun from(values: FloatArray): AffectState {
            require(values.size == 6 && values.all(Float::isFinite)) {
                "Thymos output must contain six finite values"
            }
            return AffectState(
                valence = values[0].coerceIn(0.0f, 1.0f),
                arousal = values[1].coerceIn(0.0f, 1.0f),
                dominance = values[2].coerceIn(0.0f, 1.0f),
                connectionNeed = values[3].coerceIn(0.0f, 1.0f),
                openness = values[4].coerceIn(0.0f, 1.0f),
                confidence = values[5].coerceIn(0.0f, 1.0f),
            )
        }
    }
}
