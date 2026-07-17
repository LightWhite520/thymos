package io.thymos

interface TextAffectPredictor : AutoCloseable {
    val inferenceEngineDescription: String
        get() = this::class.qualifiedName ?: this::class.simpleName ?: "unknown"

    fun predict(text: String): FloatArray
}
