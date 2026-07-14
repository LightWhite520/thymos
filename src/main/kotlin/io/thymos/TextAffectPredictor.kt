package io.thymos

interface TextAffectPredictor : AutoCloseable {
    fun predict(text: String): FloatArray
}
