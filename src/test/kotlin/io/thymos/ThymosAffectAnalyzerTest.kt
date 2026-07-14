package io.thymos

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals

class ThymosAffectAnalyzerTest {
    @Test
    fun `predictor receives original text and maps six dimensions`() = runTest {
        var received = ""
        val analyzer = ThymosAffectAnalyzer(
            predictor = object : TextAffectPredictor {
                override fun predict(text: String): FloatArray {
                    received = text
                    return floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.9f)
                }

                override fun close() = Unit
            },
        )

        val state = analyzer.analyze("别担心，我只是有点累")

        assertEquals("别担心，我只是有点累", received)
        assertEquals(0.1f, state.valence)
        assertEquals(0.9f, state.confidence)
    }

    @Test
    fun `invalid predictor output uses injected fallback`() = runTest {
        val analyzer = ThymosAffectAnalyzer(
            predictor = object : TextAffectPredictor {
                override fun predict(text: String): FloatArray = floatArrayOf(Float.NaN)
                override fun close() = Unit
            },
            fallback = { AffectState.Uncertain.copy(confidence = 0.25f) },
        )

        assertEquals(0.25f, analyzer.analyze("ambiguous").confidence)
    }
}
