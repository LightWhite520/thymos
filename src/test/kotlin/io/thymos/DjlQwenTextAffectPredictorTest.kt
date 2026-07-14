package io.thymos

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DjlQwenTextAffectPredictorTest {
    @Test
    fun `trained bundle loads and returns six finite values`() {
        val bundle = Path.of("model").toAbsolutePath().normalize()
        if (!Files.isRegularFile(bundle.resolve("model.pt"))) return
        DjlQwenTextAffectPredictor.fromBundle(bundle).use { predictor ->
            val values = predictor.predict("别担心，我只是有点累")
            assertEquals(6, values.size)
            assertTrue(values.all(Float::isFinite))
        }
    }
}
