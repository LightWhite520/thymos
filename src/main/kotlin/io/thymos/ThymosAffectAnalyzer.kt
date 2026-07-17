package io.thymos

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.nio.file.Path

class ThymosAffectAnalyzer(
    private val predictor: TextAffectPredictor,
    private val fallback: suspend (String) -> AffectState = { AffectState.Uncertain },
) : AutoCloseable {
    private val mutex = Mutex()

    suspend fun analyze(text: String): AffectState = mutex.withLock {
        runCatching { AffectState.from(predictor.predict(text)) }
            .getOrElse { fallback(text) }
    }

    override fun close() = predictor.close()

    companion object {
        fun fromBundle(
            bundlePath: Path,
            engineName: String = "PyTorch",
            devicePolicy: ThymosDevicePolicy = ThymosDevicePolicy.fromRuntime(),
            fallback: suspend (String) -> AffectState = { AffectState.Uncertain },
        ): ThymosAffectAnalyzer = ThymosAffectAnalyzer(
            predictor = DjlQwenTextAffectPredictor.fromBundle(bundlePath, engineName, devicePolicy),
            fallback = fallback,
        )
    }
}
