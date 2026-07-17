package io.thymos

import ai.djl.Model
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.file.Files
import java.nio.file.Path

class DjlQwenTextAffectPredictor private constructor(
    private val predictor: Predictor<String, FloatArray>,
    private val model: Model,
    private val tokenizer: HuggingFaceTokenizer,
) : TextAffectPredictor {
    override fun predict(text: String): FloatArray = predictor.predict(text)

    override fun close() {
        predictor.close()
        model.close()
        tokenizer.close()
    }

    companion object {
        fun fromBundle(
            bundlePath: Path,
            engineName: String = "PyTorch",
            devicePolicy: ThymosDevicePolicy = ThymosDevicePolicy.fromRuntime(),
        ): DjlQwenTextAffectPredictor {
            val metadataPath = bundlePath.resolve("metadata.json")
            val modelPath = bundlePath.resolve("model.pt")
            val tokenizerPath = bundlePath.resolve("tokenizer.json")
            require(Files.isRegularFile(metadataPath)) { "Thymos metadata is missing: $metadataPath" }
            require(Files.isRegularFile(modelPath)) { "Thymos model is missing: $modelPath" }
            require(Files.isRegularFile(tokenizerPath)) { "Thymos tokenizer is missing: $tokenizerPath" }

            val metadata = Json.parseToJsonElement(Files.readString(metadataPath)).jsonObject
            require(metadata["schemaVersion"]?.jsonPrimitive?.content?.toInt() == 2) {
                "Unsupported Thymos schema version"
            }
            require(metadata["baseModel"]?.jsonPrimitive?.content == "Qwen/Qwen3-Embedding-0.6B") {
                "Unexpected Thymos base model"
            }
            require(metadata["precision"]?.jsonPrimitive?.content == "bfloat16") {
                "Unexpected Thymos precision"
            }
            require(metadata["embeddingDimension"]?.jsonPrimitive?.content?.toInt() == 1024) {
                "Unexpected Thymos embedding dimension"
            }
            val outputs = metadata["outputs"]?.jsonArray?.map { it.jsonPrimitive.content }
            require(outputs == listOf("valence", "arousal", "dominance", "connectionNeed", "openness", "confidence")) {
                "Unexpected Thymos outputs"
            }
            val maxSequenceLength = metadata["maxSequenceLength"]?.jsonPrimitive?.content?.toInt()
                ?: error("Thymos maxSequenceLength is missing")
            require(maxSequenceLength > 0) { "Thymos maxSequenceLength must be positive" }

            var firstFailure: Throwable? = null
            for (device in devicePolicy.candidates(engineName)) {
                val tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath)
                val model = Model.newInstance("model", device, engineName)
                try {
                    model.load(bundlePath, "model")
                    val predictor = model.newPredictor(QwenAffectTranslator(tokenizer, maxSequenceLength))
                    return DjlQwenTextAffectPredictor(predictor, model, tokenizer)
                } catch (failure: Throwable) {
                    if (firstFailure == null) firstFailure = failure
                    tokenizer.close()
                    model.close()
                    if (devicePolicy != ThymosDevicePolicy.AUTO) throw failure
                }
            }
            throw firstFailure ?: IllegalStateException("No Thymos device candidates were available")
        }
    }
}

private class QwenAffectTranslator(
    private val tokenizer: HuggingFaceTokenizer,
    private val maxSequenceLength: Int,
) : Translator<String, FloatArray> {
    override fun processInput(context: TranslatorContext, input: String): NDList {
        val encoded = tokenizer.encode(input)
        val ids = LongArray(maxSequenceLength)
        val mask = LongArray(maxSequenceLength)
        val length = minOf(maxSequenceLength, encoded.ids.size)
        encoded.ids.copyInto(ids, endIndex = length)
        encoded.attentionMask.copyInto(mask, endIndex = length)
        return NDList(context.ndManager.create(ids), context.ndManager.create(mask))
    }

    override fun processOutput(context: TranslatorContext, list: NDList): FloatArray {
        val output = list.singletonOrThrow().toFloatArray()
        require(output.size == 6 && output.all(Float::isFinite)) {
            "Thymos output must contain six finite values"
        }
        return output
    }
}
