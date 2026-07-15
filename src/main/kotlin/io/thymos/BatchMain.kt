package io.thymos

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.createDirectories

data class BatchCliArguments(
    val modelDirectory: Path,
    val input: Path,
    val output: Path?,
)

fun parseBatchCliArguments(arguments: Array<String>): BatchCliArguments {
    var modelDirectory: Path? = null
    var input: Path? = null
    var output: Path? = null
    var index = 0
    while (index < arguments.size) {
        when (val argument = arguments[index]) {
            "--model-dir" -> modelDirectory = Path.of(arguments.getOrNull(++index) ?: error("Missing --model-dir value"))
            "--input" -> input = Path.of(arguments.getOrNull(++index) ?: error("Missing --input value"))
            "--output" -> output = Path.of(arguments.getOrNull(++index) ?: error("Missing --output value"))
            else -> error("Unknown argument: $argument")
        }
        index += 1
    }
    return BatchCliArguments(
        modelDirectory = requireNotNull(modelDirectory) { "--model-dir is required" },
        input = requireNotNull(input) { "--input is required" },
        output = output,
    )
}

object BatchMain {
    @JvmStatic
    fun main(arguments: Array<String>) = runBlocking {
        val options = parseBatchCliArguments(arguments)
        val json = Json { prettyPrint = true }
        val texts = Json.parseToJsonElement(Files.readString(options.input)).let { element ->
            element as? JsonArray ?: error("input must be a JSON string array")
        }.map { it.jsonPrimitive.content }
        ThymosAffectAnalyzer.fromBundle(options.modelDirectory).use { analyzer ->
            val output = buildJsonArray {
                for (text in texts) {
                    val state = analyzer.analyze(text)
                    add(buildJsonObject {
                        put("text", text)
                        put("valence", state.valence)
                        put("arousal", state.arousal)
                        put("dominance", state.dominance)
                        put("connectionNeed", state.connectionNeed)
                        put("openness", state.openness)
                        put("confidence", state.confidence)
                    })
                }
            }
            val encoded = json.encodeToString(output)
            val outputPath = options.output
            if (outputPath == null) {
                println(encoded)
            } else {
                outputPath.parent?.createDirectories()
                Files.writeString(outputPath, encoded)
            }
        }
    }
}
