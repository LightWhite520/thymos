package io.thymos

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.nio.file.Path
import kotlin.io.path.createDirectories

data class CliArguments(
    val modelDirectory: Path,
    val text: String,
)

fun parseCliArguments(arguments: Array<String>): CliArguments {
    var modelDirectory = Path.of("model").createDirectories()
    var text: String? = null
    var index = 0
    while (index < arguments.size) {
        when (val argument = arguments[index]) {
            "--model-dir" -> modelDirectory = Path.of(arguments.getOrNull(++index) ?: error("Missing --model-dir value"))
            "--text" -> text = arguments.getOrNull(++index) ?: error("Missing --text value")
            else -> error("Unknown argument: $argument")
        }
        index += 1
    }
    return CliArguments(modelDirectory, requireNotNull(text) { "--text is required" })
}

fun main(arguments: Array<String>) = runBlocking {
    val options = parseCliArguments(arguments)
    ThymosAffectAnalyzer.fromBundle(options.modelDirectory).use { analyzer ->
        val state = analyzer.analyze(options.text)
        val output = buildJsonObject {
            put("valence", state.valence)
            put("arousal", state.arousal)
            put("dominance", state.dominance)
            put("connectionNeed", state.connectionNeed)
            put("openness", state.openness)
            put("confidence", state.confidence)
        }
        println(Json { prettyPrint = true }.encodeToString(output))
    }
}
