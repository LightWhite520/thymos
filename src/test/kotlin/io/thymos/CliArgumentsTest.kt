package io.thymos

import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals

class CliArgumentsTest {
    @Test
    fun `parses model directory and original text`() {
        val arguments = parseCliArguments(arrayOf("--model-dir", "bundle", "--text", "我有点累"))

        assertEquals(Path.of("bundle"), arguments.modelDirectory)
        assertEquals("我有点累", arguments.text)
    }
}
