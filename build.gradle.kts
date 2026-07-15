import java.net.URI

plugins {
    kotlin("jvm") version "2.3.21"
    application
    `maven-publish`
}

group = "io.openeden"
version = "1.0.0"

kotlin {
    jvmToolchain(21)
}

java {
    withSourcesJar()
}

dependencies {
    api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    api("ai.djl:api:0.34.0")
    implementation("ai.djl.pytorch:pytorch-engine:0.34.0")
    implementation("ai.djl.huggingface:tokenizers:0.34.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
    if (System.getProperty("os.name").startsWith("Windows")) {
        runtimeOnly("ai.djl.pytorch:pytorch-native-cpu:2.7.1:win-x86_64")
    }
    testImplementation(kotlin("test"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.11.0")
}

application {
    mainClass = "io.thymos.MainKt"
}

tasks.register<JavaExec>("runBatch") {
    group = "application"
    description = "Run Thymos affect inference for a JSON array of input texts."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass = "io.thymos.BatchMain"
}

tasks.test {
    useJUnitPlatform()
}

val modelDir = providers.environmentVariable("THYMOS_MODEL_PATH").orElse("model")
val modelUrl = providers.environmentVariable("THYMOS_MODEL_URL")
    .orElse("https://huggingface.co/0x4C57/Thymos-6D/resolve/main")

tasks.register("ensureModel") {
    group = "thymos"
    description = "Download the Thymos runtime bundle from Hugging Face when missing."
    val targetDir = file(modelDir.get())
    val files = listOf("model.pt", "tokenizer.json", "metadata.json").map(targetDir::resolve)
    outputs.files(files)
    onlyIf { files.any { !it.exists() } }
    doLast {
        targetDir.mkdirs()
        files.filterNot { it.exists() }.forEach { target ->
            val temporary = target.resolveSibling("${target.name}.download")
            URI("${modelUrl.get()}/${target.name}").toURL().openStream().use { input ->
                temporary.outputStream().use { output -> input.copyTo(output) }
            }
            temporary.copyTo(target, overwrite = true)
            temporary.delete()
        }
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            artifactId = "thymos"
        }
    }
}
