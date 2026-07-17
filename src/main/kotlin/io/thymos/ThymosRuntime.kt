package io.thymos

object ThymosRuntime {
    private val cudaBootstrap = DjlCudaRuntimeBootstrap()

    @JvmStatic
    fun prepare(devicePolicy: ThymosDevicePolicy = ThymosDevicePolicy.fromRuntime()) {
        cudaBootstrap.prepare(devicePolicy)
    }
}
