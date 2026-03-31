package com.abhi.pocketqsar

import android.content.Context
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

class Sme2Benchmark(private val context: Context) {

    companion object {
        // Safe library load — never crashes the app
        val libraryLoaded: Boolean = try {
            System.loadLibrary("sme2bench")
            true
        } catch (e: Throwable) {
            android.util.Log.e("SME2", "Failed to load sme2bench: ${e.message}")
            false
        }

        @JvmStatic external fun nativeIsSme2Available(): Boolean
        @JvmStatic external fun nativeGetSvlBits(): Int
        @JvmStatic external fun nativeGetCapsString(): String
        @JvmStatic external fun nativeRunQsarBenchmark(iters: Int): FloatArray
        @JvmStatic external fun nativeInferenceLatency(iters: Int): FloatArray

        val LAYER_NAMES = listOf(
            "FC1 (2048→512)", "FC2 (512→256)",
            "Sol head (256→3)", "Tox head (256→2)"
        )
    }

    data class DeviceCaps(
        val sme2: Boolean,
        val sve2: Boolean,
        val svlBits: Int,
        val deviceModel: String,
        val abi: String,
        val cpuCores: Int,
        val nativeAvailable: Boolean = true,
        val loadError: String = "",
    ) {
        val tier: String get() = when {
            !nativeAvailable -> "Native library unavailable"
            sme2 -> "SME2 (Armv9.3+)"
            else -> "NEON (Armv8)"
        }
        val svlDescription: String get() =
            if (svlBits > 0) "${svlBits}-bit SVL (${svlBits/32} FP32 elements)" else "N/A"
    }

    data class LayerResult(
        val name: String,
        val neonAvgMs: Float,
        val neonP50Ms: Float,
        val neonP90Ms: Float,
        val neonStddevMs: Float,
        val sme2AvgMs: Float,
        val sme2P50Ms: Float,
        val sme2P90Ms: Float,
        val sme2StddevMs: Float,
        val speedup: Float,
    ) {
        val speedupPct: Int get() = ((speedup - 1f) * 100f).roundToInt()
        val gainMs: Float get() = neonAvgMs - sme2AvgMs
    }

    data class BenchmarkResult(
        val caps: DeviceCaps,
        val layers: List<LayerResult>,
        val totalNeonMs: Float,
        val totalSme2Ms: Float,
        val totalSpeedup: Float,
        val inferenceNeonMs: Float,
        val inferenceSme2Ms: Float,
        val iters: Int,
        val error: String = "",
        val timestampMs: Long = System.currentTimeMillis(),
    ) {
        val totalSpeedupPct: Int get() = ((totalSpeedup - 1f) * 100f).roundToInt()
        val inferenceSpeedup: Float get() =
            inferenceNeonMs / inferenceSme2Ms.coerceAtLeast(0.001f)

        fun summary() = buildString {
            if (error.isNotEmpty()) {
                appendLine("ERROR: $error")
                appendLine()
            }
            appendLine("Device : ${caps.deviceModel}")
            appendLine("ABI    : ${caps.abi}  Cores: ${caps.cpuCores}")
            appendLine("Tier   : ${caps.tier}")
            if (!caps.nativeAvailable) {
                appendLine()
                appendLine("Native library could not be loaded.")
                appendLine("Error: ${caps.loadError}")
                appendLine()
                appendLine("Possible causes:")
                appendLine("  • ABI mismatch (library built for arm64, device is x86)")
                appendLine("  • NDK version incompatibility")
                appendLine("  • Missing libsme2bench.so in APK")
                return@buildString
            }
            if (caps.sme2) appendLine("SVL    : ${caps.svlDescription}")
            appendLine()
            appendLine("Per-layer results ($iters iterations, batch=32):")
            layers.forEach { l ->
                appendLine("  ${l.name.padEnd(22)} "
                    + "NEON ${"%.2f".format(l.neonAvgMs)}ms "
                    + "→ SME2 ${"%.2f".format(l.sme2AvgMs)}ms "
                    + "(${"%.2f".format(l.speedup)}×)")
            }
            appendLine()
            appendLine("Total  : ${"%.2f".format(totalNeonMs)}ms → ${"%.2f".format(totalSme2Ms)}ms  (${"%.2f".format(totalSpeedup)}×)")
            appendLine("Infer  : ${"%.3f".format(inferenceNeonMs)}ms → ${"%.3f".format(inferenceSme2Ms)}ms  (batch=1)")
        }
    }

    fun detectCapabilities(): DeviceCaps {
        if (!libraryLoaded) {
            return DeviceCaps(
                sme2 = false, sve2 = false, svlBits = 0,
                deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}",
                abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown",
                cpuCores = Runtime.getRuntime().availableProcessors(),
                nativeAvailable = false,
                loadError = "libsme2bench.so failed to load — see Logcat tag SME2",
            )
        }
        return DeviceCaps(
            sme2        = try { nativeIsSme2Available() } catch (e: Throwable) { false },
            sve2        = false,
            svlBits     = try { nativeGetSvlBits() } catch (e: Throwable) { 0 },
            deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}",
            abi         = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown",
            cpuCores    = Runtime.getRuntime().availableProcessors(),
            nativeAvailable = true,
        )
    }

    suspend fun runQsarBenchmark(iters: Int = 30): BenchmarkResult =
        withContext(Dispatchers.Default) {
            val caps = detectCapabilities()

            // If native library didn't load, return a diagnostic result
            if (!libraryLoaded) {
                return@withContext BenchmarkResult(
                    caps = caps,
                    layers = emptyList(),
                    totalNeonMs = 0f, totalSme2Ms = 0f, totalSpeedup = 0f,
                    inferenceNeonMs = 0f, inferenceSme2Ms = 0f,
                    iters = iters,
                    error = "libsme2bench.so could not be loaded.\n" +
                            "ABI: ${caps.abi}\n" +
                            "This usually means the native library was built for arm64\n" +
                            "but the device/emulator is x86_64.",
                )
            }

            try {
                val raw     = nativeRunQsarBenchmark(iters)
                val nLayers = LAYER_NAMES.size
                val fields  = 10

                val layers = (0 until nLayers).map { i ->
                    val off = i * fields
                    LayerResult(
                        name         = LAYER_NAMES[i],
                        neonAvgMs    = raw[off+1],
                        neonP50Ms    = raw[off+2],
                        neonP90Ms    = raw[off+3],
                        neonStddevMs = raw[off+4],
                        sme2AvgMs    = raw[off+5],
                        sme2P50Ms    = raw[off+6],
                        sme2P90Ms    = raw[off+7],
                        sme2StddevMs = raw[off+8],
                        speedup      = raw[off+9],
                    )
                }

                val sumOff     = nLayers * fields
                val totalNeon  = raw[sumOff+1]
                val totalSme2  = raw[sumOff+5]
                val totalSu    = raw[sumOff+9]
                val inf        = nativeInferenceLatency(50)

                BenchmarkResult(
                    caps            = caps,
                    layers          = layers,
                    totalNeonMs     = totalNeon,
                    totalSme2Ms     = totalSme2,
                    totalSpeedup    = totalSu,
                    inferenceNeonMs = inf[0],
                    inferenceSme2Ms = inf[1],
                    iters           = iters,
                )
            } catch (e: Throwable) {
                android.util.Log.e("SME2", "Benchmark native call failed", e)
                BenchmarkResult(
                    caps = caps, layers = emptyList(),
                    totalNeonMs = 0f, totalSme2Ms = 0f, totalSpeedup = 0f,
                    inferenceNeonMs = 0f, inferenceSme2Ms = 0f,
                    iters = iters,
                    error = "${e::class.simpleName}: ${e.message}",
                )
            }
        }
}
