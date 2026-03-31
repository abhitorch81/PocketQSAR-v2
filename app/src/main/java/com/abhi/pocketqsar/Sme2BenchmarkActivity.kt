package com.abhi.pocketqsar

import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

/**
 * Rebuilt from scratch — maximum diagnostics, zero silent failures.
 * Every possible failure path shows text on screen.
 */
class Sme2BenchmarkActivity : AppCompatActivity() {

    private lateinit var tvStatus:  TextView
    private lateinit var tvLog:     TextView
    private lateinit var btnRun:    Button
    private lateinit var scroll:    ScrollView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Build layout first — before ANY other code that could crash
        buildLayout()

        log("onCreate started")
        log("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
        log("ABI: ${android.os.Build.SUPPORTED_ABIS.joinToString()}")
        log("Android: ${android.os.Build.VERSION.RELEASE} (API ${android.os.Build.VERSION.SDK_INT})")
        log("")

        // Step 1: Check if library is loaded
        log("Library loaded: ${Sme2Benchmark.libraryLoaded}")

        if (!Sme2Benchmark.libraryLoaded) {
            setStatus("LIBRARY NOT LOADED", Color.parseColor("#E24B4A"))
            log("libsme2bench.so failed to load.")
            log("Check Logcat > filter 'SME2' for the exact error.")
            btnRun.isEnabled = false
            return
        }

        // Step 2: Try hardware detection
        log("Calling nativeIsSme2Available()...")
        try {
            val sme2 = Sme2Benchmark.nativeIsSme2Available()
            log("SME2 available: $sme2")

            val svl = Sme2Benchmark.nativeGetSvlBits()
            log("SVL bits: $svl")

            val caps = Sme2Benchmark.nativeGetCapsString()
            log("Caps: $caps")

            if (sme2) {
                setStatus("SME2 AVAILABLE  •  SVL ${svl}-bit", Color.parseColor("#1D9E75"))
            } else {
                setStatus("NEON (no SME2)  •  Benchmark will run with NEON fallback",
                    Color.parseColor("#888780"))
            }
        } catch (e: Throwable) {
            setStatus("Native call failed", Color.parseColor("#E24B4A"))
            log("ERROR calling native: ${e::class.simpleName}: ${e.message}")
            log(e.stackTraceToString().take(400))
            btnRun.isEnabled = false
        }
    }

    private fun buildLayout() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
        }

        supportActionBar?.apply {
            title = "SME2 Benchmark"
            setDisplayHomeAsUpEnabled(true)
        }

        // Status badge
        tvStatus = TextView(this).apply {
            text = "Initialising…"
            textSize = 14f
            setPadding(24, 20, 24, 20)
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.GRAY)
        }
        root.addView(tvStatus, LinearLayout.LayoutParams(-1, -2))

        // Run button
        btnRun = Button(this).apply {
            text = "Run QSARNet Benchmark"
            setOnClickListener { runBenchmark() }
        }
        val btnLp = LinearLayout.LayoutParams(-1, -2)
        btnLp.setMargins(16, 16, 16, 8)
        root.addView(btnRun, btnLp)

        // Scrollable log area
        scroll = ScrollView(this)
        tvLog = TextView(this).apply {
            textSize = 12f
            setPadding(16, 16, 16, 16)
            setTextColor(Color.parseColor("#222222"))
            typeface = android.graphics.Typeface.MONOSPACE
            text = ""
        }
        scroll.addView(tvLog)
        root.addView(scroll, LinearLayout.LayoutParams(-1, -1))

        setContentView(root)
    }

    override fun onSupportNavigateUp(): Boolean { finish(); return true }

    private fun setStatus(msg: String, color: Int) {
        tvStatus.text = msg
        tvStatus.setBackgroundColor(color)
    }

    private fun log(msg: String) {
        android.util.Log.d("SME2_DIAG", msg)
        runOnUiThread {
            tvLog.append("$msg\n")
            // Auto-scroll to bottom
            scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private fun runBenchmark() {
        if (!Sme2Benchmark.libraryLoaded) {
            log("Cannot run — library not loaded")
            return
        }

        btnRun.isEnabled = false
        btnRun.text = "Running…"
        log("")
        log("─────────────────────────────")
        log("Starting benchmark (30 iters)...")

        Thread {
            try {
                log("Calling nativeRunQsarBenchmark(30)...")
                val raw = Sme2Benchmark.nativeRunQsarBenchmark(30)
                log("Raw result size: ${raw.size} floats")

                val names = Sme2Benchmark.LAYER_NAMES
                log("")
                log("Per-layer results:")
                names.forEachIndexed { i, name ->
                    val off = i * 10
                    val neon  = raw[off + 1]
                    val sme2  = raw[off + 5]
                    val speed = raw[off + 9]
                    log("  $name")
                    log("    NEON: ${"%.2f".format(neon)} ms  SME2: ${"%.2f".format(sme2)} ms  (${"%.2f".format(speed)}×)")
                }

                log("")
                log("Calling nativeInferenceLatency(50)...")
                val inf = Sme2Benchmark.nativeInferenceLatency(50)
                log("Single inference:")
                log("  NEON: ${"%.3f".format(inf[0])} ms")
                log("  SME2: ${"%.3f".format(inf[1])} ms")
                log("")
                log("BENCHMARK COMPLETE")

                runOnUiThread {
                    setStatus("DONE  •  NEON ${"%.2f".format(inf[0])}ms  →  SME2 ${"%.2f".format(inf[1])}ms",
                        Color.parseColor("#1D9E75"))
                }

            } catch (e: Throwable) {
                log("BENCHMARK ERROR: ${e::class.simpleName}")
                log("Message: ${e.message}")
                log(e.stackTraceToString().take(600))
                runOnUiThread {
                    setStatus("Error — see log below", Color.parseColor("#E24B4A"))
                }
            } finally {
                runOnUiThread {
                    btnRun.isEnabled = true
                    btnRun.text = "Run QSARNet Benchmark"
                }
            }
        }.start()
    }
}
