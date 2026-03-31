package com.abhi.pocketqsar

import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    // --- Model / data ---
    private lateinit var qsar: QsarExecuTorch
    private lateinit var molecules: List<DemoMolecule>
    private var currentMol: DemoMolecule? = null

    // --- Views ---
    private lateinit var chemicalView: ChemicalSpaceView
    private lateinit var molImageView: ImageView

    private lateinit var resultView: TextView
    private lateinit var scoreText: TextView
    private lateinit var scoreBar: ProgressBar

    private lateinit var barMw: ProgressBar
    private lateinit var barLogp: ProgressBar
    private lateinit var barTpsa: ProgressBar

    private lateinit var textMw: TextView
    private lateinit var textLogp: TextView
    private lateinit var textTpsa: TextView

    private lateinit var historyButton: Button
    private lateinit var listButton: Button
    private lateinit var runButton: Button

    // Benchmark views
    private lateinit var benchmarkButton: Button
    private lateinit var benchmarkText: TextView
    private lateinit var benchmarkModeGroup: RadioGroup   // NEW

    // --- History model ---
    data class PredictionRecord(
        val molName: String,
        val logS: Float,
        val toxProb: Float,
        val devScore: Int
    )

    private val predictionHistory = mutableListOf<PredictionRecord>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // ----- Find views -----
        chemicalView = findViewById(R.id.chemicalSpaceView)
        molImageView = findViewById(R.id.molImage)

        runButton = findViewById(R.id.runDemoButton)
        historyButton = findViewById(R.id.historyButton)
        listButton = findViewById(R.id.listButton)

        resultView = findViewById(R.id.resultText)
        scoreText = findViewById(R.id.scoreText)
        scoreBar = findViewById(R.id.scoreBar)

        barMw = findViewById(R.id.barMw)
        barLogp = findViewById(R.id.barLogp)
        barTpsa = findViewById(R.id.barTpsa)

        textMw = findViewById(R.id.textMw)
        textLogp = findViewById(R.id.textLogp)
        textTpsa = findViewById(R.id.textTpsa)

        // Benchmark views
        benchmarkButton = findViewById(R.id.benchmarkButton)
        benchmarkText = findViewById(R.id.benchmarkText)
        benchmarkModeGroup = findViewById(R.id.benchmarkModeGroup)  // NEW

        // ----- Model + JSON data -----
        qsar = QsarExecuTorch(this)
        molecules = DemoMoleculeRepository.load(this)
        currentMol = molecules.firstOrNull()

        chemicalView.molecules = molecules
        chemicalView.selectedId = currentMol?.id

        // When user taps on a point in the chemical space view
        chemicalView.onMoleculeSelected = { mol ->
            currentMol = mol
            runPredictionFor(mol)
        }

        // Run button: predict for current molecule
        runButton.setOnClickListener {
            val mol = currentMol
            if (mol == null) {
                Toast.makeText(this, "No molecules loaded from JSON.", Toast.LENGTH_SHORT).show()
            } else {
                runPredictionFor(mol)
            }
        }

        // History button: show prediction history
        historyButton.setOnClickListener {
            showHistoryDialog()
        }

        // Browse molecules button: let user pick any molecule from the list
        listButton.setOnClickListener {
            showMoleculePicker()
        }

        // Benchmark button: use mode from radio group
        benchmarkButton.setOnClickListener {
            val runs = when (benchmarkModeGroup.checkedRadioButtonId) {
                R.id.benchmarkQuick -> 50     // Quick preview
                R.id.benchmarkDeep -> 1000    // Deep / more stable
                else -> 200                   // Standard
            }
            runBenchmark(runs)
            findViewById<Button>(R.id.btn3dSpace)
                .setOnClickListener { open3dSpace() }

            findViewById<Button>(R.id.btnSme2Benchmark)
                .setOnClickListener { openSme2Benchmark() }
        }
        // ── NEW: 3D Chemical Space button ──────────────────────
        findViewById<Button>(R.id.btn3dSpace).setOnClickListener {
            startActivityForResult(
                android.content.Intent(this, MolecularSpaceActivity::class.java),
                101
            )
        }

        // ── NEW: SME2 / KleidiAI Benchmark button ──────────────
        findViewById<Button>(R.id.btnSme2Benchmark).setOnClickListener {
            startActivity(
                android.content.Intent(this, Sme2BenchmarkActivity::class.java)
            )
        }

        // Optionally run prediction once on first molecule
        currentMol?.let { runPredictionFor(it) }
    }

    // ---------------------------------------------------
    // Core prediction + UI update
    // ---------------------------------------------------
    private fun runPredictionFor(mol: DemoMolecule) {
        // highlight in scatter plot
        chemicalView.selectedId = mol.id
        chemicalView.invalidate()

        // load 2D image
        loadMoleculeImage(mol)

        val (predLogS, toxProb) = qsar.predict(mol.features)
        val toxLabel = if (toxProb < 0.5f) "non-toxic" else "toxic"

        // Prediction text
        resultView.text = """
            Molecule: ${mol.name}
            Predicted logS: %.3f
            Toxicity probability: %.3f (%s)
        """.trimIndent().format(predLogS, toxProb, toxLabel)

        // Simple developability score
        val d = mol.descriptors
        val solScore = (-(predLogS + 1.0f) * 20f).coerceIn(0f, 40f)
        val toxScore = ((1f - toxProb) * 40f).coerceIn(0f, 40f)
        val lipinskiBonus =
            if (d.mw < 500 && d.logp < 5 && d.hbd <= 5 && d.hba <= 10) 20f else 0f
        val devScore = (solScore + toxScore + lipinskiBonus)
            .toInt()
            .coerceIn(0, 100)

        scoreBar.progress = devScore
        scoreText.text = "Developability score: $devScore / 100"

        // Descriptor bars
        barMw.progress = d.mw.coerceIn(0.0, barMw.max.toDouble()).toInt()
        barLogp.progress = d.logp.coerceIn(0.0, barLogp.max.toDouble()).toInt()
        barTpsa.progress = d.tpsa.coerceIn(0.0, barTpsa.max.toDouble()).toInt()

        textMw.text = "%.1f".format(d.mw)
        textLogp.text = "%.2f".format(d.logp)
        textTpsa.text = "%.1f".format(d.tpsa)

        // --- Update in-memory history ---
        predictionHistory.add(
            PredictionRecord(
                molName = mol.name,
                logS = predLogS,
                toxProb = toxProb,
                devScore = devScore
            )
        )
        // keep last 30 entries
        if (predictionHistory.size > 30) {
            predictionHistory.removeAt(0)
        }
    }

    // ---------------------------------------------------
    // Load 2D molecule image from assets/mol_imgs
    // ---------------------------------------------------
    private fun loadMoleculeImage(mol: DemoMolecule) {
        val idKey = mol.id.toString().lowercase()
        val nameKey = mol.name.lowercase()
        val nameUnderscore = nameKey.replace(" ", "_")

        val candidates = mutableListOf(
            "${mol.id}.png",
            "${mol.name}.png"
        )

        try {
            val allFiles = assets.list("mol_imgs")?.toList().orEmpty()
            val imageFiles = allFiles.filter { file ->
                file.endsWith(".png", ignoreCase = true) ||
                        file.endsWith(".jpg", ignoreCase = true) ||
                        file.endsWith(".jpeg", ignoreCase = true)
            }

            val fuzzyMatches = imageFiles.filter { file ->
                val lower = file.lowercase()
                lower.contains(idKey) ||
                        lower.contains(nameKey) ||
                        lower.contains(nameUnderscore)
            }

            candidates.addAll(fuzzyMatches)
        } catch (_: Exception) {
            // ignore; fall back to basic list
        }

        for (file in candidates.distinct()) {
            try {
                assets.open("mol_imgs/$file").use { input ->
                    val bmp = BitmapFactory.decodeStream(input)
                    if (bmp != null) {
                        molImageView.setImageBitmap(bmp)
                        molImageView.contentDescription = "2D structure of ${mol.name}"
                        return
                    }
                }
            } catch (_: Exception) {
                // try next
            }
        }

        molImageView.setImageDrawable(null)
        molImageView.contentDescription = "No structure available for ${mol.name}"
    }

    // ---------------------------------------------------
    // On-device latency benchmark
    // ---------------------------------------------------
    private fun runBenchmark(runs: Int) {
        val mol = currentMol
        if (mol == null) {
            Toast.makeText(this, "Select a molecule first.", Toast.LENGTH_SHORT).show()
            return
        }

        val features = mol.features

        benchmarkButton.isEnabled = false
        benchmarkText.text = "Running $runs inferences on-device..."

        Thread {
            try {
                // Warm-up
                qsar.predict(features)

                val timesMs = FloatArray(runs)

                for (i in 0 until runs) {
                    val t0 = SystemClock.elapsedRealtimeNanos()
                    qsar.predict(features)
                    val t1 = SystemClock.elapsedRealtimeNanos()
                    timesMs[i] = (t1 - t0) / 1_000_000f
                }

                timesMs.sort()
                val avg = timesMs.average().toFloat()
                val p50 = timesMs[runs / 2]
                val p90 = timesMs[(runs * 0.9f).toInt().coerceAtMost(runs - 1)]
                val min = timesMs.first()
                val max = timesMs.last()

                val summary = """
                    Device: ${android.os.Build.MODEL} (${android.os.Build.HARDWARE})
                    ABI: ${android.os.Build.SUPPORTED_ABIS.joinToString()}
                    Cores: ${Runtime.getRuntime().availableProcessors()}
                    
                    Inference runs: $runs
                    Avg: ${"%.2f".format(avg)} ms
                    p50: ${"%.2f".format(p50)} ms
                    p90: ${"%.2f".format(p90)} ms
                    Min / Max: ${"%.2f".format(min)} / ${"%.2f".format(max)} ms
                """.trimIndent()

                runOnUiThread {
                    benchmarkButton.isEnabled = true
                    benchmarkText.text = summary
                }
            } catch (e: Exception) {
                runOnUiThread {
                    benchmarkButton.isEnabled = true
                    benchmarkText.text = "Benchmark failed: ${e.message}"
                }
            }
        }.start()
    }

    // ---------------------------------------------------
    // History dialog
    // ---------------------------------------------------
    private fun showHistoryDialog() {
        if (predictionHistory.isEmpty()) {
            Toast.makeText(this, "No predictions yet.", Toast.LENGTH_SHORT).show()
            return
        }

        val items = predictionHistory
            .asReversed()
            .map {
                "${it.molName} | logS=%.3f  tox=%.3f  score=${it.devScore}"
                    .format(it.logS, it.toxProb)
            }
            .toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("Prediction history")
            .setItems(items, null)
            .setPositiveButton("Close", null)
            .show()
    }

    // ---------------------------------------------------
    // Molecule picker dialog
    // ---------------------------------------------------
    private fun showMoleculePicker() {
        if (molecules.isEmpty()) {
            Toast.makeText(this, "No molecules loaded.", Toast.LENGTH_SHORT).show()
            return
        }

        val names = molecules.map { it.name }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("Browse molecules")
            .setItems(names) { _, which ->
                val mol = molecules[which]
                currentMol = mol
                runPredictionFor(mol)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ── 3D Space: receives molecule selected in GL view ──────────
    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: android.content.Intent?
    ) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101 && resultCode == RESULT_OK) {
            val molId = data?.getIntExtra("selectedMolId", -1) ?: return
            val mol   = molecules.firstOrNull { it.id == molId } ?: return
            currentMol = mol
            runPredictionFor(mol)
        }
    }

    // ── 3D Space button ───────────────────────────────────────────
    private fun open3dSpace() {
        startActivityForResult(
            android.content.Intent(this, MolecularSpaceActivity::class.java),
            101
        )
    }

    // ── SME2 Benchmark button ─────────────────────────────────────
    private fun openSme2Benchmark() {
        startActivity(
            android.content.Intent(this, Sme2BenchmarkActivity::class.java)
        )
    }

}
