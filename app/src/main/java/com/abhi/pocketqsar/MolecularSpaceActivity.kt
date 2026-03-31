package com.abhi.pocketqsar

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class MolecularSpaceActivity : AppCompatActivity() {

    private val vm by lazy { MolecularSpaceViewModel() }

    private lateinit var glView:        MolecularSpaceGLView
    private lateinit var loadingBar:    ProgressBar
    private lateinit var selectionCard: LinearLayout
    private lateinit var tvMolName:     TextView
    private lateinit var tvLogS:        TextView
    private lateinit var tvTox:         TextView
    private lateinit var tvDevScore:    TextView
    private lateinit var btnPredict:    Button
    private lateinit var btnDetails:    Button
    private lateinit var btnNeighbors:  Button
    private lateinit var btnReset:      Button
    private lateinit var tvAxisInfo:    TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildLayout())
        supportActionBar?.apply {
            title = "Chemical space 3D"
            setDisplayHomeAsUpEnabled(true)
        }
        observeState()
        vm.loadMolecules(this)
    }

    override fun onResume()  { super.onResume();  glView.onResume() }
    override fun onPause()   { super.onPause();   glView.onPause()  }
    override fun onSupportNavigateUp(): Boolean { finish(); return true }

    private fun buildLayout(): android.widget.FrameLayout {
        val root = android.widget.FrameLayout(this)
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = android.widget.FrameLayout.LayoutParams(-1, -1)
        }

        loadingBar = ProgressBar(this, null,
            android.R.attr.progressBarStyleHorizontal).apply {
            isIndeterminate = true
            layoutParams = LinearLayout.LayoutParams(-1, 8)
        }
        col.addView(loadingBar)

        glView = MolecularSpaceGLView(this).apply {
            layoutParams = LinearLayout.LayoutParams(-1, 0, 1f)
            onMoleculeSelected = { id -> vm.selectMolecule(id) }
        }
        col.addView(glView)

        tvAxisInfo = TextView(this).apply {
            text = "X: PCA1  ·  Y: PCA2  ·  Z: logS   |   Drag to rotate  ·  Pinch to zoom  ·  Tap to select"
            textSize = 10.5f
            setPadding(16, 6, 16, 6)
            setTextColor(0xFFAAAAAA.toInt())
            setBackgroundColor(0xFF111118.toInt())
        }
        col.addView(tvAxisInfo, LinearLayout.LayoutParams(-1, -2))
        col.addView(buildLegend(), LinearLayout.LayoutParams(-1, -2))

        selectionCard = buildSelectionCard()
        selectionCard.visibility = View.GONE
        col.addView(selectionCard, LinearLayout.LayoutParams(-1, -2))
        root.addView(col)

        btnReset = Button(this).apply {
            text = "Reset"
            textSize = 11f
            setPadding(12, 4, 12, 4)
            setOnClickListener { glView.resetCamera() }
            layoutParams = android.widget.FrameLayout.LayoutParams(-2, -2).apply {
                gravity = android.view.Gravity.TOP or android.view.Gravity.END
                topMargin = 8; rightMargin = 12
            }
        }
        root.addView(btnReset)
        return root
    }

    private fun buildLegend(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(20, 10, 20, 10)
            setBackgroundColor(0xFF0D0D15.toInt())
            gravity = android.view.Gravity.CENTER_VERTICAL
            fun dot(color: Int) = View(context).apply {
                setBackgroundColor(color)
                layoutParams = LinearLayout.LayoutParams(14, 14).apply { rightMargin = 6 }
            }
            fun label(txt: String) = TextView(context).apply {
                text = txt; textSize = 11f
                setTextColor(0xFFCCCCCC.toInt())
                layoutParams = LinearLayout.LayoutParams(-2, -2).apply { rightMargin = 20 }
            }
            addView(dot(0xFF1D9E75.toInt())); addView(label("Non-toxic + drug-like"))
            addView(dot(0xFF639922.toInt())); addView(label("Non-toxic"))
            addView(dot(0xFFBA7517.toInt())); addView(label("Toxic"))
            addView(dot(0xFFE24B4A.toInt())); addView(label("Toxic + insoluble"))
            addView(label("• size = dev score"))
        }
    }

    private fun buildSelectionCard(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 14, 20, 14)
            setBackgroundColor(0xFF12121E.toInt())

            val topRow = LinearLayout(this@MolecularSpaceActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
            }
            tvMolName = TextView(this@MolecularSpaceActivity).apply {
                textSize = 16f; setTextColor(0xFFFFFFFF.toInt())
                layoutParams = LinearLayout.LayoutParams(0, -2, 1f)
            }
            topRow.addView(tvMolName)
            tvDevScore = TextView(this@MolecularSpaceActivity).apply {
                textSize = 13f; setTextColor(0xFF1D9E75.toInt())
            }
            topRow.addView(tvDevScore)
            addView(topRow, LinearLayout.LayoutParams(-1, -2))

            val propRow = LinearLayout(this@MolecularSpaceActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, 6, 0, 10)
            }
            fun propTv() = TextView(this@MolecularSpaceActivity).apply {
                textSize = 12f; setTextColor(0xFFAAAAAA.toInt())
                layoutParams = LinearLayout.LayoutParams(-2, -2).apply { rightMargin = 20 }
            }
            tvLogS = propTv(); propRow.addView(tvLogS)
            tvTox  = propTv(); propRow.addView(tvTox)
            addView(propRow)

            val btnRow = LinearLayout(this@MolecularSpaceActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.END
            }
            fun actionBtn(label: String, click: () -> Unit) =
                Button(this@MolecularSpaceActivity).apply {
                    text = label; textSize = 12f
                    setPadding(16, 6, 16, 6)
                    layoutParams = LinearLayout.LayoutParams(-2, -2).apply { leftMargin = 8 }
                    setOnClickListener { click() }
                }
            btnPredict   = actionBtn("Predict")   { runPrediction() }
            btnDetails   = actionBtn("Details")   { openDetails() }
            btnNeighbors = actionBtn("Neighbors") { showNeighbors() }
            btnRow.addView(btnPredict)
            btnRow.addView(btnDetails)
            btnRow.addView(btnNeighbors)
            addView(btnRow)
        }
    }

    private fun observeState() {
        lifecycleScope.launch {
            vm.molecules3d.collectLatest { mols ->
                if (mols.isNotEmpty()) {
                    glView.setMolecules(mols)
                    loadingBar.visibility = View.GONE
                }
            }
        }
        lifecycleScope.launch {
            vm.isLoading.collectLatest { loading ->
                loadingBar.visibility = if (loading) View.VISIBLE else View.GONE
            }
        }
        lifecycleScope.launch {
            vm.selectedMol.collectLatest { mol ->
                mol ?: return@collectLatest
                glView.highlightMolecule(mol.id)
                renderSelectionCard(mol)
                selectionCard.visibility = View.VISIBLE
            }
        }
    }

    private fun renderSelectionCard(mol: Mol3D) {
        tvMolName.text  = mol.name
        tvDevScore.text = "Dev ${mol.devScore}/100"
        tvLogS.text     = "logS ${"%.2f".format(mol.logS)}"
        tvTox.text      = if (mol.toxicity == 0) "Non-toxic" else "Toxic"
        tvTox.setTextColor(
            if (mol.toxicity == 0) 0xFF1D9E75.toInt() else 0xFFE24B4A.toInt()
        )
    }

    private fun runPrediction() {
        val mol = vm.selectedMol.value ?: return
        val intent = Intent()
        intent.putExtra("selectedMolId", mol.id)
        setResult(RESULT_OK, intent)
        finish()
    }

    private fun openDetails() {
        val mol = vm.selectedMol.value ?: return
        Toast.makeText(this,
            "${mol.name}\nMW: ${mol.mw.toInt()} Da  logP: ${"%.2f".format(mol.logP)}\n" +
            "logS: ${"%.2f".format(mol.logS)}  Tox: ${if (mol.toxicity == 0) "Low" else "High"}",
            Toast.LENGTH_LONG).show()
    }

    private fun showNeighbors() {
        val mol = vm.selectedMol.value ?: return
        val neighbors = vm.molecules3d.value
            .filter { it.id != mol.id }
            .sortedBy { n ->
                val dx = n.x - mol.x; val dy = n.y - mol.y; val dz = n.z - mol.z
                dx*dx + dy*dy + dz*dz
            }.take(3)
        Toast.makeText(this,
            "Nearest to ${mol.name}:\n" + neighbors.joinToString("\n") {
                "  ${it.name}  logS=${"%.2f".format(it.logS)}"
            }, Toast.LENGTH_LONG).show()
        neighbors.forEach { n ->
            glView.queueEvent { glView.renderer.setSelectedId(n.id) }
        }
    }
}
