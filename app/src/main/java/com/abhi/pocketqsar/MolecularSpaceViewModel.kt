package com.abhi.pocketqsar

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray

/**
 * Mol3D — shared data class used by both ViewModel and Renderer.
 * Defined here (top level) so it's visible to all files in the package
 * without needing cross-file nested class references.
 */
data class Mol3D(
    val id:       Int,
    val name:     String,
    val logS:     Float,
    val toxicity: Int,
    val devScore: Int,
    val mw:       Float,
    val logP:     Float,
    val x:        Float,
    val y:        Float,
    val z:        Float,
)

class MolecularSpaceViewModel : ViewModel() {

    private val _molecules3d = MutableStateFlow<List<Mol3D>>(emptyList())
    val molecules3d: StateFlow<List<Mol3D>> = _molecules3d.asStateFlow()

    private val _selectedMol = MutableStateFlow<Mol3D?>(null)
    val selectedMol: StateFlow<Mol3D?> = _selectedMol.asStateFlow()

    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _colorMode = MutableStateFlow(ColorMode.TOXICITY)
    val colorMode: StateFlow<ColorMode> = _colorMode.asStateFlow()

    enum class ColorMode { TOXICITY, SOLUBILITY, DEVELOPABILITY }

    fun loadMolecules(context: Context) {
        if (_molecules3d.value.isNotEmpty()) return
        viewModelScope.launch {
            _isLoading.value = true
            val mols = withContext(Dispatchers.IO) { parseMolecules(context) }
            _molecules3d.value = mols
            _isLoading.value   = false
        }
    }

    fun selectMolecule(id: Int) {
        _selectedMol.value = _molecules3d.value.firstOrNull { it.id == id }
    }

    fun setColorMode(mode: ColorMode) { _colorMode.value = mode }

    private fun parseMolecules(ctx: Context): List<Mol3D> {
        val json = ctx.assets.open("mobile_molecules.json").bufferedReader().readText()
        val arr  = JSONArray(json)

        val rawX    = FloatArray(arr.length())
        val rawY    = FloatArray(arr.length())
        val rawLogS = FloatArray(arr.length())

        for (i in 0 until arr.length()) {
            val mol   = arr.getJSONObject(i)
            val embed = mol.getJSONArray("embed2d")
            rawX[i]    = embed.getDouble(0).toFloat()
            rawY[i]    = embed.getDouble(1).toFloat()
            rawLogS[i] = mol.getJSONObject("labels").getDouble("logS").toFloat()
        }

        val xMin = rawX.min(); val xMax = rawX.max()
        val yMin = rawY.min(); val yMax = rawY.max()
        val lMin = rawLogS.min(); val lMax = rawLogS.max()
        fun norm(v: Float, lo: Float, hi: Float) = 2f * (v - lo) / (hi - lo) - 1f

        val result = mutableListOf<Mol3D>()
        for (i in 0 until arr.length()) {
            val mol    = arr.getJSONObject(i)
            val desc   = mol.getJSONObject("descriptors")
            val labels = mol.getJSONObject("labels")
            val logS   = labels.getDouble("logS").toFloat()
            val tox    = labels.getInt("toxicity")
            val mw     = desc.getDouble("mw").toFloat()
            val logP   = desc.getDouble("logp").toFloat()
            val tpsa   = desc.getDouble("tpsa").toFloat()
            val dev    = computeDevScore(logS, tox, mw, logP, tpsa,
                             desc.getInt("hbd"), desc.getInt("hba"))
            result += Mol3D(
                id = mol.getInt("id"), name = mol.getString("name"),
                logS = logS, toxicity = tox, devScore = dev, mw = mw, logP = logP,
                x = norm(rawX[i], xMin, xMax),
                y = norm(rawY[i], yMin, yMax),
                z = norm(rawLogS[i], lMin, lMax),
            )
        }
        return result
    }

    private fun computeDevScore(
        logS: Float, tox: Int, mw: Float, logP: Float,
        tpsa: Float, hbd: Int, hba: Int
    ): Int {
        var score = 50
        score += when { logS > 0f -> 20; logS > -2f -> 12; logS > -4f -> 0; else -> -15 }
        score -= if (tox == 1) 25 else 0
        if (mw <= 500f && logP <= 5f && hbd <= 5 && hba <= 10) score += 10
        if (tpsa in 20f..130f) score += 5
        return score.coerceIn(0, 100)
    }
}
