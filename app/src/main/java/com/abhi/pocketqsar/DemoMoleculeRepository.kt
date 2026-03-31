package com.abhi.pocketqsar

import android.content.Context
import org.json.JSONArray
import java.io.BufferedReader

object DemoMoleculeRepository {

    fun load(context: Context): List<DemoMolecule> {
        val inputStream = context.assets.open("mobile_molecules.json")
        val jsonStr = inputStream.bufferedReader().use(BufferedReader::readText)
        val arr = JSONArray(jsonStr)

        val list = mutableListOf<DemoMolecule>()

        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)

            val featuresJson = obj.getJSONArray("features")
            val features = FloatArray(featuresJson.length()) { idx ->
                featuresJson.getDouble(idx).toFloat()
            }

            val descObj = obj.getJSONObject("descriptors")
            val descriptors = Descriptors(
                mw = descObj.getDouble("mw"),
                logp = descObj.getDouble("logp"),
                tpsa = descObj.getDouble("tpsa"),
                hbd = descObj.getInt("hbd"),
                hba = descObj.getInt("hba"),
                rotBonds = descObj.getInt("rot_bonds"),
                aromaticRings = descObj.getInt("aromatic_rings")
            )

            val labelsObj = obj.getJSONObject("labels")
            val labels = Labels(
                logS = labelsObj.getDouble("logS"),
                toxicity = labelsObj.getInt("toxicity")
            )

            // ---------- NEW: be robust to missing chemical-space fields ----------

            // embed2d: use optJSONArray so we don't crash if it's not present
            val embArr = obj.optJSONArray("embed2d")
            val embed2d: Pair<Float, Float> = if (embArr != null && embArr.length() >= 2) {
                Pair(
                    embArr.getDouble(0).toFloat(),
                    embArr.getDouble(1).toFloat()
                )
            } else {
                // Fallback: simple spread by index + toxicity (so view can still draw)
                Pair(i.toFloat(), labels.toxicity.toFloat())
            }

            // neighbors: may be missing → empty array
            val neighJson = obj.optJSONArray("neighbors")
            val neighbors: IntArray = if (neighJson != null) {
                IntArray(neighJson.length()) { idx ->
                    neighJson.getInt(idx)
                }
            } else {
                IntArray(0)
            }

            // --------------------------------------------------------------------

            val molecule = DemoMolecule(
                id = obj.getInt("id"),
                name = obj.getString("name"),
                smiles = obj.getString("smiles"),
                features = features,
                descriptors = descriptors,
                labels = labels,
                image = obj.optString("image", null),
                embed2d = embed2d,
                neighbors = neighbors
            )

            list += molecule
        }

        return list
    }
}
