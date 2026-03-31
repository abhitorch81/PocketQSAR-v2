package com.abhi.pocketqsar

data class Descriptors(
    val mw: Double,
    val logp: Double,
    val tpsa: Double,
    val hbd: Int,
    val hba: Int,
    val rotBonds: Int,
    val aromaticRings: Int
)

data class Labels(
    val logS: Double,
    val toxicity: Int
)

data class DemoMolecule(
    val id: Int,
    val name: String,
    val smiles: String,
    val features: FloatArray,
    val descriptors: Descriptors,
    val labels: Labels,
    val image: String?,
    val embed2d: Pair<Float, Float>,   // will use fallback if missing
    val neighbors: IntArray            // may be empty if missing
)
