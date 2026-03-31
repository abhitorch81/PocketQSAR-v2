package com.abhi.pocketqsar

import android.content.Context
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class QsarExecuTorch(
    context: Context,
    private val inputDim: Int = 2048
) {

    private val module: Module

    init {
        val modelPath = ModelFileHelper.getModelFilePath(
            context,
            "qsar_net_xnnpack.pte"
        )
        module = Module.load(modelPath)
    }

    fun predict(features: FloatArray): Pair<Float, Float> {
        require(features.size >= inputDim) {
            "Expected at least $inputDim features, got ${features.size}"
        }

        // Use only the first 2048 entries – what the model was trained on
        val modelInput = if (features.size == inputDim) {
            features
        } else {
            features.copyOfRange(0, inputDim)
        }

        val inputTensor = Tensor.fromBlob(
            modelInput,
            longArrayOf(1L, inputDim.toLong())
        )

        val inputEValue = EValue.from(inputTensor)
        val outputs: Array<EValue> = module.forward(inputEValue)

        val solTensor = outputs[0].toTensor()
        val toxTensor = outputs[1].toTensor()

        val solArray = solTensor.dataAsFloatArray
        val toxLogitArray = toxTensor.dataAsFloatArray

        val sol = solArray[0]
        val toxLogit = toxLogitArray[0]

        val toxProb =
            (1.0f / (1.0f + kotlin.math.exp(-toxLogit.toDouble()))).toFloat()

        return sol to toxProb
    }

}
