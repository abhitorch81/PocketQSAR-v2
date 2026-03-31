package com.abhi.pocketqsar



import android.content.Context
import java.io.File
import java.io.FileOutputStream

object ModelFileHelper {
    fun getModelFilePath(context: Context, assetName: String): String {
        val outFile = File(context.filesDir, assetName)
        if (!outFile.exists()) {
            outFile.parentFile?.mkdirs()
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(8 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read <= 0) break
                        output.write(buffer, 0, read)
                    }
                }
            }
        }
        return outFile.absolutePath
    }
}


