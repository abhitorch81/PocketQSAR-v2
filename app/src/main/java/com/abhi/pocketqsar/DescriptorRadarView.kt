package com.abhi.pocketqsar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

class DescriptorRadarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Raw descriptor values
    private var mw: Float = 0f
    private var logp: Float = 0f
    private var tpsa: Float = 0f

    private val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.LTGRAY
        strokeWidth = 3f
        style = Paint.Style.STROKE
    }

    private val polygonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#884673B8") // translucent purple-ish
        style = Paint.Style.FILL
    }

    private val outlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF4673B8")
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.DKGRAY
        textSize = 28f
    }

    /**
     * Update descriptors and redraw.
     */
    fun setValues(mw: Double, logp: Double, tpsa: Double) {
        this.mw = mw.toFloat()
        this.logp = logp.toFloat()
        this.tpsa = tpsa.toFloat()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        val cx = w / 2f
        val cy = h / 2f + 10f
        val radius = min(w, h) * 0.32f

        // 3 axes: MW, logP, TPSA spaced by 120 degrees
        val angles = floatArrayOf(-90f, 30f, 150f)
        val labels = arrayOf("MW", "logP", "TPSA")

        // Axes + labels
        for (i in 0 until 3) {
            val rad = Math.toRadians(angles[i].toDouble())
            val ax = cx + radius * cos(rad).toFloat()
            val ay = cy + radius * sin(rad).toFloat()
            canvas.drawLine(cx, cy, ax, ay, axisPaint)

            val tx = cx + (radius + 24f) * cos(rad).toFloat()
            val ty = cy + (radius + 24f) * sin(rad).toFloat()
            canvas.drawText(labels[i], tx - 30f, ty, textPaint)
        }

        // Normalize values into [0,1]
        val mwNorm = (mw / 800f).coerceIn(0f, 1f)
        val logpNorm = (logp / 8f).coerceIn(0f, 1f)   // assume 0–8 range
        val tpsaNorm = (tpsa / 200f).coerceIn(0f, 1f)

        val values = floatArrayOf(mwNorm, logpNorm, tpsaNorm)

        val path = Path()
        for (i in 0 until 3) {
            val rad = Math.toRadians(angles[i].toDouble())
            val r = radius * values[i]
            val px = cx + r * cos(rad).toFloat()
            val py = cy + r * sin(rad).toFloat()
            if (i == 0) {
                path.moveTo(px, py)
            } else {
                path.lineTo(px, py)
            }
        }
        path.close()

        canvas.drawPath(path, polygonPaint)
        canvas.drawPath(path, outlinePaint)
    }
}

