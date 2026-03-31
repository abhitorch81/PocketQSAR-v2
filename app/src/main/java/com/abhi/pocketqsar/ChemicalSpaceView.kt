package com.abhi.pocketqsar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.max
import kotlin.math.min

class ChemicalSpaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var molecules: List<DemoMolecule> = emptyList()
        set(value) {
            field = value
            computeBounds()
            invalidate()
        }

    var selectedId: Int? = null
        set(value) {
            field = value
            invalidate()
        }

    // NEW: callback into Activity when user taps a molecule
    var onMoleculeSelected: ((DemoMolecule) -> Unit)? = null

    private val toxicPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#CC4444")
    }
    private val safePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#44AA66")
    }
    private val selectedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.BLACK
    }
    private val edgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = Color.parseColor("#666666")
    }

    private var minX = 0f
    private var maxX = 1f
    private var minY = 0f
    private var maxY = 1f

    private fun computeBounds() {
        if (molecules.isEmpty()) return
        minX = Float.POSITIVE_INFINITY
        minY = Float.POSITIVE_INFINITY
        maxX = Float.NEGATIVE_INFINITY
        maxY = Float.NEGATIVE_INFINITY
        for (m in molecules) {
            val (x, y) = m.embed2d
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }
        if (minX == maxX) { minX -= 1f; maxX += 1f }
        if (minY == maxY) { minY -= 1f; maxY += 1f }
    }

    private fun mapX(x: Float): Float {
        val pad = width * 0.08f
        return pad + (x - minX) / (maxX - minX) * (width - 2 * pad)
    }

    private fun mapY(y: Float): Float {
        val pad = height * 0.08f
        return height - pad - (y - minY) / (maxY - minY) * (height - 2 * pad)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (molecules.isEmpty()) return

        val selId = selectedId

        // 1) Draw neighbor edges for selected molecule
        if (selId != null) {
            val center = molecules.find { it.id == selId }
            if (center != null) {
                val (cx, cy) = center.embed2d
                val sx = mapX(cx)
                val sy = mapY(cy)
                for (nid in center.neighbors) {
                    val n = molecules.find { it.id == nid } ?: continue
                    val (nx, ny) = n.embed2d
                    val ex = mapX(nx)
                    val ey = mapY(ny)
                    canvas.drawLine(sx, sy, ex, ey, edgePaint)
                }
            }
        }

        // 2) Draw all molecules as dots
        val radius = max(4f, min(width, height) * 0.012f)

        for (m in molecules) {
            val (x, y) = m.embed2d
            val px = mapX(x)
            val py = mapY(y)
            val paint = if (m.labels.toxicity == 1) toxicPaint else safePaint
            canvas.drawCircle(px, py, radius, paint)

            if (m.id == selId) {
                canvas.drawCircle(px, py, radius * 1.7f, selectedPaint)
            }
        }
    }

    // NEW: tap to select a molecule
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_DOWN ||
            event.action == MotionEvent.ACTION_UP
        ) {
            val x = event.x
            val y = event.y

            if (molecules.isEmpty()) return super.onTouchEvent(event)

            val radius = max(4f, min(width, height) * 0.012f) * 2f
            val radius2 = radius * radius

            var best: DemoMolecule? = null
            var bestDist2 = Float.MAX_VALUE

            for (m in molecules) {
                val (fx, fy) = m.embed2d
                val px = mapX(fx)
                val py = mapY(fy)
                val dx = x - px
                val dy = y - py
                val d2 = dx * dx + dy * dy
                if (d2 < radius2 && d2 < bestDist2) {
                    bestDist2 = d2
                    best = m
                }
            }

            if (best != null) {
                selectedId = best.id
                onMoleculeSelected?.invoke(best)
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}
