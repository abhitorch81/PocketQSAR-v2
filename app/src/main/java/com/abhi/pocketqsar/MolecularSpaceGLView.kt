package com.abhi.pocketqsar

import android.content.Context
import android.opengl.GLSurfaceView
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import kotlin.math.abs

/**
 * MolecularSpaceGLView.kt
 *
 * GLSurfaceView hosting the 3D chemical space.
 *
 * Gestures:
 *   Single finger drag   → rotate
 *   Two finger drag      → pan
 *   Pinch                → zoom
 *   Single tap           → select molecule (triggers prediction)
 *   Double tap           → reset camera
 *   Long press           → toggle auto-rotate
 *
 * Drop into any layout:
 *   <com.abhi.pocketqsar.MolecularSpaceGLView
 *       android:id="@+id/glMolecularSpace"
 *       android:layout_width="match_parent"
 *       android:layout_height="0dp"
 *       android:layout_weight="1" />
 */
class MolecularSpaceGLView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : GLSurfaceView(context, attrs) {

    val renderer: MolecularSpaceRenderer

    // Callback to MainActivity/ViewModel when user taps a molecule
    var onMoleculeSelected: ((Int) -> Unit)? = null

    // Camera defaults
    private val DEFAULT_ROT_X = 20f
    private val DEFAULT_ROT_Y = -30f
    private val DEFAULT_ZOOM  = 3.2f

    // Gesture state
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isRotating = false

    private val gestureDetector: GestureDetector
    private val scaleDetector:   ScaleGestureDetector

    init {
        setEGLContextClientVersion(3)
        setEGLConfigChooser(8, 8, 8, 8, 16, 0)

        renderer = MolecularSpaceRenderer { molId ->
            // Post to main thread
            post { onMoleculeSelected?.invoke(molId) }
        }
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
        preserveEGLContextOnPause = true

        gestureDetector = GestureDetector(context, object :
            GestureDetector.SimpleOnGestureListener() {

            override fun onSingleTapUp(e: MotionEvent): Boolean {
                // Stop auto-rotate on tap so user can inspect
                renderer.autoRotate = false
                queueEvent {
                    renderer.pickMolecule(e.x, e.y)
                }
                return true
            }

            override fun onDoubleTap(e: MotionEvent): Boolean {
                resetCamera()
                return true
            }

            override fun onLongPress(e: MotionEvent) {
                // Toggle auto-rotate
                val current = renderer.autoRotate
                renderer.autoRotate = !current
            }
        })

        scaleDetector = ScaleGestureDetector(context,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    queueEvent {
                        renderer.zoom /= detector.scaleFactor
                        renderer.zoom = renderer.zoom.coerceIn(1.2f, 8f)
                    }
                    return true
                }
            })
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        gestureDetector.onTouchEvent(event)
        scaleDetector.onTouchEvent(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.x
                lastTouchY = event.y
                isRotating = true
            }

            MotionEvent.ACTION_POINTER_DOWN -> {
                // Two-finger gesture → switch to pan mode
                isRotating = false
                lastTouchX = event.getX(0)
                lastTouchY = event.getY(0)
            }

            MotionEvent.ACTION_MOVE -> {
                val dx = event.x - lastTouchX
                val dy = event.y - lastTouchY
                lastTouchX = event.x
                lastTouchY = event.y

                if (event.pointerCount == 1 && isRotating) {
                    // Rotate
                    queueEvent {
                        renderer.rotY += dx * 0.4f
                        renderer.rotX += dy * 0.4f
                        renderer.rotX = renderer.rotX.coerceIn(-85f, 85f)
                    }
                } else if (event.pointerCount == 2) {
                    // Pan — use midpoint of two fingers
                    val midX = (event.getX(0) + event.getX(1)) / 2f
                    val midY = (event.getY(0) + event.getY(1)) / 2f
                    val pdx  = midX - lastTouchX
                    val pdy  = midY - lastTouchY
                    queueEvent {
                        renderer.panX += pdx * 0.004f
                        renderer.panY -= pdy * 0.004f
                    }
                }
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                isRotating = false
            }
        }

        return true
    }

    fun resetCamera() {
        queueEvent {
            renderer.rotX     = DEFAULT_ROT_X
            renderer.rotY     = DEFAULT_ROT_Y
            renderer.zoom     = DEFAULT_ZOOM
            renderer.panX     = 0f
            renderer.panY     = 0f
            renderer.autoRotate = true
        }
    }

    /** Called by ViewModel when molecule data is loaded. */
    fun setMolecules(mols: List<Mol3D>) {
        queueEvent { renderer.setMolecules(mols) }
    }

    /** Highlight the currently selected molecule. */
    fun highlightMolecule(id: Int) {
        queueEvent { renderer.setSelectedId(id) }
    }
}
