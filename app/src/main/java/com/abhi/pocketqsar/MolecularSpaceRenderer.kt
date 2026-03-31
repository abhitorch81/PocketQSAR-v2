package com.abhi.pocketqsar

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.*

/**
 * MolecularSpaceRenderer.kt
 *
 * OpenGL ES 3.0 renderer for the 3D chemical space view.
 *
 * Axes:
 *   X = PCA embed2d[0]  (horizontal chemical structure space)
 *   Y = embed2d[1]      (vertical chemical structure space)
 *   Z = logS            (aqueous solubility — most meaningful 3rd axis)
 *
 * Visual encoding:
 *   Color  = toxicity (green=non-toxic, red=toxic, amber=borderline)
 *   Size   = developability score (bigger = more drug-like)
 *   Glow   = selected molecule (white halo ring)
 *   Grid   = faint reference plane at Z=0 (logS=0 boundary)
 *
 * KleidiAI connection:
 *   The 3D positions are computed by the batch inference pass in
 *   Sme2Benchmark — all 100 molecules scored in one GEMM call,
 *   then positions updated from the resulting logS values.
 */
class MolecularSpaceRenderer(
    private val onMoleculeSelected: (Int) -> Unit
) : GLSurfaceView.Renderer {

    // ── GL program handles ────────────────────────────────────────────────────
    private var sphereProgram  = 0
    private var glowProgram    = 0
    private var gridProgram    = 0
    private var axisProgram    = 0

    // ── Geometry buffers ──────────────────────────────────────────────────────
    private lateinit var sphereVBO:    FloatBuffer   // instanced sphere vertices
    private lateinit var instanceVBO:  FloatBuffer   // per-molecule: xyz, color, size
    private lateinit var gridVBO:      FloatBuffer
    private lateinit var axisVBO:      FloatBuffer

    // ── Matrices ──────────────────────────────────────────────────────────────
    private val projMatrix  = FloatArray(16)
    private val viewMatrix  = FloatArray(16)
    private val mvpMatrix   = FloatArray(16)
    private val tempMatrix  = FloatArray(16)

    // ── Camera state ──────────────────────────────────────────────────────────
    var rotX = 20f          // pitch — tilted slightly down by default
    var rotY = -30f         // yaw
    var zoom = 3.2f         // camera distance
    var panX = 0f
    var panY = 0f

    // ── Scene data ────────────────────────────────────────────────────────────
    private val molecules  = mutableListOf<Mol3D>()
    private var selectedId = -1
    private var screenW    = 1
    private var screenH    = 1

    // ── Animation ─────────────────────────────────────────────────────────────
    var autoRotate   = true
    private var pulsePhase   = 0f
    private var lastFrameMs  = System.currentTimeMillis()

    // ── Sphere geometry (icosphere, 2 subdivisions, 162 vertices) ─────────────
    private var sphereVertexCount = 0
    private var sphereVBOHandle   = 0
    private var instanceVBOHandle = 0
    private var sphereVAO         = 0

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────


    fun setMolecules(mols: List<Mol3D>) {
        molecules.clear()
        molecules.addAll(mols)
        rebuildInstanceBuffer()
    }

    fun setSelectedId(id: Int) {
        selectedId = id
        rebuildInstanceBuffer()
    }


    /** Pick molecule nearest to a screen tap — call from UI thread. */
    fun pickMolecule(screenX: Float, screenY: Float): Int {
        // Unproject tap to ray, find nearest sphere
        val ndcX =  (2f * screenX / screenW - 1f)
        val ndcY = -(2f * screenY / screenH - 1f)

        var bestId   = -1
        var bestDist = Float.MAX_VALUE

        val invMvp = FloatArray(16)
        Matrix.invertM(invMvp, 0, mvpMatrix, 0)

        // Ray from camera through tap point
        val nearVec = floatArrayOf(ndcX, ndcY, -1f, 1f)
        val farVec  = floatArrayOf(ndcX, ndcY,  1f, 1f)
        val nearW   = FloatArray(4); val farW = FloatArray(4)
        Matrix.multiplyMV(nearW, 0, invMvp, 0, nearVec, 0)
        Matrix.multiplyMV(farW,  0, invMvp, 0, farVec,  0)

        val nx = nearW[0]/nearW[3]; val ny = nearW[1]/nearW[3]; val nz = nearW[2]/nearW[3]
        val fx = farW[0]/farW[3];   val fy = farW[1]/farW[3];   val fz = farW[2]/farW[3]
        val dx = fx-nx; val dy = fy-ny; val dz = fz-nz

        molecules.forEach { mol ->
            // Distance from ray to sphere centre
            val t = ((mol.x-nx)*dx + (mol.y-ny)*dy + (mol.z-nz)*dz) /
                    (dx*dx + dy*dy + dz*dz)
            val cx = nx + t*dx; val cy = ny + t*dy; val cz = nz + t*dz
            val dist2 = (cx-mol.x).pow(2) + (cy-mol.y).pow(2) + (cz-mol.z).pow(2)
            val radius = sphereRadius(mol)
            if (dist2 < (radius * 2.5f).pow(2) && dist2 < bestDist) {
                bestDist = dist2
                bestId   = mol.id
            }
        }
        if (bestId >= 0) onMoleculeSelected(bestId)
        return bestId
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GLSurfaceView.Renderer
    // ─────────────────────────────────────────────────────────────────────────

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0.06f, 0.06f, 0.10f, 1f)   // deep dark background
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        compilePrograms()
        buildSphereGeometry()
        buildGridGeometry()
        buildAxisGeometry()
        rebuildInstanceBuffer()
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        screenW = w; screenH = h
        GLES30.glViewport(0, 0, w, h)
        val aspect = w.toFloat() / h.toFloat()
        Matrix.perspectiveM(projMatrix, 0, 45f, aspect, 0.1f, 20f)
    }

    override fun onDrawFrame(gl: GL10?) {
        val now    = System.currentTimeMillis()
        val dtSec  = (now - lastFrameMs) / 1000f
        lastFrameMs = now

        // Auto-rotate
        if (autoRotate) rotY += 8f * dtSec

        // Pulse for selected molecule
        pulsePhase = (pulsePhase + dtSec * 2f) % (2f * PI.toFloat())

        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)

        buildViewMatrix()
        Matrix.multiplyMM(mvpMatrix, 0, projMatrix, 0, viewMatrix, 0)

        drawGrid()
        drawAxes()
        drawSpheres()
        if (selectedId >= 0) drawGlow()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Drawing
    // ─────────────────────────────────────────────────────────────────────────

    private fun buildViewMatrix() {
        Matrix.setIdentityM(viewMatrix, 0)
        Matrix.translateM(viewMatrix, 0, panX, panY, -zoom)
        Matrix.rotateM(viewMatrix, 0, rotX, 1f, 0f, 0f)
        Matrix.rotateM(viewMatrix, 0, rotY, 0f, 1f, 0f)
    }

    private fun drawSpheres() {
        if (molecules.isEmpty()) return
        GLES30.glUseProgram(sphereProgram)

        val mvpLoc   = GLES30.glGetUniformLocation(sphereProgram, "uMVP")
        val viewLoc  = GLES30.glGetUniformLocation(sphereProgram, "uView")
        val lightLoc = GLES30.glGetUniformLocation(sphereProgram, "uLightDir")
        val selLoc   = GLES30.glGetUniformLocation(sphereProgram, "uSelectedId")

        GLES30.glUniformMatrix4fv(mvpLoc,  1, false, mvpMatrix, 0)
        GLES30.glUniformMatrix4fv(viewLoc, 1, false, viewMatrix, 0)
        // Warm key light from upper-left, cool fill from right
        GLES30.glUniform3f(lightLoc, -0.6f, 0.8f, 0.5f)
        GLES30.glUniform1i(selLoc, selectedId)

        GLES30.glBindVertexArray(sphereVAO)
        GLES30.glDrawArraysInstanced(
            GLES30.GL_TRIANGLES, 0, sphereVertexCount, molecules.size
        )
        GLES30.glBindVertexArray(0)
    }

    private fun drawGlow() {
        val mol = molecules.firstOrNull { it.id == selectedId } ?: return
        GLES30.glUseProgram(glowProgram)
        val mvpLoc   = GLES30.glGetUniformLocation(glowProgram, "uMVP")
        val posLoc   = GLES30.glGetUniformLocation(glowProgram, "uPos")
        val sizeLoc  = GLES30.glGetUniformLocation(glowProgram, "uSize")
        val phaseLoc = GLES30.glGetUniformLocation(glowProgram, "uPhase")
        GLES30.glUniformMatrix4fv(mvpLoc, 1, false, mvpMatrix, 0)
        GLES30.glUniform3f(posLoc, mol.x, mol.y, mol.z)
        GLES30.glUniform1f(sizeLoc, sphereRadius(mol) * 2.8f + sin(pulsePhase) * 0.015f)
        GLES30.glUniform1f(phaseLoc, pulsePhase)
        // Draw a billboard quad (4 vertices)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
    }

    private fun drawGrid() {
        GLES30.glUseProgram(gridProgram)
        val mvpLoc = GLES30.glGetUniformLocation(gridProgram, "uMVP")
        GLES30.glUniformMatrix4fv(mvpLoc, 1, false, mvpMatrix, 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)

        val posAttr = GLES30.glGetAttribLocation(gridProgram, "aPos")
        GLES30.glEnableVertexAttribArray(posAttr)
        gridVBO.position(0)
        GLES30.glVertexAttribPointer(posAttr, 3, GLES30.GL_FLOAT, false, 0, gridVBO)
        GLES30.glLineWidth(1f)
        GLES30.glDrawArrays(GLES30.GL_LINES, 0, gridLineCount)
        GLES30.glDisableVertexAttribArray(posAttr)
    }

    private fun drawAxes() {
        GLES30.glUseProgram(axisProgram)
        val mvpLoc   = GLES30.glGetUniformLocation(axisProgram, "uMVP")
        val colorLoc = GLES30.glGetUniformLocation(axisProgram, "uColor")
        GLES30.glUniformMatrix4fv(mvpLoc, 1, false, mvpMatrix, 0)

        val posAttr = GLES30.glGetAttribLocation(axisProgram, "aPos")
        GLES30.glEnableVertexAttribArray(posAttr)
        axisVBO.position(0)
        GLES30.glVertexAttribPointer(posAttr, 3, GLES30.GL_FLOAT, false, 0, axisVBO)

        // X axis — red (embed X)
        GLES30.glUniform4f(colorLoc, 0.9f, 0.3f, 0.3f, 0.8f)
        GLES30.glDrawArrays(GLES30.GL_LINES, 0, 2)
        // Y axis — green (embed Y)
        GLES30.glUniform4f(colorLoc, 0.3f, 0.9f, 0.3f, 0.8f)
        GLES30.glDrawArrays(GLES30.GL_LINES, 2, 2)
        // Z axis — blue (logS)
        GLES30.glUniform4f(colorLoc, 0.3f, 0.5f, 0.9f, 0.8f)
        GLES30.glDrawArrays(GLES30.GL_LINES, 4, 2)

        GLES30.glDisableVertexAttribArray(posAttr)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Instance buffer — one entry per molecule
    // Layout per instance: x y z r g b a radius  (8 floats)
    // ─────────────────────────────────────────────────────────────────────────

    private fun rebuildInstanceBuffer() {
        if (molecules.isEmpty()) return
        val data = FloatArray(molecules.size * 8)
        molecules.forEachIndexed { i, mol ->
            val c = toxicityColor(mol)
            val radius = sphereRadius(mol)
            val off = i * 8
            data[off+0] = mol.x;   data[off+1] = mol.y;  data[off+2] = mol.z
            data[off+3] = c[0];    data[off+4] = c[1];   data[off+5] = c[2]
            data[off+6] = if (mol.id == selectedId) 1f else 0.85f  // alpha
            data[off+7] = radius
        }
        instanceVBO = ByteBuffer.allocateDirect(data.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        instanceVBO.put(data).position(0)

        // Upload to GPU
        if (instanceVBOHandle == 0) {
            val handles = IntArray(1)
            GLES30.glGenBuffers(1, handles, 0)
            instanceVBOHandle = handles[0]
        }
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, instanceVBOHandle)
        GLES30.glBufferData(
            GLES30.GL_ARRAY_BUFFER,
            data.size * 4,
            instanceVBO,
            GLES30.GL_DYNAMIC_DRAW
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Colour + size helpers
    // ─────────────────────────────────────────────────────────────────────────

    private fun toxicityColor(mol: Mol3D): FloatArray {
        return when {
            mol.toxicity == 0 && mol.devScore >= 70 ->
                floatArrayOf(0.11f, 0.62f, 0.46f)   // teal-green: safe + drug-like
            mol.toxicity == 0 ->
                floatArrayOf(0.39f, 0.60f, 0.14f)   // muted green: safe
            mol.toxicity == 1 && mol.logS < -4f ->
                floatArrayOf(0.88f, 0.30f, 0.29f)   // red: toxic + insoluble
            else ->
                floatArrayOf(0.73f, 0.46f, 0.09f)   // amber: toxic but moderate
        }
    }

    private fun sphereRadius(mol: Mol3D): Float {
        // Size 0.04 (low dev) to 0.09 (high dev)
        return 0.04f + mol.devScore / 100f * 0.05f
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Geometry generation
    // ─────────────────────────────────────────────────────────────────────────

    /** Build a unit icosphere (2 subdivisions = 320 triangles, 162 vertices). */
    private fun buildSphereGeometry() {
        // Start with icosahedron
        val phi = (1f + sqrt(5f)) / 2f
        val base = listOf(
            floatArrayOf(-1f, phi, 0f), floatArrayOf(1f, phi, 0f),
            floatArrayOf(-1f,-phi, 0f), floatArrayOf(1f,-phi, 0f),
            floatArrayOf(0f,-1f, phi), floatArrayOf(0f, 1f, phi),
            floatArrayOf(0f,-1f,-phi), floatArrayOf(0f, 1f,-phi),
            floatArrayOf(phi, 0f,-1f), floatArrayOf(phi, 0f, 1f),
            floatArrayOf(-phi,0f,-1f), floatArrayOf(-phi,0f, 1f),
        ).map { v -> val l = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); v.map{it/l}.toFloatArray() }

        val faces0 = listOf(
            intArrayOf(0,11,5), intArrayOf(0,5,1),  intArrayOf(0,1,7),
            intArrayOf(0,7,10),intArrayOf(0,10,11), intArrayOf(1,5,9),
            intArrayOf(5,11,4),intArrayOf(11,10,2), intArrayOf(10,7,6),
            intArrayOf(7,1,8), intArrayOf(3,9,4),   intArrayOf(3,4,2),
            intArrayOf(3,2,6), intArrayOf(3,6,8),   intArrayOf(3,8,9),
            intArrayOf(4,9,5), intArrayOf(2,4,11),  intArrayOf(6,2,10),
            intArrayOf(8,6,7), intArrayOf(9,8,1),
        )

        fun midpoint(v1: FloatArray, v2: FloatArray): FloatArray {
            val mx = (v1[0]+v2[0])/2f
            val my = (v1[1]+v2[1])/2f
            val mz = (v1[2]+v2[2])/2f
            val l  = sqrt(mx*mx+my*my+mz*mz)
            return floatArrayOf(mx/l, my/l, mz/l)
        }

        var verts = base.toMutableList()
        var faces = faces0

        repeat(2) {   // 2 subdivisions
            val newFaces = mutableListOf<IntArray>()
            faces.forEach { f ->
                val i0 = f[0]; val i1 = f[1]; val i2 = f[2]
                val m01 = verts.size; verts.add(midpoint(verts[i0], verts[i1]))
                val m12 = verts.size; verts.add(midpoint(verts[i1], verts[i2]))
                val m20 = verts.size; verts.add(midpoint(verts[i2], verts[i0]))
                newFaces += intArrayOf(i0, m01, m20)
                newFaces += intArrayOf(i1, m12, m01)
                newFaces += intArrayOf(i2, m20, m12)
                newFaces += intArrayOf(m01, m12, m20)
            }
            faces = newFaces
        }

        // Pack into float buffer: position(xyz) + normal(xyz) = 6 floats/vertex
        val vertexData = FloatArray(faces.size * 3 * 6)
        var idx = 0
        faces.forEach { f ->
            f.forEach { vi ->
                val v = verts[vi]
                // Normal = position on unit sphere
                vertexData[idx++] = v[0]; vertexData[idx++] = v[1]; vertexData[idx++] = v[2]
                vertexData[idx++] = v[0]; vertexData[idx++] = v[1]; vertexData[idx++] = v[2]
            }
        }
        sphereVertexCount = faces.size * 3

        sphereVBO = ByteBuffer.allocateDirect(vertexData.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        sphereVBO.put(vertexData).position(0)

        // Create VAO + sphere VBO + instance VBO
        val vaos = IntArray(1); GLES30.glGenVertexArrays(1, vaos, 0); sphereVAO = vaos[0]
        val vbos = IntArray(2); GLES30.glGenBuffers(2, vbos, 0)
        sphereVBOHandle   = vbos[0]
        instanceVBOHandle = vbos[1]

        GLES30.glBindVertexArray(sphereVAO)

        // Sphere geometry
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, sphereVBOHandle)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER,
            vertexData.size * 4, sphereVBO, GLES30.GL_STATIC_DRAW)
        GLES30.glEnableVertexAttribArray(0)   // aPos
        GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, 24, 0)
        GLES30.glEnableVertexAttribArray(1)   // aNormal
        GLES30.glVertexAttribPointer(1, 3, GLES30.GL_FLOAT, false, 24, 12)

        // Instance data (will be filled by rebuildInstanceBuffer)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, instanceVBOHandle)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, 0, null, GLES30.GL_DYNAMIC_DRAW)
        // aOffset  = attrib 2 (xyz)
        GLES30.glEnableVertexAttribArray(2)
        GLES30.glVertexAttribPointer(2, 3, GLES30.GL_FLOAT, false, 32, 0)
        GLES30.glVertexAttribDivisor(2, 1)
        // aColor   = attrib 3 (rgba)
        GLES30.glEnableVertexAttribArray(3)
        GLES30.glVertexAttribPointer(3, 4, GLES30.GL_FLOAT, false, 32, 12)
        GLES30.glVertexAttribDivisor(3, 1)
        // aRadius  = attrib 4 (float)
        GLES30.glEnableVertexAttribArray(4)
        GLES30.glVertexAttribPointer(4, 1, GLES30.GL_FLOAT, false, 32, 28)
        GLES30.glVertexAttribDivisor(4, 1)

        GLES30.glBindVertexArray(0)
    }

    private var gridLineCount = 0

    private fun buildGridGeometry() {
        // 11×11 grid on Y=0 plane (logS=0 boundary) from -1.2 to 1.2
        val lines = mutableListOf<Float>()
        val r = 1.2f; val step = 0.24f
        var v = -r
        while (v <= r + 0.001f) {
            lines += listOf(-r, 0f, v,  r, 0f, v)   // Z lines
            lines += listOf(v, 0f, -r,  v, 0f, r)   // X lines
            v += step
        }
        gridLineCount = lines.size / 3
        val arr = lines.toFloatArray()
        gridVBO = ByteBuffer.allocateDirect(arr.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        gridVBO.put(arr).position(0)
    }

    private fun buildAxisGeometry() {
        val axisData = floatArrayOf(
            0f, 0f, 0f,  1.3f, 0f, 0f,   // X
            0f, 0f, 0f,  0f, 1.3f, 0f,   // Y
            0f, 0f, 0f,  0f, 0f, 1.3f,   // Z
        )
        axisVBO = ByteBuffer.allocateDirect(axisData.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        axisVBO.put(axisData).position(0)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Shader compilation
    // ─────────────────────────────────────────────────────────────────────────

    private fun compilePrograms() {
        sphereProgram = createProgram(SPHERE_VERT, SPHERE_FRAG)
        glowProgram   = createProgram(GLOW_VERT,   GLOW_FRAG)
        gridProgram   = createProgram(GRID_VERT,   GRID_FRAG)
        axisProgram   = createProgram(AXIS_VERT,   AXIS_FRAG)
    }

    private fun createProgram(vertSrc: String, fragSrc: String): Int {
        val vert = compileShader(GLES30.GL_VERTEX_SHADER,   vertSrc)
        val frag = compileShader(GLES30.GL_FRAGMENT_SHADER, fragSrc)
        return GLES30.glCreateProgram().also { prog ->
            GLES30.glAttachShader(prog, vert)
            GLES30.glAttachShader(prog, frag)
            GLES30.glLinkProgram(prog)
        }
    }

    private fun compileShader(type: Int, src: String): Int {
        return GLES30.glCreateShader(type).also { s ->
            GLES30.glShaderSource(s, src)
            GLES30.glCompileShader(s)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GLSL Shaders
    // ─────────────────────────────────────────────────────────────────────────

    // Sphere: instanced rendering with Phong + rim lighting
    private val SPHERE_VERT = """
        #version 300 es
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec3 aNormal;
        layout(location=2) in vec3 aOffset;    // instance: world position
        layout(location=3) in vec4 aColor;     // instance: rgb + alpha
        layout(location=4) in float aRadius;   // instance: sphere radius

        uniform mat4 uMVP;
        uniform mat4 uView;
        uniform vec3 uLightDir;
        uniform int  uSelectedId;

        out vec3 vNormal;
        out vec3 vColor;
        out float vAlpha;
        out vec3 vViewDir;

        void main() {
            vec3 worldPos = aPos * aRadius + aOffset;
            gl_Position   = uMVP * vec4(worldPos, 1.0);

            vNormal  = normalize(mat3(uView) * aNormal);
            vColor   = aColor.rgb;
            vAlpha   = aColor.a;
            vViewDir = normalize(-(uView * vec4(aOffset, 1.0)).xyz);
        }
    """.trimIndent()

    private val SPHERE_FRAG = """
        #version 300 es
        precision mediump float;

        in vec3 vNormal;
        in vec3 vColor;
        in float vAlpha;
        in vec3 vViewDir;

        out vec4 fragColor;

        void main() {
            vec3 light    = normalize(vec3(-0.6, 0.8, 0.5));
            float diff    = max(dot(vNormal, light), 0.0);
            float rim     = pow(1.0 - max(dot(vNormal, vViewDir), 0.0), 2.5);
            vec3 ambient  = vColor * 0.25;
            vec3 diffuse  = vColor * diff * 0.70;
            vec3 rimLight = vec3(0.6, 0.8, 1.0) * rim * 0.35;
            vec3 spec     = vec3(1.0) * pow(max(dot(reflect(-light,vNormal),vViewDir),0.0), 32.0) * 0.4;

            fragColor = vec4(ambient + diffuse + rimLight + spec, vAlpha);
        }
    """.trimIndent()

    // Glow: billboard quad around selected molecule, pulsing ring
    private val GLOW_VERT = """
        #version 300 es
        uniform mat4  uMVP;
        uniform vec3  uPos;
        uniform float uSize;

        const vec2 quad[4] = vec2[](
            vec2(-1.0,-1.0), vec2(1.0,-1.0),
            vec2(-1.0, 1.0), vec2(1.0, 1.0)
        );

        out vec2 vUV;
        void main() {
            vUV = quad[gl_VertexID];
            vec3 worldPos = uPos + vec3(quad[gl_VertexID] * uSize, 0.0);
            gl_Position = uMVP * vec4(worldPos, 1.0);
        }
    """.trimIndent()

    private val GLOW_FRAG = """
        #version 300 es
        precision mediump float;
        in vec2 vUV;
        uniform float uPhase;
        out vec4 fragColor;
        void main() {
            float d    = length(vUV);
            float ring = smoothstep(0.75, 0.80, d) * smoothstep(1.0, 0.90, d);
            float glow = smoothstep(1.0, 0.0, d) * 0.25;
            float pulse = 0.7 + 0.3 * sin(uPhase);
            fragColor = vec4(1.0, 1.0, 1.0, (ring + glow) * pulse);
        }
    """.trimIndent()

    // Grid: faint reference plane
    private val GRID_VERT = """
        #version 300 es
        in vec3 aPos;
        uniform mat4 uMVP;
        void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
    """.trimIndent()

    private val GRID_FRAG = """
        #version 300 es
        precision mediump float;
        out vec4 fragColor;
        void main() { fragColor = vec4(0.4, 0.45, 0.55, 0.18); }
    """.trimIndent()

    // Axis lines
    private val AXIS_VERT = """
        #version 300 es
        in vec3 aPos;
        uniform mat4 uMVP;
        void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
    """.trimIndent()

    private val AXIS_FRAG = """
        #version 300 es
        precision mediump float;
        uniform vec4 uColor;
        out vec4 fragColor;
        void main() { fragColor = uColor; }
    """.trimIndent()
}
