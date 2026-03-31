/**
 * Sme2Benchmark.cpp  —  PocketQSAR SME2 extension
 *
 * Drops into the existing PocketQSAR Android project alongside
 * QsarExecuTorch.kt without touching any original files.
 *
 * Matches the real project:
 *   • Feature vectors: 2055-dim (2048 Morgan FP + 7 RDKit descriptors)
 *   • Model: QSARNet  input(2048) → Linear(512) → ReLU → Linear(256) → ReLU
 *             → solubility_head(3)  +  toxicity_head(2)
 *   • Dataset: ESOL 100 molecules from mobile_molecules.json
 *
 * What this adds:
 *   1. HWCAP2_SME2 runtime detection
 *   2. Baseline NEON matmul (fp32, mirrors XNNPACK's path on non-SME2 devices)
 *   3. SME2 outer-product matmul (FMOPA into ZA tile)
 *   4. Paired benchmark: runs both kernels, computes full latency statistics
 *   5. Per-operator breakdown sized to QSARNet layer dimensions
 *   6. JNI surface — all functions named for Sme2Benchmark Kotlin companion
 *
 * Build requirements (add to existing app/build.gradle):
 *   externalNativeBuild { cmake { path "src/main/cpp/CMakeLists.txt" } }
 *   defaultConfig { externalNativeBuild { cmake { abiFilters "arm64-v8a" } } }
 *
 * NDK r26b+ required for <arm_sme.h>.
 */

#include <jni.h>
#include <android/log.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <arm_neon.h>

#if __has_include(<arm_sme.h>)
#  include <arm_sme.h>
#  define HAS_SME2 1
#else
#  define HAS_SME2 0
#endif

#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <sstream>

#define TAG "PocketQSAR_SME2"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Hardware detection
// ─────────────────────────────────────────────────────────────────────────────

#ifndef HWCAP2_SME2
#define HWCAP2_SME2 (1UL << 37)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1UL << 1)
#endif

struct Caps { bool sme2; bool sve2; bool neon; uint32_t svl_bits; };
static Caps g_caps{};
static bool g_caps_ready = false;

static const Caps& caps() {
    if (g_caps_ready) return g_caps;
    g_caps_ready = true;
    unsigned long hw2 = getauxval(AT_HWCAP2);
    g_caps.neon     = true;
    g_caps.sve2     = (hw2 & HWCAP2_SVE2) != 0;
    g_caps.sme2     = (hw2 & HWCAP2_SME2) != 0;
    g_caps.svl_bits = 0;
#if HAS_SME2
    if (g_caps.sme2) {
        __arm_start_streaming();
        g_caps.svl_bits = (uint32_t)(svcntsw() * 32);
        __arm_stop_streaming();
    }
#endif
    LOGI("caps: sme2=%d sve2=%d svl=%u-bit", g_caps.sme2, g_caps.sve2, g_caps.svl_bits);
    return g_caps;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  QSARNet layer dimensions (matches the real model)
//
//   Layer            M   K     N    (C = A[M×K] × B[K×N])
//   ─────────────────────────────────────────────────────
//   Input → FC1      1  2048  512   (batch=1 inference)
//   FC1 → FC2        1   512  256
//   FC2 → sol_head   1   256    3
//   FC2 → tox_head   1   256    2
//
// For the *benchmark* we use batch=32 to make timing measurable.
// ─────────────────────────────────────────────────────────────────────────────

struct LayerDim { const char* name; int M, K, N; };
static const LayerDim QSAR_LAYERS[] = {
    {"FC1 (2048→512)",   32, 2048, 512},
    {"FC2 (512→256)",    32,  512, 256},
    {"Sol head (256→3)", 32,  256,   3},
    {"Tox head (256→2)", 32,  256,   2},
};
static const int N_LAYERS = 4;

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Kernels
// ─────────────────────────────────────────────────────────────────────────────

static void fill_rand(float* p, int n) {
    for (int i = 0; i < n; i++) p[i] = ((float)(rand()%2000)-1000.f)/5000.f;
}

// ── 3a. Baseline NEON FP32 ────────────────────────────────────────────────────
static void matmul_neon(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float*       __restrict__ C,
                        int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float32x4_t acc = vdupq_n_f32(0.f);
            int k = 0;
            for (; k <= K-4; k += 4) {
                float32x4_t a = vld1q_f32(A + m*K + k);
                float32x4_t b = {B[k*N+n], B[(k+1)*N+n],
                                  B[(k+2)*N+n], B[(k+3)*N+n]};
                acc = vmlaq_f32(acc, a, b);
            }
            float32x2_t s = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            float v = vget_lane_f32(vpadd_f32(s, s), 0);
            for (; k < K; k++) v += A[m*K+k] * B[k*N+n];
            C[m*N+n] = v;
        }
    }
}

// ── 3b. SME2 FP32 outer-product (FMOPA into ZA tile) ─────────────────────────
#if HAS_SME2 && defined(__ARM_FEATURE_SME2)
__attribute__((target("arch=armv9-a+sme2")))
static void matmul_sme2(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float*       __restrict__ C,
                         int M, int K, int N) {
    __arm_start_streaming();
    __arm_za_enable();
    const uint64_t svl = svcntsw();

    for (int m = 0; m < M; m += (int)svl) {
        int tm = std::min((int)svl, M - m);
        for (int n = 0; n < N; n += (int)svl) {
            int tn = std::min((int)svl, N - n);
            svzero_za();
            svbool_t pg_m = svwhilelt_b32((uint32_t)0, (uint32_t)tm);
            svbool_t pg_n = svwhilelt_b32((uint32_t)0, (uint32_t)tn);
            for (int k = 0; k < K; k++) {
                svfloat32_t va = svld1_gather_s32index_f32(
                    pg_m, A + k, svindex_s32(m*K, K));
                svfloat32_t vb = svld1_f32(pg_n, B + k*N + n);
                svmopa_za32_f32_m(0, pg_m, pg_n, va, vb);
            }
            for (int i = 0; i < tm; i++) {
                svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)tn);
                svfloat32_t row = svread_za_f32_m(svundef_f32(), pg, 0, (uint32_t)i);
                svst1_f32(pg, C + (m+i)*N + n, row);
            }
        }
    }
    __arm_za_disable();
    __arm_stop_streaming();
}
#else
static void matmul_sme2(const float* A, const float* B, float* C,
                         int M, int K, int N) {
    LOGW("SME2 not compiled in — falling back to NEON");
    matmul_neon(A, B, C, M, K, N);
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Statistics
// ─────────────────────────────────────────────────────────────────────────────

struct Stats { double avg,p50,p90,p95,min,max,stddev; int n; };

static Stats compute(std::vector<double>& s) {
    std::sort(s.begin(), s.end());
    int n = (int)s.size();
    double sum = std::accumulate(s.begin(),s.end(),0.0);
    double avg = sum/n, var = 0;
    for (double v:s) var += (v-avg)*(v-avg);
    return {avg, s[n/2], s[(int)(n*.9)], s[(int)(n*.95)],
            s.front(), s.back(), std::sqrt(var/n), n};
}

static Stats bench_layer(const LayerDim& l, bool use_sme2, int iters) {
    std::vector<float> A(l.M*l.K), B(l.K*l.N), C(l.M*l.N);
    fill_rand(A.data(), l.M*l.K);
    fill_rand(B.data(), l.K*l.N);
    std::vector<double> samples; samples.reserve(iters);
    for (int i = 0; i < iters; i++) {
        std::fill(C.begin(), C.end(), 0.f);
        auto t0 = Clock::now();
        if (use_sme2 && caps().sme2) matmul_sme2(A.data(),B.data(),C.data(),l.M,l.K,l.N);
        else                          matmul_neon (A.data(),B.data(),C.data(),l.M,l.K,l.N);
        samples.push_back(Ms(Clock::now()-t0).count());
    }
    return compute(samples);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  JNI  —  package com.abhi.pocketqsar, class Sme2Benchmark
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

// ── 5a. Hardware query ────────────────────────────────────────────────────────

JNIEXPORT jboolean JNICALL
Java_com_abhi_pocketqsar_Sme2Benchmark_nativeIsSme2Available(JNIEnv*,jclass) {
    return (jboolean)caps().sme2;
}

JNIEXPORT jint JNICALL
Java_com_abhi_pocketqsar_Sme2Benchmark_nativeGetSvlBits(JNIEnv*,jclass) {
    return (jint)caps().svl_bits;
}

JNIEXPORT jstring JNICALL
Java_com_abhi_pocketqsar_Sme2Benchmark_nativeGetCapsString(JNIEnv* env,jclass) {
    const Caps& c = caps();
    std::ostringstream ss;
    ss << "SME2:" << (c.sme2?"YES":"NO")
       << " SVE2:" << (c.sve2?"YES":"NO")
       << " NEON:YES";
    if (c.sme2) ss << " SVL:" << c.svl_bits << "b";
    return env->NewStringUTF(ss.str().c_str());
}

// ── 5b. Full QSARNet benchmark ────────────────────────────────────────────────
//
// Returns float[] packed as:
//   For each of N_LAYERS layers:
//     [0]  layer_index
//     [1]  neon_avg_ms
//     [2]  neon_p50_ms
//     [3]  neon_p90_ms
//     [4]  neon_stddev_ms
//     [5]  sme2_avg_ms
//     [6]  sme2_p50_ms
//     [7]  sme2_p90_ms
//     [8]  sme2_stddev_ms
//     [9]  speedup
//   Then a summary row (index = N_LAYERS):
//     [0]  N_LAYERS (sentinel)
//     [1]  total_neon_avg
//     [5]  total_sme2_avg
//     [9]  total_speedup

JNIEXPORT jfloatArray JNICALL
Java_com_abhi_pocketqsar_Sme2Benchmark_nativeRunQsarBenchmark(
        JNIEnv* env, jclass, jint iters) {

    caps();  // ensure detection runs first
    const int FIELDS = 10;
    const int ROWS   = N_LAYERS + 1;  // layers + summary
    std::vector<jfloat> buf(ROWS * FIELDS, 0.f);

    double total_neon = 0, total_sme2 = 0;

    for (int i = 0; i < N_LAYERS; i++) {
        Stats neon = bench_layer(QSAR_LAYERS[i], false, iters);
        Stats sme2 = bench_layer(QSAR_LAYERS[i], true,  iters);
        double su  = neon.avg / std::max(sme2.avg, 0.001);
        total_neon += neon.avg;
        total_sme2 += sme2.avg;

        LOGI("[%s] NEON=%.2fms SME2=%.2fms %.2fx",
             QSAR_LAYERS[i].name, neon.avg, sme2.avg, su);

        int off = i * FIELDS;
        buf[off+0] = (jfloat)i;
        buf[off+1] = (jfloat)neon.avg;
        buf[off+2] = (jfloat)neon.p50;
        buf[off+3] = (jfloat)neon.p90;
        buf[off+4] = (jfloat)neon.stddev;
        buf[off+5] = (jfloat)sme2.avg;
        buf[off+6] = (jfloat)sme2.p50;
        buf[off+7] = (jfloat)sme2.p90;
        buf[off+8] = (jfloat)sme2.stddev;
        buf[off+9] = (jfloat)su;
    }

    // Summary row
    int off = N_LAYERS * FIELDS;
    buf[off+0] = (jfloat)N_LAYERS;
    buf[off+1] = (jfloat)total_neon;
    buf[off+5] = (jfloat)total_sme2;
    buf[off+9] = (jfloat)(total_neon / std::max(total_sme2, 0.001));

    jfloatArray arr = env->NewFloatArray((jsize)buf.size());
    env->SetFloatArrayRegion(arr, 0, (jsize)buf.size(), buf.data());
    return arr;
}

// ── 5c. Single inference latency (mirrors QsarExecuTorch.kt usage) ───────────
// Returns float[2]: [neon_ms, sme2_ms]

JNIEXPORT jfloatArray JNICALL
Java_com_abhi_pocketqsar_Sme2Benchmark_nativeInferenceLatency(
        JNIEnv* env, jclass, jint iters) {

    // One forward pass = sum of all layer latencies, batch=1
    LayerDim single_layers[] = {
        {"FC1", 1, 2048, 512},
        {"FC2", 1,  512, 256},
        {"Sol", 1,  256,   3},
        {"Tox", 1,  256,   2},
    };

    double neon_total = 0, sme2_total = 0;
    for (auto& l : single_layers) {
        neon_total += bench_layer(l, false, iters).avg;
        sme2_total += bench_layer(l, true,  iters).avg;
    }

    jfloat buf[2] = {(jfloat)neon_total, (jfloat)sme2_total};
    jfloatArray arr = env->NewFloatArray(2);
    env->SetFloatArrayRegion(arr, 0, 2, buf);
    return arr;
}

} // extern "C"
