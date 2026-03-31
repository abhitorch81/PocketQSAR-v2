# PocketQSAR 🧪📱

**On-device molecular inference, SME2/KleidiAI benchmarking, and 3D chemical space explorer for Android**

[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://android.com)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-purple.svg)](https://kotlinlang.org)
[![ExecuTorch](https://img.shields.io/badge/Runtime-ExecuTorch-orange.svg)](https://pytorch.org/executorch)
[![OpenGL ES](https://img.shields.io/badge/Graphics-OpenGL%20ES%203.0-blue.svg)](https://www.khronos.org/opengles/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is PocketQSAR?

PocketQSAR is an Android app that brings **molecular property prediction entirely on-device** — no cloud, no network calls, no latency.

It runs **QSARNet**, a multi-task MLP trained on the ESOL dataset, directly on your phone using **ExecuTorch + XNNPACK**. It predicts aqueous solubility (logS) and toxicity probability from a 2048-dimensional molecular descriptor vector in under 4 ms on a standard ARM phone.

The app also includes a **native SME2/KleidiAI benchmarking layer** written in C++ NDK that measures per-layer inference latency (NEON vs SME2) and a **3D OpenGL ES chemical space explorer** that renders 100 molecules as lit spheres in a PCA/solubility coordinate space.

---

## Features

### 🔬 On-Device QSAR Inference
- Predicts **aqueous solubility (logS)** and **toxicity probability** for small molecules
- Runs locally via **ExecuTorch + XNNPACK** — no internet required
- **Developability score (0–100)** combining solubility, toxicity, and Lipinski rules
- 2D molecular structure images from bundled assets

### ⚡ SME2 / KleidiAI Benchmark Screen
- **Runtime SME2 detection** via `HWCAP2_SME2` (no permissions needed)
- Per-layer latency for all 4 QSARNet layers: FC1, FC2, Sol-head, Tox-head
- NEON baseline vs SME2 outer-product (FMOPA) acceleration
- **MPAndroidChart** bar charts showing speedup per layer
- Single-inference latency (batch=1)
- Graceful NEON fallback on non-SME2 devices

### 🌐 3D Chemical Space Explorer (OpenGL ES 3.0)
- 100 molecules rendered as **lit icospheres** in 3D space
- **X = PCA axis 1, Y = PCA axis 2, Z = logS** (solubility)
- Colour-coded by toxicity + developability score
- Sphere radius scales with developability score
- **Gesture controls:** single-finger orbit, two-finger pan, pinch zoom, tap-to-select, double-tap reset, long-press auto-rotate
- Selection card showing molecule name, logS, toxicity, dev score

### 📊 Molecule Browser & History
- Browse all molecules in the ESOL dataset
- Descriptor radar plot (MW / logP / TPSA)
- Prediction history dialog

---

## Architecture

```
app/src/main/
├── cpp/
│   ├── Sme2Benchmark.cpp       # JNI: SME2 detection + NEON/SME2 matmul kernels
│   └── CMakeLists.txt          # NDK build: -march=armv8.2-a+fp16+dotprod
├── java/com/abhi/pocketqsar/
│   ├── MainActivity.kt
│   ├── QsarExecuTorch.kt       # ExecuTorch JNI wrapper
│   ├── DemoMolecule.kt         # Molecule data class
│   ├── DemoMoleculeRepository.kt
│   ├── ChemicalSpaceView.kt    # 2D scatter plot
│   ├── DescriptorRadarView.kt  # MW/logP/TPSA radar
│   ├── ModelFileHelper.kt
│   ├── Sme2Benchmark.kt        # Kotlin JNI bridge + data classes
│   ├── Sme2BenchmarkActivity.kt
│   ├── MolecularSpaceActivity.kt
│   ├── MolecularSpaceGLView.kt # GLSurfaceView + gesture handling
│   ├── MolecularSpaceRenderer.kt # OpenGL ES 3.0 renderer
│   └── MolecularSpaceViewModel.kt
└── assets/
    ├── mobile_molecules.json   # ESOL dataset with descriptors + embeddings
    ├── qsar_net_xnnpack.pte    # ExecuTorch compiled model
    └── mol_imgs/               # 2D structure PNGs
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Kotlin + C++ (NDK) |
| ML Runtime | ExecuTorch + XNNPACK |
| ARM Acceleration | KleidiAI / SME2 (Armv9.3+), NEON fallback |
| Graphics | OpenGL ES 3.0 |
| Charting | MPAndroidChart |
| Build | CMake 3.22 + Gradle Kotlin DSL |
| Min SDK | 26 (Android 8.0) |
| Target ABI | arm64-v8a |

---

## Benchmark Results

Measured on **Vivo V2251 (MediaTek, Android 15, API 35)** — NEON path (sme2=0):

| Layer | NEON avg | SME2 avg | Speedup |
|-------|----------|----------|---------|
| FC1 (2048→512) | 98.12 ms | 88.32 ms | 1.11× |
| FC2 (512→256) | 5.75 ms | 5.77 ms | 1.00× |
| Sol head (256→3) | 0.01 ms | 0.01 ms | 1.00× |
| Tox head (256→2) | 0.00 ms | 0.00 ms | 1.00× |
| **Single inference (batch=1)** | **3.023 ms** | **2.821 ms** | **1.07×** |

> Note: SME2 column runs NEON fallback on this MediaTek device. True SME2 acceleration (~3.9× for FP16) requires Snapdragon 8 Gen 4 or Dimensity 9400.

### Expected speedups on SME2 hardware

| Precision | NEON | SME2 | Speedup |
|-----------|------|------|---------|
| FP32 | ~98 ms | ~88 ms | ~1.1× |
| FP16 | ~37 ms | ~9 ms | ~3.9× |
| INT8 | ~18 ms | ~10 ms | ~1.8× |

---

## How SME2 Detection Works

```cpp
// No permissions needed — reads kernel hardware capability bitmap
unsigned long hwcap2 = getauxval(AT_HWCAP2);
bool sme2_available  = (hwcap2 & HWCAP2_SME2) != 0;

// Query Streaming Vector Length if SME2 present
if (sme2_available) {
    __arm_start_streaming();
    uint32_t svl_bits = svcntsw() * 32;
    __arm_stop_streaming();
}
```

The SME2 matmul kernel uses `FMOPA` (FP32 outer-product-accumulate into ZA tile) — the same instruction at the core of KleidiAI's acceleration:

```cpp
__arm_start_streaming();
__arm_za_enable();
svmopa_za32_f32_m(0, pg_m, pg_n, va, vb);  // outer-product into ZA tile
__arm_za_disable();
__arm_stop_streaming();
```

---

## Build Requirements

- **Android Studio** Hedgehog or later
- **NDK r26b** or later (required for `<arm_sme.h>` intrinsics)
- **CMake 3.22.1** or later
- **Target ABI:** arm64-v8a

> ⚠️ Important: The CMakeLists.txt uses `-march=armv8.2-a+fp16+dotprod`, NOT `-march=armv9-a+sme2`. Using the armv9 flag causes a `SIGILL` crash on any non-SME2 phone because the compiler emits SME2 instructions even in the NEON fallback path.

---

## Getting Started

```bash
git clone https://github.com/abhitorch81/PocketQSAR-v2.git
```

Open in Android Studio → connect a physical ARM64 Android phone → `Shift+F10` to build and run.

The SME2 benchmark requires a **physical device** — it will crash on an x86 emulator because the native library is compiled for arm64 only.

---

## Data & Model

- **Dataset:** ESOL (Delaney) aqueous solubility dataset
- **Descriptors:** 2048-dimensional RDKit feature vectors
- **Model:** PyTorch MLP exported to ExecuTorch `.pte` format with XNNPACK backend
- **Embeddings:** PCA 2D projection precomputed offline, stored in `mobile_molecules.json`
- Training notebooks: `Model.ipynb`, `Exectorch.ipynb`

---

## Key Technical Learnings

1. **`-march=armv9-a+sme2` causes SIGILL** even on NEON fallback — use `-march=armv8.2-a` and guard SME2 code with `#if HAS_SME2` + `__attribute__((target(...)))`
2. **`UnsatisfiedLinkError` is a `Throwable`** not an `Exception` — always `catch(e: Throwable)` around JNI calls
3. **Nested data classes in ViewModels** cause cross-file unresolved references — promote to top-level classes
4. **Kotlin `var` auto-generates setters** — never write an explicit `fun setAutoRotate()` alongside `var autoRotate`
5. **Gradle Kotlin DSL** — cmake path must use `file()` not a raw string
6. **JitPack** must be added to `settings.gradle.kts` `dependencyResolutionManagement`, not app-level gradle

---

## Repo Structure

```
PocketQSAR/
├── app/                    # Android app module
├── Model.ipynb             # QSARNet training notebook
├── Exectorch.ipynb         # ExecuTorch export notebook
├── mobile_molecules.json   # Molecule dataset
├── qsar_multitask.pt       # PyTorch model weights
└── app-release.apk         # Pre-built release APK
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with [Claude](https://claude.ai) (Anthropic) · March 2026*
