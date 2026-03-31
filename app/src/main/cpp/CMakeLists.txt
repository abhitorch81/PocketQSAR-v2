cmake_minimum_required(VERSION 3.22)
project(pocketqsar_sme2)

if(NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Requires arm64-v8a. Got ${CMAKE_ANDROID_ARCH_ABI}.")
endif()

add_library(sme2bench SHARED Sme2Benchmark.cpp)

target_compile_options(sme2bench PRIVATE
    # Use armv8.2-a+neon — runs on ALL modern Android phones (2018+)
    # SME2 instructions are guarded at RUNTIME inside the C++ code via HWCAP2
    # so the binary works everywhere but accelerates on SME2 hardware
    -march=armv8.2-a+fp16+dotprod
    -O3
    -fno-omit-frame-pointer
    -fstrict-aliasing
    -std=c++17
)

target_compile_definitions(sme2bench PRIVATE _GNU_SOURCE)

find_library(log-lib log)
target_link_libraries(sme2bench ${log-lib})
