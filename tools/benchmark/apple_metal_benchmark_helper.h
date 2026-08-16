// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_TOOLS_BENCHMARK_APPLE_METAL_BENCHMARK_HELPER_H_
#define JPEGLI_TOOLS_BENCHMARK_APPLE_METAL_BENCHMARK_HELPER_H_

#include <stddef.h>
#include <stdint.h>

struct JpegliAppleMetalBenchmarkTarget;

JpegliAppleMetalBenchmarkTarget* JpegliAppleMetalBenchmarkTargetCreate(
    size_t width, size_t height);
void* JpegliAppleMetalBenchmarkCommandBuffer(
    JpegliAppleMetalBenchmarkTarget* target);
void* JpegliAppleMetalBenchmarkTexture(JpegliAppleMetalBenchmarkTarget* target);
bool JpegliAppleMetalBenchmarkCommitWaitAndRead(
    JpegliAppleMetalBenchmarkTarget* target, uint8_t* rgba,
    size_t rgba_row_bytes);
void JpegliAppleMetalBenchmarkTargetDestroy(
    JpegliAppleMetalBenchmarkTarget* target);

#endif  // JPEGLI_TOOLS_BENCHMARK_APPLE_METAL_BENCHMARK_HELPER_H_
