// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <atomic>
#include <cstring>

#include "lib/jpegli/apple_metal_internal.h"
#include "lib/jpegli/decode_internal.h"

#if !defined(JPEGLI_ENABLE_APPLE_METAL)

namespace {
std::atomic<size_t> g_stub_crossover_pixels{786432};
}

namespace jpegli {

bool AppleMetalShouldAttempt(j_decompress_ptr cinfo, bool direct_output) {
  SetAppleMetalFallbackReason(cinfo,
                              "Metal support is not enabled in this build");
  return false;
}

bool AppleMetalReconstruct(j_decompress_ptr cinfo, bool direct_output) {
  SetAppleMetalFallbackReason(cinfo,
                              "Metal support is not enabled in this build");
  return false;
}

JDIMENSION AppleMetalReadScanlines(j_decompress_ptr cinfo, JSAMPARRAY scanlines,
                                   JDIMENSION max_lines) {
  return 0;
}

bool AppleMetalExportOutput(j_decompress_ptr cinfo,
                            JpegliAppleMetalOutput* output) {
  return false;
}

bool AppleMetalUploadCpuOutput(j_decompress_ptr cinfo, const uint8_t* pixels,
                               size_t row_bytes,
                               JpegliAppleMetalOutput* output) {
  return false;
}

void AppleMetalResetDecoder(j_decompress_ptr cinfo) {}

}  // namespace jpegli

void jpegli_apple_metal_release_output(JpegliAppleMetalOutput* output) {
  if (output != nullptr) *output = {};
}

void jpegli_apple_metal_release_cached_resources(
    JpegliAppleMetalReleaseMode mode) {}

void jpegli_apple_metal_set_crossover_pixels(size_t pixels) {
  g_stub_crossover_pixels.store(pixels, std::memory_order_relaxed);
}

size_t jpegli_apple_metal_get_crossover_pixels(void) {
  return g_stub_crossover_pixels.load(std::memory_order_relaxed);
}

int jpegli_apple_metal_is_available(void) { return 0; }

#endif  // !JPEGLI_ENABLE_APPLE_METAL
