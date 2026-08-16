// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_DECODE_PROFILE_H_
#define JPEGLI_LIB_JPEGLI_DECODE_PROFILE_H_

#include <cstddef>
#include <cstdint>

#include "jpeglib.h"

namespace jpegli {

// Internal profiling stages for jpegli's decompressor. Profiling is disabled
// by default and is intended for benchmarks, not production telemetry.
enum class DecodeProfileStage : size_t {
  kSetup,
  kMarkers,
  kEntropy,
  kCoefficientAnalysis,
  kIdct,
  kUpsampling,
  kColorConversion,
  kPixelOutput,
  kCount,
};

constexpr size_t kNumDecodeProfileStages =
    static_cast<size_t>(DecodeProfileStage::kCount);

struct DecodeProfile {
  uint64_t nanoseconds[kNumDecodeProfileStages] = {};
  uint64_t calls[kNumDecodeProfileStages] = {};
};

void SetDecodeProfileEnabled(j_decompress_ptr cinfo, bool enabled);
void ResetDecodeProfile(j_decompress_ptr cinfo);
DecodeProfile GetDecodeProfile(j_decompress_ptr cinfo);

}  // namespace jpegli

#endif  // JPEGLI_LIB_JPEGLI_DECODE_PROFILE_H_
