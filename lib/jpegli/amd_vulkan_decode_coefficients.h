// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_AMD_VULKAN_DECODE_COEFFICIENTS_H_
#define JPEGLI_LIB_JPEGLI_AMD_VULKAN_DECODE_COEFFICIENTS_H_

#include <cstdint>

#include "lib/jpegli/common.h"

namespace jpegli {

// Progressive entropy parsing emits one additive event for every initial
// coefficient and every asserted refinement bit. Initial coefficient values
// carry their sign, so all later deltas commute and may be atomically combined
// by a single GPU dispatch after the final scan.
struct AmdVulkanDecodeCoefficientEvent {
  uint32_t coefficient_index;
  int32_t delta;
};
static_assert(sizeof(AmdVulkanDecodeCoefficientEvent) == 8,
              "GPU decode coefficient event ABI changed");

// Selects the experimental event path when
// JPEGLI_AMD_VULKAN_DECODE_COEFFICIENTS is set to "1", "force", or "cpu".
// "cpu" retains event formation but reconstructs on the CPU as an
// architecture-overhead control. Unsupported inputs stay on the stock path.
bool AmdVulkanDecodeCoefficientsPrepare(j_decompress_ptr cinfo);

// Reconstructs the coefficient planes once all progressive scans are parsed.
// GPU failures use the event stream for an exact CPU fallback.
bool AmdVulkanDecodeCoefficientsFinish(j_decompress_ptr cinfo);

}  // namespace jpegli

#endif  // JPEGLI_LIB_JPEGLI_AMD_VULKAN_DECODE_COEFFICIENTS_H_
