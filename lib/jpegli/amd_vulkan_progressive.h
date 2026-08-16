// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_AMD_VULKAN_PROGRESSIVE_H_
#define JPEGLI_LIB_JPEGLI_AMD_VULKAN_PROGRESSIVE_H_

#include <cstddef>
#include <cstdint>

#include "lib/jpegli/common.h"

namespace jpegli {

// A fixed-stride block descriptor produced by the AMD wave64 kernel. Word 0
// is the number of local tokens, word 1 contains block metadata, and the
// remaining words contain packed Token values in coefficient order.
struct AmdVulkanACResult {
  const uint32_t* words = nullptr;
  size_t stride_words = 0;
  size_t num_blocks = 0;
  // Optional latency-oriented compact streams. Initial AC tokens use the
  // packed Token ABI directly. Refinement tokens, correction bits, and EOB-run
  // values use one uint32_t per output element so independent block scatters
  // never contend for packed words. The CPU consumer only performs a linear
  // type conversion; all JPEG ordering and EOB-run state is resolved on GPU.
  const uint32_t* compact_header = nullptr;
  const uint32_t* compact_tokens = nullptr;
  const uint32_t* compact_refbits = nullptr;
  const uint32_t* compact_eobruns = nullptr;
  const uint32_t* symbol_histogram = nullptr;
};

#if defined(JPEGLI_ENABLE_AMD_VULKAN)

// Reserves a UMA-backed input slot for the experimental latency front end and
// exposes its component planes through jpeg_comp_master. Input conversion,
// color transform, downsampling, and adaptive-quantization analysis remain on
// the CPU; DCT and quantization are fused with progressive token generation on
// the GPU. Enabled with JPEGLI_AMD_VULKAN_FRONTEND=1.
bool AmdVulkanFrontendPrepare(j_compress_ptr cinfo);

// Copies the completed quantized coefficient planes into one of two retained
// UMA slots and submits all supported progressive AC scans without waiting.
// The compressor must be finished or aborted on the same thread.
bool AmdVulkanProgressiveSubmit(j_compress_ptr cinfo);

// Uploads the coefficient planes once for all progressive scans in an image.
// If the image was submitted earlier this waits for its slot; otherwise this
// performs a synchronous submit-and-wait for compatibility with the existing
// encoder entry point.
bool AmdVulkanProgressiveBegin(j_compress_ptr cinfo);
void AmdVulkanProgressiveEnd(j_compress_ptr cinfo);

// Tokenizes an initial progressive AC scan (Ss > 0, Ah == 0). The GPU emits
// exact per-block token sequences; the caller retains the serial JPEG EOB-run
// state and stitches those sequences together.
bool AmdVulkanTokenizeInitialAC(j_compress_ptr cinfo, int scan_index,
                                int context, AmdVulkanACResult* result);

// Compacts a refinement scan into ordered significant-coefficient events.
// Each event records its coefficient position, new/existing state, sign and
// correction bit. The CPU consumes only these events instead of rescanning 63
// coefficients per block.
bool AmdVulkanTokenizeRefinementAC(j_compress_ptr cinfo, int scan_index,
                                   AmdVulkanACResult* result);

// Returns packed progressive-DC tokens and a 256-bin symbol histogram for a
// supported non-interleaved initial DC scan. Both are generated in the same
// component dispatch as the AC descriptors.
bool AmdVulkanTokenizeDC(j_compress_ptr cinfo, int scan_index,
                         AmdVulkanACResult* result);

#else

inline bool AmdVulkanFrontendPrepare(j_compress_ptr) { return false; }
inline bool AmdVulkanProgressiveSubmit(j_compress_ptr) { return false; }
inline bool AmdVulkanProgressiveBegin(j_compress_ptr) { return false; }
inline void AmdVulkanProgressiveEnd(j_compress_ptr) {}
inline bool AmdVulkanTokenizeInitialAC(j_compress_ptr, int, int,
                                       AmdVulkanACResult*) {
  return false;
}
inline bool AmdVulkanTokenizeRefinementAC(j_compress_ptr, int,
                                          AmdVulkanACResult*) {
  return false;
}
inline bool AmdVulkanTokenizeDC(j_compress_ptr, int, AmdVulkanACResult*) {
  return false;
}

#endif

}  // namespace jpegli

#endif  // JPEGLI_LIB_JPEGLI_AMD_VULKAN_PROGRESSIVE_H_
