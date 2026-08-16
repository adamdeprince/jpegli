// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_AMD_VULKAN_DECODE_ENTROPY_H_
#define JPEGLI_LIB_JPEGLI_AMD_VULKAN_DECODE_ENTROPY_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jpegli/common.h"

namespace jpegli {

// One independently decodable entropy range. Restart markers reset the DC
// predictor, EOB run and bit alignment, so a range always starts at block 0 of
// a scan or immediately after a restart marker.
struct AmdVulkanDecodeEntropySegment {
  uint32_t data_offset;
  uint32_t data_size;
  uint32_t block_start;
  uint32_t block_count;
};

// Snapshot of one non-interleaved progressive scan. Huffman lookup tables are
// packed as bits | (value << 16), matching Jpegli's two-level decoder table.
struct AmdVulkanDecodeEntropyScan {
  uint32_t component;
  uint32_t ss;
  uint32_t se;
  uint32_t ah;
  uint32_t al;
  uint32_t restart_interval;
  uint32_t total_blocks;
  std::vector<uint32_t> dc_lut;
  std::vector<uint32_t> ac_lut;
  std::vector<AmdVulkanDecodeEntropySegment> segments;
};

// Activates the experimental full-scan capture path when
// JPEGLI_AMD_VULKAN_DECODE_ENTROPY requests independent-scan or restart-range
// parsing. The ordinary decoder remains byte-for-byte isolated when disabled.
bool AmdVulkanDecodeEntropyPrepare(j_decompress_ptr cinfo);
bool AmdVulkanDecodeEntropyActive(const jpeg_decomp_master* m);

// Captures and indexes the current entropy scan. Returns one of the decoder's
// internal status values (JPEG_SCAN_COMPLETED or kNeedMoreInput).
int AmdVulkanDecodeEntropyCaptureScan(j_decompress_ptr cinfo,
                                      const uint8_t* data, size_t len,
                                      size_t* pos, size_t* bit_pos);

// Builds independent component/block-range chains and reconstructs all
// coefficients. GPU failure uses the same captured representation on the CPU.
bool AmdVulkanDecodeEntropyFinish(j_decompress_ptr cinfo);

}  // namespace jpegli

#endif  // JPEGLI_LIB_JPEGLI_AMD_VULKAN_DECODE_ENTROPY_H_
