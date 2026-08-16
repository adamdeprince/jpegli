// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_DECODE_STAGE_PROFILE_H_
#define JPEGLI_LIB_JPEGLI_DECODE_STAGE_PROFILE_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <jpeglib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JPEGLI_DECODE_STAGE_PROFILE_ABI_VERSION 1u

// Stages follow the decoder's implementation boundaries. Entropy decoding is
// fused with run-length expansion, inverse coefficient ordering, DC prediction,
// and progressive coefficient refinement. Dequantization is fused with the
// inverse transform and its optional smoothing/bias preparation.
typedef enum jpegli_decode_stage_profile_stage {
  JPEGLI_DECODE_STAGE_BUFFER_ALLOCATION = 0,
  JPEGLI_DECODE_STAGE_MARKER_PARSING,
  JPEGLI_DECODE_STAGE_SCAN_PREPARATION,
  JPEGLI_DECODE_STAGE_ENTROPY_AND_COEFFICIENT_RECONSTRUCTION,
  JPEGLI_DECODE_STAGE_OUTPUT_PREPARATION,
  JPEGLI_DECODE_STAGE_DEQUANT_BIAS_SMOOTHING_AND_IDCT,
  JPEGLI_DECODE_STAGE_CHROMA_UPSAMPLING,
  JPEGLI_DECODE_STAGE_COLOR_CONVERSION_AND_LEVEL_SHIFT,
  JPEGLI_DECODE_STAGE_OUTPUT_SAMPLE_CONVERSION_AND_WRITE,
  JPEGLI_DECODE_STAGE_COUNT
} jpegli_decode_stage_profile_stage;

typedef struct jpegli_decode_stage_profile_entry {
  uint64_t elapsed_ticks;
  uint64_t observations;
} jpegli_decode_stage_profile_entry;

typedef struct jpegli_decode_stage_profile {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t timer_overhead_ticks;
  double timer_ticks_per_second;
  uint64_t decoder_elapsed_ns;
  jpegli_decode_stage_profile_entry stages[JPEGLI_DECODE_STAGE_COUNT];
} jpegli_decode_stage_profile;

// Enables profiling for one decompressor. The caller-owned profile must remain
// alive through jpeg_finish_decompress(). Returns nonzero when profiling is
// compiled in and the arguments and platform timer are valid.
int jpegli_enable_decode_stage_profiling(
    j_decompress_ptr cinfo, jpegli_decode_stage_profile* profile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // JPEGLI_LIB_JPEGLI_DECODE_STAGE_PROFILE_H_
