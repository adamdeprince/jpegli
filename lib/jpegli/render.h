// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_RENDER_H_
#define JPEGLI_LIB_JPEGLI_RENDER_H_

#include <cstddef>
#include <cstdint>

#include "lib/jpegli/common.h"

namespace jpegli {

void GatherBlockStats(const int16_t* coeffs, size_t coeffs_size,
                      int32_t* nonzeros, int32_t* sumabs);

void ComputeOptimalLaplacianBiases(int num_blocks, const int* nonzeros,
                                   const int* sumabs, float* biases);

void PrepareForOutput(j_decompress_ptr cinfo);

void ProcessOutput(j_decompress_ptr cinfo, size_t* num_output_rows,
                   JSAMPARRAY scanlines, size_t max_output_rows);

void ProcessRawOutput(j_decompress_ptr cinfo, JSAMPIMAGE data);

}  // namespace jpegli

#endif  // JPEGLI_LIB_JPEGLI_RENDER_H_
