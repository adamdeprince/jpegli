// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_ENCODE_STREAMING_H_
#define JPEGLI_LIB_JPEGLI_ENCODE_STREAMING_H_

#include "lib/jpegli/common.h"

namespace jpegli {

void ComputeCoefficientsForiMCURow(j_compress_ptr cinfo);

// Materializes coefficient arrays from the retained AMD front-end planes if a
// Vulkan submit fails before tokenization. This is a cold failure path.
void ComputeAmdVulkanFrontendCoefficients(j_compress_ptr cinfo);

void ComputeTokensForiMCURow(j_compress_ptr cinfo);

void WriteiMCURow(j_compress_ptr cinfo);

}  // namespace jpegli

#endif  // JPEGLI_LIB_JPEGLI_ENCODE_STREAMING_H_
