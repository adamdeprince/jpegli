// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_APPLE_METAL_INTERNAL_H_
#define JPEGLI_LIB_JPEGLI_APPLE_METAL_INTERNAL_H_

#include <cstddef>

#include "lib/jpegli/apple_metal.h"

namespace jpegli {

// Returns true when Metal should own final reconstruction for this cinfo. The
// final eligibility check is repeated by AppleMetalReconstruct after the input
// scans have completed.
bool AppleMetalShouldAttempt(j_decompress_ptr cinfo, bool direct_output);

// Lazily allocates the shared Metal coefficient backing before entropy decode.
// On success AppleMetalCoefficientPlane returns the component bases that are
// installed into the ordinary libjpeg virtual-array interface.
bool AppleMetalPrepareCoefficientStorage(j_decompress_ptr cinfo);
JBLOCK* AppleMetalCoefficientPlane(j_decompress_ptr cinfo, int component);

// Reconstructs the decoded coefficient planes. PrepareForOutput must already
// have been called and all scans must have been consumed.
bool AppleMetalReconstruct(j_decompress_ptr cinfo, bool direct_output);

// Copies completed shared-buffer rows to the libjpeg scanline destination.
JDIMENSION AppleMetalReadScanlines(j_decompress_ptr cinfo, JSAMPARRAY scanlines,
                                   JDIMENSION max_lines);

bool AppleMetalExportOutput(j_decompress_ptr cinfo,
                            JpegliAppleMetalOutput* output);

// Used by the direct endpoint when reconstruction is unsupported: the normal
// CPU renderer remains authoritative, then its completed RGBA pixels are
// uploaded into one shared Metal allocation.
bool AppleMetalUploadCpuOutput(j_decompress_ptr cinfo, const uint8_t* pixels,
                               size_t row_bytes,
                               JpegliAppleMetalOutput* output);

void AppleMetalResetDecoder(j_decompress_ptr cinfo);

// Implemented in decode.cc so both the enabled backend and CPU-only stubs share
// identical per-cinfo policy behavior.
void SetAppleMetalFallbackReason(j_decompress_ptr cinfo, const char* reason);

}  // namespace jpegli

#endif  // JPEGLI_LIB_JPEGLI_APPLE_METAL_INTERNAL_H_
