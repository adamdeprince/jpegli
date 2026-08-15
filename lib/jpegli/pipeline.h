// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_PIPELINE_H_
#define JPEGLI_LIB_JPEGLI_PIPELINE_H_

#include <jpeglib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Submits the progressive AC ordering and symbol-formation work without
// waiting for it. Call this after writing all input rows and before
// jpeg_finish_compress(). The compressor must be finished or aborted on the
// same thread. Two compressor objects may have work in flight at once.
//
// Returns TRUE when work was submitted (or was already pending). FALSE means
// that this image is not eligible or the AMD Vulkan path is unavailable;
// jpeg_finish_compress() remains valid and completes through the normal path.
boolean jpegli_pipeline_submit(j_compress_ptr cinfo);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // JPEGLI_LIB_JPEGLI_PIPELINE_H_
