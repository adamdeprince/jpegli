// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_APPLE_METAL_H_
#define JPEGLI_LIB_JPEGLI_APPLE_METAL_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "jpeglib.h"

#ifdef __cplusplus
extern "C" {
#endif

// This API is experimental. It is available in CPU-only builds as a set of
// harmless capability/fallback entry points, so applications do not need
// conditional declarations. A successful direct decode requires an Apple
// Metal-enabled build and a Metal device.

typedef enum {
  JPEGLI_APPLE_METAL_AUTO = 0,
  JPEGLI_APPLE_METAL_DISABLED = 1,
  JPEGLI_APPLE_METAL_FORCE = 2,
} JpegliAppleMetalMode;

typedef enum {
  JPEGLI_APPLE_METAL_PIXEL_FORMAT_RGBA8_UNORM = 1,
} JpegliAppleMetalPixelFormat;

typedef enum {
  JPEGLI_APPLE_METAL_RELEASE_SCRATCH = 0,
  JPEGLI_APPLE_METAL_RELEASE_ALL = 1,
} JpegliAppleMetalReleaseMode;

typedef struct {
  // Objective-C id<MTLTexture> and id<MTLBuffer> handles. They remain valid
  // until jpegli_apple_metal_release_output() is called. Under ARC, bridge
  // these as non-transferring references.
  void* texture;
  void* buffer;
  // CPU address of the shared buffer. Access after the decode has completed is
  // coherent on Apple unified-memory GPUs; no explicit readback is required.
  const void* buffer_contents;
  size_t width;
  size_t height;
  size_t row_bytes;
  JpegliAppleMetalPixelFormat pixel_format;
  void* private_handle;
} JpegliAppleMetalOutput;

typedef struct {
  uint64_t metal_initialization_ns;
  uint64_t coefficient_analysis_ns;
  uint64_t coefficient_copy_ns;
  uint64_t command_encoding_ns;
  uint64_t submission_overhead_ns;
  uint64_t gpu_dequant_idct_ns;
  uint64_t gpu_upsample_color_ns;
  uint64_t gpu_fused_reconstruction_ns;
  uint64_t cpu_output_copy_ns;
  uint64_t reconstruction_total_ns;
  uint64_t cpu_decoder_bytes;
  uint64_t metal_buffer_bytes;
  uint64_t decoder_retained_bytes;
  uint64_t unified_coefficient_bytes;
  uint64_t float_plane_bytes;
  int used_metal;
  int direct_output;
  int fused_pipeline;
  int caller_command_buffer;
  char fallback_reason[128];
} JpegliAppleMetalStats;

// Selects Metal policy for this decompressor. The default is AUTO. This must
// be called after jpegli_create_decompress() and before
// jpegli_start_decompress(). FORCE bypasses the image-size crossover but still
// falls back safely for unsupported JPEG/output modes.
void jpegli_apple_metal_set_mode(j_decompress_ptr cinfo,
                                 JpegliAppleMetalMode mode);

JpegliAppleMetalMode jpegli_apple_metal_get_mode(j_decompress_ptr cinfo);

// Returns nonzero only when this decoder actually reconstructed with Metal.
int jpegli_apple_metal_was_used(j_decompress_ptr cinfo);

// Gets measurements for the current image. Valid until the decoder is aborted,
// reused, or destroyed.
void jpegli_apple_metal_get_stats(j_decompress_ptr cinfo,
                                  JpegliAppleMetalStats* stats);

// Controls the process-wide AUTO crossover. FORCE and the direct-output API
// ignore this threshold. The measured M4 default is 480000 output pixels.
void jpegli_apple_metal_set_crossover_pixels(size_t pixels);
size_t jpegli_apple_metal_get_crossover_pixels(void);

// Starts decompression and returns the completed image in a shared Metal buffer
// and RGBA8Unorm texture without CPU pixel readback. Set out_color_space to
// JCS_EXT_RGBA before calling. On success the caller may immediately call
// jpegli_finish_decompress(); release of the returned output is independent of
// decoder lifetime.
boolean jpegli_start_decompress_to_apple_metal(j_decompress_ptr cinfo,
                                               JpegliAppleMetalOutput* output);

// Encodes reconstruction into a caller-owned id<MTLCommandBuffer> and writes
// directly to a caller-owned RGBA8Unorm id<MTLTexture>. The command buffer is
// not committed or waited by JPEGli. Both objects must use the same Metal
// device as JPEGli; the texture must exactly match output dimensions and have
// MTLTextureUsageShaderWrite. The returned output retains all source resources
// needed by the encoded work and must be released after the caller has
// completed (or discarded) the command buffer. buffer and buffer_contents are
// null because no CPU-readable destination is allocated.
boolean jpegli_start_decompress_to_apple_metal_command_buffer(
    j_decompress_ptr cinfo, void* command_buffer, void* destination_texture,
    JpegliAppleMetalOutput* output);

void jpegli_apple_metal_release_output(JpegliAppleMetalOutput* output);

// Releases the single cached scratch set, or all inexpensive global Metal
// state as well. Objects still referenced by active decoders/outputs remain
// alive until those owners release them.
void jpegli_apple_metal_release_cached_resources(
    JpegliAppleMetalReleaseMode mode);

// Returns nonzero if this library was built with Metal support and a Metal
// device can be created.
int jpegli_apple_metal_is_available(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // JPEGLI_LIB_JPEGLI_APPLE_METAL_H_
