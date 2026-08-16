// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "lib/jpegli/apple_metal_internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>

#include "lib/jpegli/common_internal.h"
#include "lib/jpegli/decode_internal.h"
#include "lib/jpegli/memory_manager.h"
#include "lib/jpegli/render.h"

#if defined(JPEGLI_ENABLE_APPLE_METAL)

namespace jpegli {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t NowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
          .count());
}

size_t AlignUp(size_t value, size_t alignment) {
  if (value > std::numeric_limits<size_t>::max() - alignment + 1) return 0;
  return (value + alignment - 1) / alignment * alignment;
}

bool SafeAdd(size_t a, size_t b, size_t* result) {
  if (a > std::numeric_limits<size_t>::max() - b) return false;
  *result = a + b;
  return true;
}

bool SafeMul(size_t a, size_t b, size_t* result) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
  *result = a * b;
  return true;
}

constexpr size_t kDefaultCrossoverPixels = 786432;
constexpr size_t kMaxMetalWorkingSet = 512u << 20;
constexpr size_t kMaxCachedScratch = 96u << 20;
constexpr size_t kMetalAlignment = 256;

std::atomic<size_t> g_crossover_pixels{kDefaultCrossoverPixels};

struct ComponentParams {
  uint32_t coeff_offset;
  uint32_t plane_offset;
  uint32_t blocks_w;
  uint32_t blocks_h;
  uint32_t h_factor;
  uint32_t v_factor;
  uint32_t v_samp_factor;
  uint32_t reserved;
};

struct DecodeParams {
  ComponentParams comp[3];
  uint32_t width;
  uint32_t height;
  uint32_t output_row_pixels;
  uint32_t num_components;
  uint32_t jpeg_color_space;
  uint32_t fancy_upsampling;
  uint32_t total_imcu_rows;
  uint32_t reserved;
};

static_assert(sizeof(ComponentParams) == 32, "Metal component ABI mismatch");
static_assert(sizeof(DecodeParams) == 128, "Metal decode ABI mismatch");
static_assert(sizeof(JCOEF) == sizeof(int16_t), "Metal coefficients require 16-bit JCOEF");

#if defined(JPEGLI_APPLE_METAL_PRECOMPILED_LIBRARY)
#include "jpegli_apple_metal_metallib.inc"
#endif

const char kMetalSource[] = R"metal(
#include <metal_stdlib>
using namespace metal;

struct ComponentParams {
  uint coeff_offset;
  uint plane_offset;
  uint blocks_w;
  uint blocks_h;
  uint h_factor;
  uint v_factor;
  uint v_samp_factor;
  uint reserved;
};

struct DecodeParams {
  ComponentParams comp[3];
  uint width;
  uint height;
  uint output_row_pixels;
  uint num_components;
  uint jpeg_color_space;
  uint fancy_upsampling;
  uint total_imcu_rows;
  uint reserved;
};

inline void idct2(thread const float* input, thread float* output) {
  const float a = input[0];
  const float b = input[1];
  output[0] = a + b;
  output[1] = a - b;
}

inline void idct4(thread const float* input, thread float* output) {
  float even_in[2] = {input[0], input[2]};
  float even_out[2];
  idct2(even_in, even_out);
  float odd_in[2];
  odd_in[1] = input[3] + input[1];
  odd_in[0] = input[1] * as_type<float>(0x3fb504f3u);
  float odd_out[2];
  idct2(odd_in, odd_out);
  const float wc0 = as_type<float>(0x3f0a8bd4u);
  const float wc1 = as_type<float>(0x3fa73d75u);
  output[0] = fma(wc0, odd_out[0], even_out[0]);
  output[3] = fma(-wc0, odd_out[0], even_out[0]);
  output[1] = fma(wc1, odd_out[1], even_out[1]);
  output[2] = fma(-wc1, odd_out[1], even_out[1]);
}

inline void idct8(thread const float* input, thread float* output) {
  float even_in[4] = {input[0], input[2], input[4], input[6]};
  float even_out[4];
  idct4(even_in, even_out);
  float odd_in[4] = {input[1], input[3], input[5], input[7]};
  odd_in[3] = odd_in[3] + odd_in[2];
  odd_in[2] = odd_in[2] + odd_in[1];
  odd_in[1] = odd_in[1] + odd_in[0];
  odd_in[0] = odd_in[0] * as_type<float>(0x3fb504f3u);
  float odd_out[4];
  idct4(odd_in, odd_out);
  const float wc0 = as_type<float>(0x3f0281f7u);
  const float wc1 = as_type<float>(0x3f19f1bdu);
  const float wc2 = as_type<float>(0x3f6664d7u);
  const float wc3 = as_type<float>(0x402406cfu);
  output[0] = fma(wc0, odd_out[0], even_out[0]);
  output[7] = fma(-wc0, odd_out[0], even_out[0]);
  output[1] = fma(wc1, odd_out[1], even_out[1]);
  output[6] = fma(-wc1, odd_out[1], even_out[1]);
  output[2] = fma(wc2, odd_out[2], even_out[2]);
  output[5] = fma(-wc2, odd_out[2], even_out[2]);
  output[3] = fma(wc3, odd_out[3], even_out[3]);
  output[4] = fma(-wc3, odd_out[3], even_out[3]);
}

kernel void jpegli_idct(
    device const short* coefficients [[buffer(0)]],
    device const float* dequant [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device float* planes [[buffer(3)]],
    constant DecodeParams& params [[buffer(4)]],
    constant uint& component [[buffer(5)]],
    uint2 block_pos [[thread_position_in_grid]]) {
  const ComponentParams cp = params.comp[component];
  if (block_pos.x >= cp.blocks_w || block_pos.y >= cp.blocks_h) return;
  const uint block_index = block_pos.y * cp.blocks_w + block_pos.x;
  const uint imcu = min(block_pos.y / cp.v_samp_factor,
                        params.total_imcu_rows - 1);
  float block[64];
  for (uint k = 0; k < 64; ++k) {
    const short qi = coefficients[cp.coeff_offset + block_index * 64 + k];
    const float q = float(qi);
    const float bias = biases[(component * params.total_imcu_rows + imcu) * 64 + k];
    block[k] = qi == 0 ? 0.0f
                       : (q - copysign(bias, q)) * dequant[component * 64 + k];
  }
  float horizontal[64];
  for (uint y = 0; y < 8; ++y) {
    float input[8];
    float output[8];
    for (uint x = 0; x < 8; ++x) input[x] = block[y * 8 + x];
    idct8(input, output);
    for (uint x = 0; x < 8; ++x) horizontal[y * 8 + x] = output[x];
  }
  const uint plane_width = cp.blocks_w * 8;
  for (uint x = 0; x < 8; ++x) {
    float input[8];
    float output[8];
    for (uint y = 0; y < 8; ++y) input[y] = horizontal[y * 8 + x];
    idct8(input, output);
    for (uint y = 0; y < 8; ++y) {
      const uint px = block_pos.x * 8 + x;
      const uint py = block_pos.y * 8 + y;
      planes[cp.plane_offset + py * plane_width + px] = output[y];
    }
  }
}

inline float vertical_sample(device const float* planes,
                             constant ComponentParams& cp, uint x, uint y,
                             bool fancy) {
  const uint plane_width = cp.blocks_w * 8;
  const uint plane_height = cp.blocks_h * 8;
  if (cp.v_factor == 1) {
    return planes[cp.plane_offset + min(y, plane_height - 1) * plane_width + x];
  }
  const uint mid_y = min(y >> 1, plane_height - 1);
  if (!fancy) {
    return planes[cp.plane_offset + mid_y * plane_width + x];
  }
  const uint other_y = (y & 1) ? min(mid_y + 1, plane_height - 1)
                               : (mid_y == 0 ? 0 : mid_y - 1);
  const float mid = planes[cp.plane_offset + mid_y * plane_width + x];
  const float other = planes[cp.plane_offset + other_y * plane_width + x];
  return fma(other, 0.25f, mid * 0.75f);
}

inline float sample_component(device const float* planes,
                              constant ComponentParams& cp, uint x, uint y,
                              bool fancy) {
  const uint plane_width = cp.blocks_w * 8;
  if (cp.h_factor == 1) {
    return vertical_sample(planes, cp, min(x, plane_width - 1), y, fancy);
  }
  const uint mid_x = min(x >> 1, plane_width - 1);
  if (!fancy) return vertical_sample(planes, cp, mid_x, y, false);
  const uint other_x = (x & 1) ? min(mid_x + 1, plane_width - 1)
                               : (mid_x == 0 ? 0 : mid_x - 1);
  const float mid = vertical_sample(planes, cp, mid_x, y, true);
  const float other = vertical_sample(planes, cp, other_x, y, true);
  return fma(other, 0.25f, mid * 0.75f);
}

inline uchar quantize(float centered) {
  const float value = clamp((centered + as_type<float>(0x3f008081u)) * 255.0f,
                            0.0f, 255.0f);
  return uchar(uint(rint(value)));
}

kernel void jpegli_convert(
    device const float* planes [[buffer(0)]],
    device uchar4* output [[buffer(1)]],
    constant DecodeParams& params [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
  if (pos.x >= params.width || pos.y >= params.height) return;
  const bool fancy = params.fancy_upsampling != 0;
  float c0 = sample_component(planes, params.comp[0], pos.x, pos.y, fancy);
  float r;
  float g;
  float b;
  if (params.num_components == 1) {
    r = c0;
    g = c0;
    b = c0;
  } else {
    const float c1 = sample_component(planes, params.comp[1], pos.x, pos.y, fancy);
    const float c2 = sample_component(planes, params.comp[2], pos.x, pos.y, fancy);
    if (params.jpeg_color_space == 3) {  // JCS_YCbCr
      r = fma(as_type<float>(0x3fb374bcu), c2, c0);
      g = fma(as_type<float>(0xbf36d1a2u), c2,
              fma(as_type<float>(0xbeb032a1u), c1, c0));
      b = fma(as_type<float>(0x3fe2d0e5u), c1, c0);
    } else {  // JCS_RGB
      r = c0;
      g = c1;
      b = c2;
    }
  }
  output[pos.y * params.output_row_pixels + pos.x] =
      uchar4(quantize(r), quantize(g), quantize(b), uchar(255));
}
)metal";

struct ScratchBuffers {
  id<MTLBuffer> coefficients;
  id<MTLBuffer> dequant;
  id<MTLBuffer> biases;
  id<MTLBuffer> planes;
  size_t coefficient_capacity = 0;
  size_t dequant_capacity = 0;
  size_t bias_capacity = 0;
  size_t plane_capacity = 0;

  size_t Capacity() const {
    return coefficient_capacity + dequant_capacity + bias_capacity + plane_capacity;
  }
};

class MetalContext {
 public:
  static std::shared_ptr<MetalContext> Create(uint64_t* initialization_ns, std::string* error) {
    const uint64_t start = NowNs();
    std::shared_ptr<MetalContext> context(new (std::nothrow) MetalContext());
    if (!context) {
      *error = "unable to allocate Metal context";
      return nullptr;
    }
    context->device_ = MTLCreateSystemDefaultDevice();
    if (context->device_ == nil) {
      *error = "no Apple Metal device is available";
      return nullptr;
    }
    context->queue_ = [context->device_ newCommandQueue];
    if (context->queue_ == nil) {
      *error = "unable to create Metal command queue";
      return nullptr;
    }
    NSError* ns_error = nil;
#if defined(JPEGLI_APPLE_METAL_PRECOMPILED_LIBRARY)
    dispatch_data_t library_data =
        dispatch_data_create(kJpegliAppleMetalLibraryBytes,
                             sizeof(kJpegliAppleMetalLibraryBytes), nullptr,
                             DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    context->library_ =
        [context->device_ newLibraryWithData:library_data error:&ns_error];
#endif
    if (context->library_ == nil) {
      MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
      if (@available(macOS 15.0, iOS 18.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
      } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
      }
      ns_error = nil;
      context->library_ = [context->device_ newLibraryWithSource:@(kMetalSource)
                                                         options:options
                                                           error:&ns_error];
    }
    if (context->library_ == nil) {
      *error = "Metal shader compilation failed: " +
               std::string(ns_error.localizedDescription.UTF8String ?: "unknown error");
      return nullptr;
    }
    id<MTLFunction> idct = [context->library_ newFunctionWithName:@"jpegli_idct"];
    id<MTLFunction> convert = [context->library_ newFunctionWithName:@"jpegli_convert"];
    context->idct_pipeline_ = [context->device_ newComputePipelineStateWithFunction:idct
                                                                              error:&ns_error];
    if (context->idct_pipeline_ == nil) {
      *error = "Metal IDCT pipeline creation failed: " +
               std::string(ns_error.localizedDescription.UTF8String ?: "unknown error");
      return nullptr;
    }
    context->convert_pipeline_ = [context->device_ newComputePipelineStateWithFunction:convert
                                                                                 error:&ns_error];
    if (context->convert_pipeline_ == nil) {
      *error = "Metal conversion pipeline creation failed: " +
               std::string(ns_error.localizedDescription.UTF8String ?: "unknown error");
      return nullptr;
    }
    const BOOL stage_counters =
        [context->device_ supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
    if (stage_counters) {
      id<MTLCounterSet> timestamp_set = nil;
      for (id<MTLCounterSet> counter_set in context->device_.counterSets) {
        if ([counter_set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
          timestamp_set = counter_set;
          break;
        }
      }
      if (timestamp_set != nil) {
        MTLCounterSampleBufferDescriptor* descriptor =
            [[MTLCounterSampleBufferDescriptor alloc] init];
        descriptor.counterSet = timestamp_set;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.sampleCount = 4;
        context->timestamp_samples_ =
            [context->device_ newCounterSampleBufferWithDescriptor:descriptor error:&ns_error];
      }
    }
    *initialization_ns = NowNs() - start;
    return context;
  }

  bool Acquire(size_t coefficient_size, size_t dequant_size, size_t bias_size, size_t plane_size,
               ScratchBuffers* result, std::string* error) {
    {
      std::lock_guard<std::mutex> lock(scratch_mutex_);
      if (cached_.coefficients != nil && cached_.coefficient_capacity >= coefficient_size &&
          cached_.dequant_capacity >= dequant_size && cached_.bias_capacity >= bias_size &&
          cached_.plane_capacity >= plane_size) {
        *result = std::move(cached_);
        cached_ = {};
        return true;
      }
      cached_ = {};
    }
    result->coefficients = [device_
        newBufferWithLength:coefficient_size
                    options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
    result->dequant = [device_
        newBufferWithLength:dequant_size
                    options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
    result->biases = [device_
        newBufferWithLength:bias_size
                    options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
    result->planes = [device_ newBufferWithLength:plane_size options:MTLResourceStorageModeShared];
    if (result->coefficients == nil || result->dequant == nil || result->biases == nil ||
        result->planes == nil) {
      *result = {};
      *error = "Metal buffer allocation failed";
      return false;
    }
    result->coefficient_capacity = coefficient_size;
    result->dequant_capacity = dequant_size;
    result->bias_capacity = bias_size;
    result->plane_capacity = plane_size;
    return true;
  }

  void Return(ScratchBuffers buffers) {
    if (buffers.Capacity() > kMaxCachedScratch) return;
    std::lock_guard<std::mutex> lock(scratch_mutex_);
    if (cached_.coefficients == nil || buffers.Capacity() > cached_.Capacity()) {
      cached_ = std::move(buffers);
    }
  }

  void ReleaseScratch() {
    std::lock_guard<std::mutex> lock(scratch_mutex_);
    cached_ = {};
  }

  id<MTLDevice> device() const { return device_; }
  id<MTLCommandQueue> queue() const { return queue_; }
  id<MTLComputePipelineState> idct_pipeline() const { return idct_pipeline_; }
  id<MTLComputePipelineState> convert_pipeline() const { return convert_pipeline_; }
  id<MTLCounterSampleBuffer> timestamp_samples() const { return timestamp_samples_; }
  std::mutex& command_mutex() { return command_mutex_; }

 private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLLibrary> library_;
  id<MTLComputePipelineState> idct_pipeline_;
  id<MTLComputePipelineState> convert_pipeline_;
  id<MTLCounterSampleBuffer> timestamp_samples_;
  std::mutex command_mutex_;
  std::mutex scratch_mutex_;
  ScratchBuffers cached_;
};

std::mutex g_context_mutex;
std::shared_ptr<MetalContext> g_context;
std::once_flag g_availability_once;
int g_metal_available = 0;

std::shared_ptr<MetalContext> GetContext(uint64_t* initialization_ns, std::string* error) {
  std::lock_guard<std::mutex> lock(g_context_mutex);
  if (g_context) {
    *initialization_ns = 0;
    return g_context;
  }
  g_context = MetalContext::Create(initialization_ns, error);
  return g_context;
}

struct DecoderState {
  ~DecoderState() {
    if (context) context->Return(std::move(scratch));
  }
  std::shared_ptr<MetalContext> context;
  ScratchBuffers scratch;
  id<MTLBuffer> output;
  id<MTLTexture> texture;
  DecodeParams params = {};
  size_t output_row_bytes = 0;
};

struct OutputHandle {
  id<MTLBuffer> buffer;
  id<MTLTexture> texture;
};

bool BasicEligibility(j_decompress_ptr cinfo, bool direct_output, const char** reason) {
  jpeg_decomp_master* m = cinfo->master;
  if (m->apple_metal_mode_ == JPEGLI_APPLE_METAL_DISABLED) {
    *reason = "Metal was disabled for this decoder";
    return false;
  }
  if (!jpegli_apple_metal_is_available()) {
    *reason = "no Apple Metal device is available";
    return false;
  }
  if (cinfo->buffered_image) {
    *reason = "incremental buffered-image output uses the CPU renderer";
    return false;
  }
  if (cinfo->raw_data_out || cinfo->quantize_colors) {
    *reason = "raw or color-quantized output is not a Metal path";
    return false;
  }
  if (m->output_data_type_ != JPEGLI_TYPE_UINT8) {
    *reason = "Metal currently supports 8-bit output only";
    return false;
  }
  if (cinfo->scale_num != 1 || cinfo->scale_denom != 1) {
    *reason = "scaled IDCT output uses the CPU renderer";
    return false;
  }
#if defined(JCS_ALPHA_EXTENSIONS)
  if (cinfo->out_color_space != JCS_EXT_RGBA) {
    *reason = "Metal scanline output currently requires JCS_EXT_RGBA";
    return false;
  }
#else
  *reason = "this libjpeg ABI has no RGBA output colorspace";
  return false;
#endif
  if (cinfo->num_components != 1 && cinfo->num_components != 3) {
    *reason = "Metal supports grayscale and three-component JPEGs";
    return false;
  }
  if (cinfo->jpeg_color_space != JCS_GRAYSCALE && cinfo->jpeg_color_space != JCS_RGB &&
      cinfo->jpeg_color_space != JCS_YCbCr) {
    *reason = "JPEG colorspace is not implemented by the Metal kernel";
    return false;
  }
  for (int c = 0; c < cinfo->num_components; ++c) {
    if ((m->h_factor[c] != 1 && m->h_factor[c] != 2) ||
        (m->v_factor[c] != 1 && m->v_factor[c] != 2)) {
      *reason = "sampling factors are outside 4:4:4/4:2:2/4:2:0";
      return false;
    }
  }
  size_t pixels = 0;
  if (!SafeMul(cinfo->output_width, cinfo->output_height, &pixels)) {
    *reason = "image dimensions overflow Metal size calculations";
    return false;
  }
  if (!direct_output && m->apple_metal_mode_ == JPEGLI_APPLE_METAL_AUTO &&
      pixels < g_crossover_pixels.load(std::memory_order_relaxed)) {
    *reason = "image is below the measured CPU/Metal crossover";
    return false;
  }
  return true;
}

void ComputeBiases(j_decompress_ptr cinfo, float* output) {
  jpeg_decomp_master* m = cinfo->master;
  const size_t rows = cinfo->total_iMCU_rows;
  memset(output, 0, static_cast<size_t>(cinfo->num_components) * rows * DCTSIZE2 * sizeof(float));
  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info& comp = cinfo->comp_info[c];
    if (comp.h_samp_factor != cinfo->max_h_samp_factor ||
        comp.v_samp_factor != cinfo->max_v_samp_factor) {
      continue;
    }
    std::array<int, DCTSIZE2> nonzeros = {};
    std::array<int, DCTSIZE2> sumabs = {};
    std::array<float, DCTSIZE2> biases = {};
    size_t num_blocks = 0;
    for (size_t imcu = 0; imcu < rows; ++imcu) {
      const size_t by0 = imcu * comp.v_samp_factor;
      const size_t nrows = std::min<size_t>(comp.v_samp_factor, comp.height_in_blocks - by0);
      JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(reinterpret_cast<j_common_ptr>(cinfo),
                                                             m->coef_arrays[c], by0, nrows, FALSE);
      for (size_t iy = 0; iy < nrows; ++iy) {
        const int16_t* coeff = &blocks[iy][0][0];
        GatherBlockStats(coeff, comp.width_in_blocks * DCTSIZE2, nonzeros.data(), sumabs.data());
        num_blocks += comp.width_in_blocks;
      }
      if ((imcu & 3) == 3) {
        ComputeOptimalLaplacianBiases(static_cast<int>(num_blocks), nonzeros.data(), sumabs.data(),
                                      biases.data());
      }
      memcpy(output + (static_cast<size_t>(c) * rows + imcu) * DCTSIZE2, biases.data(),
             DCTSIZE2 * sizeof(float));
    }
  }
}

bool FillParamsAndSizes(j_decompress_ptr cinfo, DecodeParams* params, size_t* coefficient_size,
                        size_t* dequant_size, size_t* bias_size, size_t* plane_size,
                        size_t* output_size, size_t* output_row_bytes, std::string* error) {
  jpeg_decomp_master* m = cinfo->master;
  *params = {};
  params->width = cinfo->output_width;
  params->height = cinfo->output_height;
  params->num_components = cinfo->num_components;
  params->jpeg_color_space = cinfo->jpeg_color_space;
  params->fancy_upsampling = cinfo->do_fancy_upsampling ? 1 : 0;
  params->total_imcu_rows = cinfo->total_iMCU_rows;
  size_t coeff_total = 0;
  size_t plane_total = 0;
  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info& comp = cinfo->comp_info[c];
    size_t blocks = 0;
    size_t coeff_count = 0;
    size_t plane_width = 0;
    size_t plane_height = 0;
    size_t plane_count = 0;
    if (!SafeMul(comp.width_in_blocks, comp.height_in_blocks, &blocks) ||
        !SafeMul(blocks, DCTSIZE2, &coeff_count) ||
        !SafeMul(comp.width_in_blocks, DCTSIZE, &plane_width) ||
        !SafeMul(comp.height_in_blocks, DCTSIZE, &plane_height) ||
        !SafeMul(plane_width, plane_height, &plane_count) ||
        coeff_total > std::numeric_limits<uint32_t>::max() ||
        plane_total > std::numeric_limits<uint32_t>::max()) {
      *error = "Metal component size overflow";
      return false;
    }
    ComponentParams& cp = params->comp[c];
    cp.coeff_offset = static_cast<uint32_t>(coeff_total);
    cp.plane_offset = static_cast<uint32_t>(plane_total);
    cp.blocks_w = comp.width_in_blocks;
    cp.blocks_h = comp.height_in_blocks;
    cp.h_factor = m->h_factor[c];
    cp.v_factor = m->v_factor[c];
    cp.v_samp_factor = comp.v_samp_factor;
    if (!SafeAdd(coeff_total, coeff_count, &coeff_total) ||
        !SafeAdd(plane_total, plane_count, &plane_total)) {
      *error = "Metal aggregate plane size overflow";
      return false;
    }
  }
  size_t coeff_bytes = 0;
  size_t plane_bytes = 0;
  size_t bias_count = 0;
  if (!SafeMul(coeff_total, sizeof(int16_t), &coeff_bytes) ||
      !SafeMul(plane_total, sizeof(float), &plane_bytes) ||
      !SafeMul(static_cast<size_t>(cinfo->num_components), cinfo->total_iMCU_rows, &bias_count) ||
      !SafeMul(bias_count, DCTSIZE2 * sizeof(float), bias_size)) {
    *error = "Metal byte size overflow";
    return false;
  }
  *coefficient_size = AlignUp(coeff_bytes, kMetalAlignment);
  *dequant_size = AlignUp(static_cast<size_t>(cinfo->num_components) * DCTSIZE2 * sizeof(float),
                          kMetalAlignment);
  *plane_size = AlignUp(plane_bytes, kMetalAlignment);
  *bias_size = AlignUp(*bias_size, kMetalAlignment);
  *output_row_bytes = AlignUp(static_cast<size_t>(cinfo->output_width) * 4, kMetalAlignment);
  if (*coefficient_size == 0 || *dequant_size == 0 || *plane_size == 0 || *bias_size == 0 ||
      *output_row_bytes == 0 || !SafeMul(*output_row_bytes, cinfo->output_height, output_size)) {
    *error = "Metal aligned size overflow";
    return false;
  }
  params->output_row_pixels = *output_row_bytes / 4;
  size_t working = 0;
  if (!SafeAdd(*coefficient_size, *dequant_size, &working) ||
      !SafeAdd(working, *bias_size, &working) || !SafeAdd(working, *plane_size, &working) ||
      !SafeAdd(working, *output_size, &working) || working > kMaxMetalWorkingSet) {
    *error = "image exceeds the bounded 512 MiB Metal working set";
    return false;
  }
  return true;
}

void CopyCoefficients(j_decompress_ptr cinfo, const DecodeParams& params, int16_t* destination) {
  jpeg_decomp_master* m = cinfo->master;
  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info& comp = cinfo->comp_info[c];
    int16_t* dst = destination + params.comp[c].coeff_offset;
    for (size_t by = 0; by < comp.height_in_blocks;) {
      const size_t rows = std::min<size_t>(comp.v_samp_factor, comp.height_in_blocks - by);
      JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(reinterpret_cast<j_common_ptr>(cinfo),
                                                             m->coef_arrays[c], by, rows, FALSE);
      const size_t row_bytes = static_cast<size_t>(comp.width_in_blocks) * sizeof(JBLOCK);
      for (size_t iy = 0; iy < rows; ++iy) {
        memcpy(dst + (by + iy) * comp.width_in_blocks * DCTSIZE2, &blocks[iy][0][0], row_bytes);
      }
      by += rows;
    }
  }
}

bool CompleteCommandBuffer(id<MTLCommandBuffer> command_buffer, uint64_t wall_start,
                           uint64_t* gpu_ns, uint64_t* overhead_ns, std::string* error) {
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  const uint64_t wall_ns = NowNs() - wall_start;
  if (command_buffer.status == MTLCommandBufferStatusError) {
    *error = "Metal command buffer failed: " +
             std::string(command_buffer.error.localizedDescription.UTF8String ?: "unknown error");
    return false;
  }
  const double start = command_buffer.GPUStartTime;
  const double end = command_buffer.GPUEndTime;
  *gpu_ns = end > start ? static_cast<uint64_t>((end - start) * 1.0e9) : 0;
  *overhead_ns += wall_ns > *gpu_ns ? wall_ns - *gpu_ns : 0;
  return true;
}

}  // namespace

bool AppleMetalShouldAttempt(j_decompress_ptr cinfo, bool direct_output) {
  const char* reason = nullptr;
  if (!BasicEligibility(cinfo, direct_output, &reason)) {
    SetAppleMetalFallbackReason(cinfo, reason);
    return false;
  }
  // Reject oversized layouts before jpegli_start_decompress() chooses a full
  // coefficient buffer. This preserves the CPU decoder's bounded streaming
  // behavior when the Metal working-set cap would reject reconstruction.
  DecodeParams params = {};
  size_t coefficient_size = 0;
  size_t dequant_size = 0;
  size_t bias_size = 0;
  size_t plane_size = 0;
  size_t output_size = 0;
  size_t output_row_bytes = 0;
  std::string error;
  if (!FillParamsAndSizes(cinfo, &params, &coefficient_size, &dequant_size, &bias_size, &plane_size,
                          &output_size, &output_row_bytes, &error)) {
    SetAppleMetalFallbackReason(cinfo, error.c_str());
    return false;
  }
  SetAppleMetalFallbackReason(cinfo, "");
  return true;
}

bool AppleMetalReconstruct(j_decompress_ptr cinfo, bool direct_output) {
  @autoreleasepool {
    jpeg_decomp_master* m = cinfo->master;
    const uint64_t total_start = NowNs();
    const char* eligibility_reason = nullptr;
    if (!BasicEligibility(cinfo, direct_output, &eligibility_reason)) {
      SetAppleMetalFallbackReason(cinfo, eligibility_reason);
      return false;
    }
    if (m->apply_smoothing) {
      SetAppleMetalFallbackReason(cinfo,
                                  "incremental progressive block smoothing uses the CPU renderer");
      return false;
    }

    std::string error;
    uint64_t initialization_ns = 0;
    std::shared_ptr<MetalContext> context = GetContext(&initialization_ns, &error);
    m->apple_metal_stats_.metal_initialization_ns = initialization_ns;
    if (!context) {
      SetAppleMetalFallbackReason(cinfo, error.c_str());
      return false;
    }

    std::unique_ptr<DecoderState> state(new (std::nothrow) DecoderState());
    if (!state) {
      SetAppleMetalFallbackReason(cinfo, "unable to allocate Metal decoder state");
      return false;
    }
    state->context = context;
    size_t coefficient_size = 0;
    size_t dequant_size = 0;
    size_t bias_size = 0;
    size_t plane_size = 0;
    size_t output_size = 0;
    if (!FillParamsAndSizes(cinfo, &state->params, &coefficient_size, &dequant_size, &bias_size,
                            &plane_size, &output_size, &state->output_row_bytes, &error)) {
      SetAppleMetalFallbackReason(cinfo, error.c_str());
      return false;
    }
    if (!context->Acquire(coefficient_size, dequant_size, bias_size, plane_size, &state->scratch,
                          &error)) {
      SetAppleMetalFallbackReason(cinfo, error.c_str());
      return false;
    }
    state->output = [context->device() newBufferWithLength:output_size
                                                   options:MTLResourceStorageModeShared];
    if (state->output == nil) {
      SetAppleMetalFallbackReason(cinfo, "Metal output buffer allocation failed");
      return false;
    }
    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:cinfo->output_width
                                                          height:cinfo->output_height
                                                       mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    descriptor.storageMode = MTLStorageModeShared;
    state->texture = [state->output newTextureWithDescriptor:descriptor
                                                      offset:0
                                                 bytesPerRow:state->output_row_bytes];
    if (state->texture == nil) {
      SetAppleMetalFallbackReason(cinfo, "unable to create RGBA8 Metal texture view");
      return false;
    }

    const uint64_t analysis_start = NowNs();
    ComputeBiases(cinfo, static_cast<float*>(state->scratch.biases.contents));
    m->apple_metal_stats_.coefficient_analysis_ns = NowNs() - analysis_start;

    const uint64_t copy_start = NowNs();
    CopyCoefficients(cinfo, state->params,
                     static_cast<int16_t*>(state->scratch.coefficients.contents));
    memcpy(state->scratch.dequant.contents, m->dequant_,
           static_cast<size_t>(cinfo->num_components) * DCTSIZE2 * sizeof(float));
    m->apple_metal_stats_.coefficient_copy_ns = NowNs() - copy_start;

    uint64_t encode_ns = 0;
    uint64_t overhead_ns = 0;
    uint64_t idct_gpu_ns = 0;
    uint64_t color_gpu_ns = 0;
    uint64_t total_gpu_ns = 0;
    std::unique_lock<std::mutex> command_lock(context->command_mutex());
    id<MTLCounterSampleBuffer> timestamp_samples = context->timestamp_samples();
    const uint64_t encode_start = NowNs();
    id<MTLCommandBuffer> command = [context->queue() commandBuffer];
    MTLComputePassDescriptor* idct_pass = nil;
    if (timestamp_samples != nil) {
      idct_pass = [MTLComputePassDescriptor computePassDescriptor];
      MTLComputePassSampleBufferAttachmentDescriptor* attachment =
          idct_pass.sampleBufferAttachments[0];
      attachment.sampleBuffer = timestamp_samples;
      attachment.startOfEncoderSampleIndex = 0;
      attachment.endOfEncoderSampleIndex = 1;
    }
    id<MTLComputeCommandEncoder> idct_encoder =
        idct_pass == nil ? [command computeCommandEncoder]
                         : [command computeCommandEncoderWithDescriptor:idct_pass];
    [idct_encoder setComputePipelineState:context->idct_pipeline()];
    [idct_encoder setBuffer:state->scratch.coefficients offset:0 atIndex:0];
    [idct_encoder setBuffer:state->scratch.dequant offset:0 atIndex:1];
    [idct_encoder setBuffer:state->scratch.biases offset:0 atIndex:2];
    [idct_encoder setBuffer:state->scratch.planes offset:0 atIndex:3];
    [idct_encoder setBytes:&state->params length:sizeof(state->params) atIndex:4];
    const MTLSize block_threads = MTLSizeMake(8, 8, 1);
    for (uint32_t c = 0; c < state->params.num_components; ++c) {
      [idct_encoder setBytes:&c length:sizeof(c) atIndex:5];
      [idct_encoder dispatchThreads:MTLSizeMake(state->params.comp[c].blocks_w,
                                                state->params.comp[c].blocks_h, 1)
              threadsPerThreadgroup:block_threads];
    }
    [idct_encoder endEncoding];
    MTLComputePassDescriptor* color_pass = nil;
    if (timestamp_samples != nil) {
      color_pass = [MTLComputePassDescriptor computePassDescriptor];
      MTLComputePassSampleBufferAttachmentDescriptor* attachment =
          color_pass.sampleBufferAttachments[0];
      attachment.sampleBuffer = timestamp_samples;
      attachment.startOfEncoderSampleIndex = 2;
      attachment.endOfEncoderSampleIndex = 3;
    }
    id<MTLComputeCommandEncoder> color_encoder =
        color_pass == nil ? [command computeCommandEncoder]
                          : [command computeCommandEncoderWithDescriptor:color_pass];
    [color_encoder setComputePipelineState:context->convert_pipeline()];
    [color_encoder setBuffer:state->scratch.planes offset:0 atIndex:0];
    [color_encoder setBuffer:state->output offset:0 atIndex:1];
    [color_encoder setBytes:&state->params length:sizeof(state->params) atIndex:2];
    [color_encoder dispatchThreads:MTLSizeMake(cinfo->output_width, cinfo->output_height, 1)
             threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [color_encoder endEncoding];
    encode_ns = NowNs() - encode_start;
    if (!CompleteCommandBuffer(command, NowNs(), &total_gpu_ns, &overhead_ns, &error)) {
      SetAppleMetalFallbackReason(cinfo, error.c_str());
      return false;
    }
    if (timestamp_samples != nil && total_gpu_ns != 0) {
      NSData* resolved = [timestamp_samples resolveCounterRange:NSMakeRange(0, 4)];
      if (resolved.length >= 4 * sizeof(MTLCounterResultTimestamp)) {
        const MTLCounterResultTimestamp* timestamps =
            static_cast<const MTLCounterResultTimestamp*>(resolved.bytes);
        const uint64_t begin = timestamps[0].timestamp;
        const uint64_t idct_end = timestamps[1].timestamp;
        const uint64_t color_begin = timestamps[2].timestamp;
        const uint64_t end = timestamps[3].timestamp;
        if (begin != MTLCounterErrorValue && idct_end != MTLCounterErrorValue &&
            color_begin != MTLCounterErrorValue && end != MTLCounterErrorValue && end > begin &&
            idct_end >= begin && end >= color_begin) {
          const double nanoseconds_per_tick =
              static_cast<double>(total_gpu_ns) / static_cast<double>(end - begin);
          idct_gpu_ns =
              static_cast<uint64_t>(static_cast<double>(idct_end - begin) * nanoseconds_per_tick);
          color_gpu_ns =
              static_cast<uint64_t>(static_cast<double>(end - color_begin) * nanoseconds_per_tick);
        }
      }
    }
    command_lock.unlock();
    m->apple_metal_stats_.command_encoding_ns = encode_ns;
    m->apple_metal_stats_.submission_overhead_ns = overhead_ns;
    m->apple_metal_stats_.gpu_dequant_idct_ns = idct_gpu_ns;
    m->apple_metal_stats_.gpu_upsample_color_ns = color_gpu_ns;
    m->apple_metal_stats_.cpu_decoder_bytes =
        MemoryManagerCurrentBytes(reinterpret_cast<j_common_ptr>(cinfo));
    m->apple_metal_stats_.metal_buffer_bytes = state->scratch.Capacity() + output_size;
    m->apple_metal_stats_.decoder_retained_bytes =
        m->apple_metal_stats_.cpu_decoder_bytes + m->apple_metal_stats_.metal_buffer_bytes;
    m->apple_metal_stats_.reconstruction_total_ns = NowNs() - total_start;
    m->apple_metal_stats_.used_metal = 1;
    m->apple_metal_stats_.direct_output = direct_output ? 1 : 0;
    SetAppleMetalFallbackReason(cinfo, "");
    m->apple_metal_decoder_ = state.release();
    return true;
  }
}

JDIMENSION AppleMetalReadScanlines(j_decompress_ptr cinfo, JSAMPARRAY scanlines,
                                   JDIMENSION max_lines) {
  DecoderState* state = static_cast<DecoderState*>(cinfo->master->apple_metal_decoder_);
  if (state == nullptr) return 0;
  const uint64_t start = NowNs();
  const uint8_t* source = static_cast<const uint8_t*>(state->output.contents);
  const size_t output_bytes = static_cast<size_t>(cinfo->output_width) * 4;
  JDIMENSION rows = 0;
  while (rows < max_lines && cinfo->output_scanline < cinfo->output_height) {
    if (scanlines != nullptr) {
      memcpy(scanlines[rows],
             source + static_cast<size_t>(cinfo->output_scanline) * state->output_row_bytes +
                 cinfo->master->xoffset_ * 4,
             output_bytes);
    }
    ++rows;
    ++cinfo->output_scanline;
  }
  if (rows != 0 && cinfo->output_scanline == cinfo->output_height) {
    cinfo->output_iMCU_row = cinfo->total_iMCU_rows;
    ++cinfo->master->output_passes_done_;
  }
  cinfo->master->apple_metal_stats_.cpu_output_copy_ns += NowNs() - start;
  return rows;
}

bool AppleMetalExportOutput(j_decompress_ptr cinfo, JpegliAppleMetalOutput* output) {
  DecoderState* state = static_cast<DecoderState*>(cinfo->master->apple_metal_decoder_);
  if (state == nullptr || state->output == nil || state->texture == nil) {
    return false;
  }
  OutputHandle* handle = new (std::nothrow) OutputHandle();
  if (handle == nullptr) return false;
  handle->buffer = state->output;
  handle->texture = state->texture;
  output->buffer = (__bridge void*)handle->buffer;
  output->texture = (__bridge void*)handle->texture;
  output->buffer_contents = handle->buffer.contents;
  output->width = cinfo->output_width;
  output->height = cinfo->output_height;
  output->row_bytes = state->output_row_bytes;
  output->pixel_format = JPEGLI_APPLE_METAL_PIXEL_FORMAT_RGBA8_UNORM;
  output->private_handle = handle;
  return true;
}

bool AppleMetalUploadCpuOutput(j_decompress_ptr cinfo, const uint8_t* pixels,
                               size_t source_row_bytes, JpegliAppleMetalOutput* output) {
  @autoreleasepool {
    const uint64_t start = NowNs();
    std::string error;
    uint64_t initialization_ns = 0;
    std::shared_ptr<MetalContext> context = GetContext(&initialization_ns, &error);
    if (!context) {
      SetAppleMetalFallbackReason(cinfo, error.c_str());
      return false;
    }
    const size_t row_bytes = AlignUp(static_cast<size_t>(cinfo->output_width) * 4, kMetalAlignment);
    size_t output_size = 0;
    if (row_bytes == 0 || !SafeMul(row_bytes, cinfo->output_height, &output_size)) {
      return false;
    }
    OutputHandle* handle = new (std::nothrow) OutputHandle();
    if (handle == nullptr) return false;
    handle->buffer = [context->device() newBufferWithLength:output_size
                                                    options:MTLResourceStorageModeShared];
    if (handle->buffer == nil) {
      delete handle;
      return false;
    }
    const uint64_t copy_start = NowNs();
    uint8_t* destination = static_cast<uint8_t*>(handle->buffer.contents);
    for (size_t y = 0; y < cinfo->output_height; ++y) {
      memcpy(destination + y * row_bytes, pixels + y * source_row_bytes,
             static_cast<size_t>(cinfo->output_width) * 4);
    }
    const uint64_t copy_ns = NowNs() - copy_start;
    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:cinfo->output_width
                                                          height:cinfo->output_height
                                                       mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    descriptor.storageMode = MTLStorageModeShared;
    handle->texture = [handle->buffer newTextureWithDescriptor:descriptor
                                                        offset:0
                                                   bytesPerRow:row_bytes];
    if (handle->texture == nil) {
      delete handle;
      return false;
    }
    output->buffer = (__bridge void*)handle->buffer;
    output->texture = (__bridge void*)handle->texture;
    output->buffer_contents = handle->buffer.contents;
    output->width = cinfo->output_width;
    output->height = cinfo->output_height;
    output->row_bytes = row_bytes;
    output->pixel_format = JPEGLI_APPLE_METAL_PIXEL_FORMAT_RGBA8_UNORM;
    output->private_handle = handle;
    JpegliAppleMetalStats& stats = cinfo->master->apple_metal_stats_;
    stats.metal_initialization_ns += initialization_ns;
    stats.cpu_output_copy_ns += copy_ns;
    stats.cpu_decoder_bytes = MemoryManagerCurrentBytes(reinterpret_cast<j_common_ptr>(cinfo));
    stats.metal_buffer_bytes = output_size;
    stats.decoder_retained_bytes = stats.cpu_decoder_bytes + output_size;
    stats.reconstruction_total_ns += NowNs() - start;
    stats.direct_output = 1;
    // used_metal remains false: Metal owns the destination, but the CPU did
    // the unsupported reconstruction.
    return true;
  }
}

void AppleMetalResetDecoder(j_decompress_ptr cinfo) {
  if (cinfo == nullptr || cinfo->master == nullptr) return;
  DecoderState* state = static_cast<DecoderState*>(cinfo->master->apple_metal_decoder_);
  delete state;
  cinfo->master->apple_metal_decoder_ = nullptr;
  cinfo->master->apple_metal_active_ = false;
}

}  // namespace jpegli

void jpegli_apple_metal_set_crossover_pixels(size_t pixels) {
  jpegli::g_crossover_pixels.store(pixels, std::memory_order_relaxed);
}

size_t jpegli_apple_metal_get_crossover_pixels(void) {
  return jpegli::g_crossover_pixels.load(std::memory_order_relaxed);
}

void jpegli_apple_metal_release_output(JpegliAppleMetalOutput* output) {
  if (output == nullptr) return;
  delete static_cast<jpegli::OutputHandle*>(output->private_handle);
  *output = {};
}

void jpegli_apple_metal_release_cached_resources(JpegliAppleMetalReleaseMode mode) {
  std::lock_guard<std::mutex> lock(jpegli::g_context_mutex);
  if (!jpegli::g_context) return;
  jpegli::g_context->ReleaseScratch();
  if (mode == JPEGLI_APPLE_METAL_RELEASE_ALL) {
    jpegli::g_context.reset();
  }
}

int jpegli_apple_metal_is_available(void) {
  std::call_once(jpegli::g_availability_once, [] {
    @autoreleasepool {
      jpegli::g_metal_available = MTLCreateSystemDefaultDevice() != nil;
    }
  });
  return jpegli::g_metal_available;
}

#endif  // JPEGLI_ENABLE_APPLE_METAL
