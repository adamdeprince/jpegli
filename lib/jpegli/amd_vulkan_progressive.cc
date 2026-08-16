// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jpegli/amd_vulkan_progressive.h"

#if defined(JPEGLI_ENABLE_AMD_VULKAN)

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "amd_vulkan_frontend_spv.h"
#include "amd_vulkan_progressive_spv.h"
#include "lib/jpegli/encode_internal.h"
#include "lib/jpegli/encode_streaming.h"

namespace jpegli {
namespace {

constexpr uint32_t kAmdVendorId = 0x1002;
constexpr uint32_t kWaveSize = 64;
constexpr uint32_t kCompactBlocksPerGroup = 256;
constexpr size_t kCoefficientWordsPerBlock = DCTSIZE2 / 2;
constexpr size_t kAutoMinComponentBlocks = 4096;
constexpr size_t kPipelineSlots = 2;
constexpr uint32_t kTimestampQueries = 9;

bool EnvironmentDisablesGpu() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_PROGRESSIVE");
  if (value == nullptr) return false;
  return strcmp(value, "0") == 0 || strcmp(value, "off") == 0 ||
         strcmp(value, "false") == 0;
}

bool EnvironmentForcesGpu() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_PROGRESSIVE");
  return value != nullptr && strcmp(value, "force") == 0;
}

bool EnvironmentEnablesThroughputFusion() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_THROUGHPUT");
  if (value == nullptr) return false;
  return strcmp(value, "0") != 0 && strcmp(value, "off") != 0 &&
         strcmp(value, "false") != 0;
}

bool EnvironmentEnablesLatencyCompact() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_LATENCY_COMPACT");
  if (value == nullptr) return false;
  return strcmp(value, "0") != 0 && strcmp(value, "off") != 0 &&
         strcmp(value, "false") != 0;
}

bool EnvironmentEnablesLatencyCompactMode(uint32_t mode) {
  if (!EnvironmentEnablesLatencyCompact()) return false;
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_LATENCY_COMPACT");
  if (strcmp(value, "initial") == 0) return mode == 0;
  if (strcmp(value, "refinement") == 0) return mode == 1;
  return true;
}

bool EnvironmentEnablesDC() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_DC");
  if (value == nullptr) return true;
  return strcmp(value, "0") != 0 && strcmp(value, "off") != 0 &&
         strcmp(value, "false") != 0;
}

bool EnvironmentEnablesFrontend() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_FRONTEND");
  if (value == nullptr) return false;
  return strcmp(value, "0") != 0 && strcmp(value, "off") != 0 &&
         strcmp(value, "false") != 0;
}

bool EnvironmentVerifiesFrontend() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_FRONTEND_VERIFY");
  return value != nullptr && strcmp(value, "0") != 0;
}

bool TraceEnabled() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_TRACE");
  return value != nullptr && strcmp(value, "0") != 0;
}

void Trace(const char* message) {
  if (TraceEnabled()) fprintf(stderr, "jpegli amd-vulkan: %s\n", message);
}

struct MappedBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
};

enum class AmdTuningProfile {
  kPhoenix,
  kStrixHalo,
  kOtherUnifiedAmd,
};

struct PushConstants {
  uint32_t coefficient_word_offset;
  uint32_t num_blocks;
  uint32_t spectral_start;
  uint32_t spectral_end;
  uint32_t successive_low;
  uint32_t output_stride_words;
  uint32_t context;
  uint32_t dispatch_width;
  uint32_t mode;
  uint32_t output_word_offset;
  uint32_t scan_parameter_offset;
  uint32_t scan_parameter_count;
  uint32_t compact_header_word_offset;
  uint32_t compact_token_word_offset;
  uint32_t compact_refbit_word_offset;
  uint32_t compact_eobrun_word_offset;
  uint32_t compact_scratch_word_offset;
  uint32_t compact_scratch_stride_words;
  uint32_t compact_token_capacity;
  uint32_t compact_refbit_capacity;
};
static_assert(sizeof(PushConstants) == 80, "shader push-constant ABI changed");

struct FrontendPushConstants {
  uint32_t coefficient_word_offset;
  uint32_t num_blocks;
  uint32_t plane_float_offset;
  uint32_t plane_stride;
  uint32_t width_blocks;
  uint32_t height_blocks;
  uint32_t quant_field_float_offset;
  uint32_t quant_field_stride;
  uint32_t quant_mul_float_offset;
  uint32_t zero_bias_offset_float_offset;
  uint32_t zero_bias_mul_float_offset;
  uint32_t dc_float_offset;
  uint32_t dispatch_width;
  uint32_t mode;
  uint32_t h_factor;
  uint32_t max_v_samp_factor;
  uint32_t xsize_mcus;
  uint32_t h_samp_factor;
  uint32_t v_samp_factor;
  uint32_t adaptive_quant;
};
static_assert(sizeof(FrontendPushConstants) == sizeof(PushConstants),
              "front-end push-constant ABI changed");

struct PipelineSlot {
  j_compress_ptr cinfo = nullptr;
  MappedBuffer coefficient_buffer;
  MappedBuffer descriptor_buffer;
  MappedBuffer compact_buffer;
  MappedBuffer scan_parameter_buffer;
  MappedBuffer frontend_buffer;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool query_pool = VK_NULL_HANDLE;
  bool descriptors_dirty = true;
  bool in_flight = false;
  bool ready = false;
  bool frontend_prepared = false;
  std::array<size_t, kMaxComponents> component_word_offsets = {};
  std::array<size_t, kMaxComponents> component_num_blocks = {};
  std::array<FrontendPushConstants, kMaxComponents> frontend_push = {};
  std::vector<AmdVulkanACResult> cached_results;
  std::chrono::steady_clock::time_point submit_time;
};

class AmdVulkanProgressiveTokenizer {
 public:
  ~AmdVulkanProgressiveTokenizer() { Shutdown(); }

  bool PrepareFrontend(j_compress_ptr cinfo) {
    if (!EnvironmentEnablesFrontend() || cinfo == nullptr ||
        cinfo->master == nullptr || !cinfo->progressive_mode ||
        cinfo->master->psnr_target > 0 || EnvironmentDisablesGpu() ||
        cinfo->num_components <= 0 || cinfo->num_components > kMaxComponents) {
      return false;
    }
    for (int scan_index = 0; scan_index < cinfo->num_scans; ++scan_index) {
      const jpeg_scan_info& scan = cinfo->scan_info[scan_index];
      if (scan.comps_in_scan != 1 || scan.Al < 0 || scan.Al > 15 ||
          cinfo->master->scan_token_info[scan_index].restart_interval != 0) {
        return false;
      }
      if (scan.Ss == 0) {
        if (scan.Se != 0 || scan.Ah != 0) return false;
      } else if (scan.Ss < 1 || scan.Se < scan.Ss || scan.Se >= DCTSIZE2) {
        return false;
      }
    }

    size_t image_blocks = 0;
    for (int c = 0; c < cinfo->num_components; ++c) {
      const jpeg_component_info& comp = cinfo->comp_info[c];
      image_blocks +=
          static_cast<size_t>(comp.width_in_blocks) * comp.height_in_blocks;
    }
    if (!EnvironmentForcesGpu() && image_blocks < kAutoMinComponentBlocks) {
      return false;
    }
    if (!Initialize()) return false;
    PipelineSlot* slot = FindFreeSlot();
    if (slot == nullptr) return false;

    size_t total_coefficient_words = 0;
    for (int c = 0; c < cinfo->num_components; ++c) {
      const jpeg_component_info& comp = cinfo->comp_info[c];
      const size_t num_blocks =
          static_cast<size_t>(comp.width_in_blocks) * comp.height_in_blocks;
      if (num_blocks >
          (std::numeric_limits<size_t>::max() - total_coefficient_words) /
              kCoefficientWordsPerBlock) {
        return false;
      }
      slot->component_word_offsets[c] = total_coefficient_words;
      slot->component_num_blocks[c] = num_blocks;
      total_coefficient_words += num_blocks * kCoefficientWordsPerBlock;
    }
    if (total_coefficient_words > std::numeric_limits<uint32_t>::max() ||
        !EnsureBuffer(slot, &slot->coefficient_buffer,
                      total_coefficient_words * sizeof(uint32_t))) {
      return false;
    }

    size_t total_floats = 0;
    const auto reserve_floats = [&](size_t count, uint32_t* offset) {
      total_floats = (total_floats + 15u) & ~size_t{15u};
      if (total_floats > std::numeric_limits<uint32_t>::max() ||
          count > std::numeric_limits<uint32_t>::max() - total_floats) {
        return false;
      }
      *offset = static_cast<uint32_t>(total_floats);
      total_floats += count;
      return true;
    };
    std::array<uint32_t, kMaxComponents> plane_offsets = {};
    uint32_t quant_field_offset = 0;
    for (int c = 0; c < cinfo->num_components; ++c) {
      const jpeg_component_info& comp = cinfo->comp_info[c];
      const size_t stride = static_cast<size_t>(comp.width_in_blocks) * DCTSIZE;
      const size_t rows = static_cast<size_t>(comp.height_in_blocks) * DCTSIZE;
      if (rows != 0 && stride > std::numeric_limits<size_t>::max() / rows) {
        return false;
      }
      if (!reserve_floats(stride * rows, &plane_offsets[c])) return false;
    }
    jpeg_comp_master* master = cinfo->master;
    if (!reserve_floats(master->xsize_blocks * master->ysize_blocks,
                        &quant_field_offset)) {
      return false;
    }
    const uint32_t xsize_mcus = static_cast<uint32_t>(
        DivCeil(cinfo->image_width, DCTSIZE * cinfo->max_h_samp_factor));
    for (int c = 0; c < cinfo->num_components; ++c) {
      const jpeg_component_info& comp = cinfo->comp_info[c];
      FrontendPushConstants& push = slot->frontend_push[c];
      push = {};
      push.coefficient_word_offset =
          static_cast<uint32_t>(slot->component_word_offsets[c]);
      push.num_blocks = static_cast<uint32_t>(slot->component_num_blocks[c]);
      push.plane_float_offset = plane_offsets[c];
      push.plane_stride = comp.width_in_blocks * DCTSIZE;
      push.width_blocks = comp.width_in_blocks;
      push.height_blocks = comp.height_in_blocks;
      push.quant_field_float_offset = quant_field_offset;
      push.quant_field_stride = master->xsize_blocks;
      if (!reserve_floats(DCTSIZE2, &push.quant_mul_float_offset) ||
          !reserve_floats(DCTSIZE2, &push.zero_bias_offset_float_offset) ||
          !reserve_floats(DCTSIZE2, &push.zero_bias_mul_float_offset) ||
          !reserve_floats(push.num_blocks, &push.dc_float_offset)) {
        return false;
      }
      const size_t frontend_groups = (push.num_blocks + 7u) / 8u;
      push.dispatch_width = static_cast<uint32_t>(std::min<size_t>(
          frontend_groups, properties_.limits.maxComputeWorkGroupCount[0]));
      push.h_factor = master->h_factor[c];
      push.max_v_samp_factor = cinfo->max_v_samp_factor;
      push.xsize_mcus = xsize_mcus;
      push.h_samp_factor = comp.h_samp_factor;
      push.v_samp_factor = comp.v_samp_factor;
      push.adaptive_quant = master->use_adaptive_quantization ? 1u : 0u;
    }
    if (!EnsureBuffer(slot, &slot->frontend_buffer,
                      total_floats * sizeof(float))) {
      return false;
    }
    float* frontend = static_cast<float*>(slot->frontend_buffer.mapped);
    for (int c = 0; c < cinfo->num_components; ++c) {
      const FrontendPushConstants& push = slot->frontend_push[c];
      master->amd_vulkan_planes[c] = frontend + push.plane_float_offset;
      master->amd_vulkan_plane_stride[c] = push.plane_stride;
      master->amd_vulkan_dc_coefficients[c] = frontend + push.dc_float_offset;
      memcpy(frontend + push.quant_mul_float_offset, master->quant_mul[c],
             DCTSIZE2 * sizeof(float));
      memcpy(frontend + push.zero_bias_offset_float_offset,
             master->zero_bias_offset[c], DCTSIZE2 * sizeof(float));
      memcpy(frontend + push.zero_bias_mul_float_offset,
             master->zero_bias_mul[c], DCTSIZE2 * sizeof(float));
    }
    master->amd_vulkan_quant_field = frontend + quant_field_offset;
    master->amd_vulkan_trace = TraceEnabled();
    slot->cinfo = cinfo;
    slot->frontend_prepared = true;
    slot->ready = false;
    if (TraceEnabled()) {
      fprintf(stderr,
              "jpegli amd-vulkan: reserved %.2f MiB direct front-end input "
              "for %zu blocks\n",
              total_floats * sizeof(float) / 1048576.0, image_blocks);
    }
    return true;
  }

  bool SubmitImage(j_compress_ptr cinfo) {
    if (cinfo == nullptr || !cinfo->progressive_mode ||
        cinfo->master == nullptr)
      return false;
    PipelineSlot* slot = FindSlot(cinfo);
    const bool frontend = slot != nullptr && slot->frontend_prepared;
    if (slot != nullptr && (slot->in_flight || slot->ready)) return true;
    if (slot != nullptr && !frontend) return true;
    if (EnvironmentDisablesGpu()) return false;
    size_t image_blocks = 0;
    for (int c = 0; c < cinfo->num_components; ++c) {
      const jpeg_component_info& comp = cinfo->comp_info[c];
      image_blocks +=
          static_cast<size_t>(comp.width_in_blocks) * comp.height_in_blocks;
    }
    if (!EnvironmentForcesGpu() && image_blocks < kAutoMinComponentBlocks) {
      Trace("image is below the latency crossover; using CPU tokenizer");
      return false;
    }
    if (!frontend) {
      if (!Initialize()) return false;
      slot = FindFreeSlot();
      if (slot == nullptr) {
        Trace("both pipeline slots are occupied; using CPU tokenizer");
        return false;
      }
    }
    const auto upload_start = std::chrono::steady_clock::now();
    if (cinfo->num_components >
        static_cast<int>(slot->component_word_offsets.size())) {
      return false;
    }

    size_t total_words = 0;
    if (!frontend) {
      for (int c = 0; c < cinfo->num_components; ++c) {
        const jpeg_component_info& comp = cinfo->comp_info[c];
        const size_t num_blocks =
            static_cast<size_t>(comp.width_in_blocks) * comp.height_in_blocks;
        if (num_blocks > (std::numeric_limits<size_t>::max() - total_words) /
                             kCoefficientWordsPerBlock) {
          return false;
        }
        slot->component_word_offsets[c] = total_words;
        slot->component_num_blocks[c] = num_blocks;
        total_words += num_blocks * kCoefficientWordsPerBlock;
      }
      if (total_words > std::numeric_limits<uint32_t>::max() ||
          !EnsureBuffer(slot, &slot->coefficient_buffer,
                        total_words * sizeof(uint32_t))) {
        return false;
      }

      static_assert(sizeof(coeff_t) == sizeof(JCOEF),
                    "GPU coefficient ABI must match JBLOCK storage");
      auto* destination =
          static_cast<uint8_t*>(slot->coefficient_buffer.mapped);
      jpeg_comp_master* master = cinfo->master;
      for (int c = 0; c < cinfo->num_components; ++c) {
        const jpeg_component_info& comp = cinfo->comp_info[c];
        uint8_t* component_destination =
            destination + slot->component_word_offsets[c] * sizeof(uint32_t);
        const size_t row_bytes = static_cast<size_t>(comp.width_in_blocks) *
                                 DCTSIZE2 * sizeof(JCOEF);
        for (JDIMENSION by = 0; by < comp.height_in_blocks; ++by) {
          JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(
              reinterpret_cast<j_common_ptr>(cinfo), master->coeff_buffers[c],
              by, 1, FALSE);
          memcpy(component_destination + static_cast<size_t>(by) * row_bytes,
                 &blocks[0][0][0], row_bytes);
        }
      }
    } else {
      for (int c = 0; c < cinfo->num_components; ++c) {
        total_words +=
            slot->component_num_blocks[c] * kCoefficientWordsPerBlock;
      }
    }
    slot->cinfo = cinfo;
    slot->ready = false;
    if (TraceEnabled()) {
      if (frontend) {
        fprintf(stderr,
                "jpegli amd-vulkan: frontend CPU plane capture %.3f ms, DC "
                "recurrence %.3f ms\n",
                cinfo->master->amd_vulkan_plane_capture_ns / 1000000.0,
                cinfo->master->amd_vulkan_dc_ns / 1000000.0);
      }
      const double milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - upload_start)
              .count();
      fprintf(stderr,
              "jpegli amd-vulkan: %s %zu coefficient words in %.3f ms\n",
              frontend ? "prepared front-end for" : "uploaded", total_words,
              milliseconds);
    }
    if (!DispatchAllACScans(cinfo, slot)) {
      // A failure can occur after the fence has been reset. Disable this
      // thread's backend so a later image cannot reuse an unsignaled fence.
      // The encoder transparently recomputes this image on the CPU.
      Trace("AC batch failed; disabling AMD Vulkan tokenizer for this thread");
      available_ = false;
      ReleaseSlot(slot);
      return false;
    }
    return true;
  }

  bool BeginImage(j_compress_ptr cinfo) {
    active_slot_ = nullptr;
    PipelineSlot* slot = FindSlot(cinfo);
    if (slot == nullptr ||
        (slot->frontend_prepared && !slot->in_flight && !slot->ready)) {
      if (!SubmitImage(cinfo)) return false;
      slot = FindSlot(cinfo);
    }
    if (slot == nullptr) return false;
    if (!WaitForSlot(slot)) {
      available_ = false;
      ReleaseSlot(slot);
      return false;
    }
    if (slot->frontend_prepared && EnvironmentVerifiesFrontend()) {
      ComputeAmdVulkanFrontendCoefficients(cinfo);
      size_t mismatches = 0;
      size_t dc_mismatches = 0;
      size_t ac_mismatches = 0;
      const int16_t* gpu_coefficients =
          static_cast<const int16_t*>(slot->coefficient_buffer.mapped);
      for (int c = 0; c < cinfo->num_components; ++c) {
        const jpeg_component_info& comp = cinfo->comp_info[c];
        const size_t component_half_offset =
            slot->component_word_offsets[c] * 2;
        for (JDIMENSION by = 0; by < comp.height_in_blocks; ++by) {
          JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(
              reinterpret_cast<j_common_ptr>(cinfo),
              cinfo->master->coeff_buffers[c], by, 1, FALSE);
          for (JDIMENSION bx = 0; bx < comp.width_in_blocks; ++bx) {
            const size_t block =
                static_cast<size_t>(by) * comp.width_in_blocks + bx;
            for (size_t k = 0; k < DCTSIZE2; ++k) {
              const int cpu = blocks[0][bx][k];
              const int gpu = gpu_coefficients[component_half_offset +
                                               block * DCTSIZE2 + k];
              if (cpu == gpu) continue;
              if (mismatches < 16) {
                fprintf(stderr,
                        "jpegli amd-vulkan: frontend mismatch c=%d block=%zu "
                        "k=%zu cpu=%d gpu=%d\n",
                        c, block, k, cpu, gpu);
              }
              ++mismatches;
              k == 0 ? ++dc_mismatches : ++ac_mismatches;
            }
          }
        }
      }
      fprintf(stderr,
              "jpegli amd-vulkan: frontend verification %zu mismatches "
              "(%zu DC, %zu AC)\n",
              mismatches, dc_mismatches, ac_mismatches);
    }
    active_slot_ = slot;
    return true;
  }

  void EndImage(j_compress_ptr cinfo) {
    PipelineSlot* slot = FindSlot(cinfo);
    if (slot == nullptr) return;
    if (slot->in_flight && !WaitForSlot(slot)) available_ = false;
    if (active_slot_ == slot) active_slot_ = nullptr;
    ReleaseSlot(slot);
  }

  bool TokenizeInitialAC(j_compress_ptr cinfo, int scan_index, int context,
                         AmdVulkanACResult* result) {
    return TokenizeAC(cinfo, scan_index, context, 0, result);
  }

  bool TokenizeRefinementAC(j_compress_ptr cinfo, int scan_index,
                            AmdVulkanACResult* result) {
    return TokenizeAC(cinfo, scan_index, 0, 1, result);
  }

  bool TokenizeDC(j_compress_ptr cinfo, int scan_index,
                  AmdVulkanACResult* result) {
    if (active_slot_ == nullptr || active_slot_->cinfo != cinfo ||
        !active_slot_->ready || result == nullptr || scan_index < 0 ||
        scan_index >= static_cast<int>(active_slot_->cached_results.size())) {
      return false;
    }
    const jpeg_scan_info& scan = cinfo->scan_info[scan_index];
    const AmdVulkanACResult& cached = active_slot_->cached_results[scan_index];
    if (scan.comps_in_scan != 1 || scan.Ss != 0 || scan.Se != 0 ||
        scan.Ah != 0 || cached.compact_tokens == nullptr ||
        cached.symbol_histogram == nullptr) {
      return false;
    }
    *result = cached;
    return true;
  }

  bool TokenizeAC(j_compress_ptr cinfo, int scan_index, int context,
                  uint32_t mode, AmdVulkanACResult* result) {
    if (active_slot_ == nullptr || active_slot_->cinfo != cinfo ||
        !active_slot_->ready || result == nullptr) {
      return false;
    }
    const jpeg_scan_info& scan = cinfo->scan_info[scan_index];
    if (scan.comps_in_scan != 1 || scan.Ss <= 0 ||
        (mode == 0 && scan.Ah != 0) || (mode == 1 && scan.Ah == 0) ||
        mode > 1 || scan.Se < scan.Ss || scan.Se >= DCTSIZE2 || scan.Al < 0 ||
        scan.Al > 15 || context < 0 || context > 255) {
      return false;
    }
    if (scan_index >= 0 &&
        scan_index < static_cast<int>(active_slot_->cached_results.size()) &&
        active_slot_->cached_results[scan_index].words != nullptr) {
      *result = active_slot_->cached_results[scan_index];
      return true;
    }
    return false;
  }

  bool DispatchAllACScans(j_compress_ptr cinfo, PipelineSlot* slot) {
    struct ScanDispatch {
      int scan_index;
      int component;
      uint32_t width;
      uint32_t height;
      bool compact;
      PushConstants push;
    };
    struct DCDispatch {
      int scan_index;
      int component;
      uint32_t header_word_offset;
      uint32_t token_word_offset;
      uint32_t histogram_word_offset;
      uint32_t successive_low;
    };
    std::vector<ScanDispatch> dispatches;
    std::vector<DCDispatch> dc_dispatches;
    slot->cached_results.assign(cinfo->num_scans, {});
    size_t total_output_words = 0;
    size_t total_compact_words = 0;
    size_t total_wave64_workgroups = 0;
    bool has_ac_compact = false;
    const bool throughput_fusion = EnvironmentEnablesThroughputFusion();
    const bool latency_compact = EnvironmentEnablesLatencyCompact();
    const bool gpu_dc =
        EnvironmentEnablesDC() || cinfo->master->amd_vulkan_frontend;
    const auto checked_multiply = [](size_t a, size_t b, size_t* product) {
      if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
      *product = a * b;
      return true;
    };
    const auto reserve_compact_words = [&](size_t words, uint32_t* offset) {
      if (total_compact_words > std::numeric_limits<uint32_t>::max() ||
          words > std::numeric_limits<uint32_t>::max() - total_compact_words) {
        return false;
      }
      *offset = static_cast<uint32_t>(total_compact_words);
      total_compact_words += words;
      return true;
    };
    for (int scan_index = 0; scan_index < cinfo->num_scans; ++scan_index) {
      const jpeg_scan_info& scan = cinfo->scan_info[scan_index];
      if (gpu_dc && scan.Ss == 0 && scan.Se == 0 && scan.Ah == 0 &&
          scan.comps_in_scan == 1 &&
          cinfo->master->scan_token_info[scan_index].restart_interval == 0) {
        const int component = scan.component_index[0];
        const size_t num_blocks = slot->component_num_blocks[component];
        uint32_t header_word_offset = 0;
        uint32_t token_word_offset = 0;
        uint32_t histogram_word_offset = 0;
        if (num_blocks == 0 ||
            num_blocks > std::numeric_limits<uint32_t>::max() ||
            !reserve_compact_words(8, &header_word_offset) ||
            !reserve_compact_words(num_blocks, &token_word_offset) ||
            !reserve_compact_words(256, &histogram_word_offset)) {
          return false;
        }
        dc_dispatches.push_back({scan_index, component, header_word_offset,
                                 token_word_offset, histogram_word_offset,
                                 static_cast<uint32_t>(scan.Al)});
        slot->cached_results[scan_index].num_blocks = num_blocks;
      }
      if (scan.Ss <= 0 || scan.comps_in_scan != 1 || scan.Se < scan.Ss ||
          scan.Se >= DCTSIZE2 || scan.Al < 0 || scan.Al > 15) {
        continue;
      }
      const int component = scan.component_index[0];
      const size_t num_blocks = slot->component_num_blocks[component];
      const size_t stride_words = 2 + scan.Se - scan.Ss + 1;
      if (num_blocks == 0 ||
          num_blocks > std::numeric_limits<uint32_t>::max() ||
          total_output_words > std::numeric_limits<uint32_t>::max() ||
          num_blocks >
              (std::numeric_limits<uint32_t>::max() - total_output_words) /
                  stride_words) {
        return false;
      }
      const uint32_t width = static_cast<uint32_t>(std::min<size_t>(
          num_blocks, properties_.limits.maxComputeWorkGroupCount[0]));
      const uint32_t height =
          static_cast<uint32_t>((num_blocks + width - 1) / width);
      if (height > properties_.limits.maxComputeWorkGroupCount[1]) {
        return false;
      }
      const uint32_t mode = scan.Ah == 0 ? 0 : 1;
      PushConstants push = {};
      push.coefficient_word_offset =
          static_cast<uint32_t>(slot->component_word_offsets[component]);
      push.num_blocks = static_cast<uint32_t>(num_blocks);
      push.spectral_start = static_cast<uint32_t>(scan.Ss);
      push.spectral_end = static_cast<uint32_t>(scan.Se);
      push.successive_low = static_cast<uint32_t>(scan.Al);
      push.output_stride_words = static_cast<uint32_t>(stride_words);
      push.context =
          static_cast<uint32_t>(cinfo->master->ac_ctx_offset[scan_index]);
      push.dispatch_width = width;
      push.mode = mode;
      push.output_word_offset = static_cast<uint32_t>(total_output_words);

      const bool compact =
          EnvironmentEnablesLatencyCompactMode(mode) &&
          cinfo->master->scan_token_info[scan_index].restart_interval == 0;
      has_ac_compact |= compact;
      if (compact) {
        const size_t band_size = static_cast<size_t>(scan.Se - scan.Ss + 1);
        size_t token_capacity = 0;
        size_t refbit_capacity = 0;
        size_t eobrun_capacity = 0;
        size_t scratch_words = 0;
        if (!reserve_compact_words(8, &push.compact_header_word_offset)) {
          return false;
        }
        if (mode == 0) {
          const size_t compact_groups =
              (num_blocks + kCompactBlocksPerGroup - 1) /
              kCompactBlocksPerGroup;
          size_t block_summary_words = 0;
          size_t group_summary_words = 0;
          if (!checked_multiply(num_blocks, band_size + 1, &token_capacity) ||
              !checked_multiply(num_blocks, size_t{3}, &block_summary_words) ||
              !checked_multiply(compact_groups, size_t{8},
                                &group_summary_words) ||
              block_summary_words >
                  std::numeric_limits<size_t>::max() - group_summary_words) {
            return false;
          }
          scratch_words = block_summary_words + group_summary_words;
          push.compact_scratch_stride_words = 3;
        } else {
          const size_t max_tokens_per_block =
              band_size + (band_size + 15) / 16 + 1;
          const size_t compact_groups =
              (num_blocks + kCompactBlocksPerGroup - 1) /
              kCompactBlocksPerGroup;
          size_t summary_words = 0;
          size_t group_summary_words = 0;
          if (!checked_multiply(num_blocks, max_tokens_per_block,
                                &token_capacity) ||
              !checked_multiply(num_blocks, band_size, &refbit_capacity) ||
              !checked_multiply(num_blocks, size_t{9}, &summary_words) ||
              !checked_multiply(compact_groups, size_t{12},
                                &group_summary_words) ||
              token_capacity >
                  std::numeric_limits<size_t>::max() - summary_words ||
              group_summary_words > std::numeric_limits<size_t>::max() -
                                        summary_words - token_capacity) {
            return false;
          }
          scratch_words = summary_words + token_capacity + group_summary_words;
          eobrun_capacity = num_blocks / 2 + 1;
          if (max_tokens_per_block > std::numeric_limits<uint32_t>::max()) {
            return false;
          }
          push.compact_scratch_stride_words =
              static_cast<uint32_t>(max_tokens_per_block);
        }
        if (token_capacity > std::numeric_limits<uint32_t>::max() ||
            refbit_capacity > std::numeric_limits<uint32_t>::max() ||
            !reserve_compact_words(token_capacity,
                                   &push.compact_token_word_offset) ||
            !reserve_compact_words(refbit_capacity,
                                   &push.compact_refbit_word_offset) ||
            !reserve_compact_words(eobrun_capacity,
                                   &push.compact_eobrun_word_offset) ||
            !reserve_compact_words(scratch_words,
                                   &push.compact_scratch_word_offset)) {
          return false;
        }
        push.compact_token_capacity = static_cast<uint32_t>(token_capacity);
        push.compact_refbit_capacity = static_cast<uint32_t>(refbit_capacity);
      }
      dispatches.push_back(
          {scan_index, component, width, height, compact, push});
      slot->cached_results[scan_index].stride_words = stride_words;
      slot->cached_results[scan_index].num_blocks = num_blocks;
      total_output_words += num_blocks * stride_words;
      total_wave64_workgroups += num_blocks;
    }
    struct FusedComponentDispatch {
      uint32_t width;
      uint32_t height;
      PushConstants push;
    };
    std::vector<std::array<uint32_t, 20>> scan_parameters;
    std::vector<FusedComponentDispatch> fused_dispatches;
    size_t fused_wave64_workgroups = 0;
    if (throughput_fusion || latency_compact || gpu_dc) {
      for (int component = 0; component < cinfo->num_components; ++component) {
        const size_t parameter_begin = scan_parameters.size();
        const ScanDispatch* first = nullptr;
        for (const ScanDispatch& dispatch : dispatches) {
          if (dispatch.component != component) continue;
          if (first == nullptr) first = &dispatch;
          scan_parameters.push_back({
              dispatch.push.num_blocks,
              dispatch.push.spectral_start,
              dispatch.push.spectral_end,
              dispatch.push.successive_low,
              dispatch.push.output_stride_words,
              dispatch.push.context,
              dispatch.push.mode,
              dispatch.push.output_word_offset,
              static_cast<uint32_t>(dispatch.compact),
              dispatch.push.compact_header_word_offset,
              dispatch.push.compact_token_word_offset,
              dispatch.push.compact_refbit_word_offset,
              dispatch.push.compact_eobrun_word_offset,
              dispatch.push.compact_scratch_word_offset,
              dispatch.push.compact_scratch_stride_words,
              dispatch.push.compact_token_capacity,
              dispatch.push.compact_refbit_capacity,
              0,
              0,
              0,
          });
        }
        if (first == nullptr) continue;
        const size_t parameter_count = scan_parameters.size() - parameter_begin;
        if (parameter_begin > std::numeric_limits<uint32_t>::max() ||
            parameter_count > std::numeric_limits<uint32_t>::max()) {
          return false;
        }
        PushConstants push = {};
        push.coefficient_word_offset =
            static_cast<uint32_t>(slot->component_word_offsets[component]);
        push.num_blocks = first->push.num_blocks;
        push.dispatch_width = first->width;
        push.mode = 4;
        push.scan_parameter_offset = static_cast<uint32_t>(parameter_begin);
        push.scan_parameter_count = static_cast<uint32_t>(parameter_count);
        for (const DCDispatch& dc : dc_dispatches) {
          if (dc.component != component) continue;
          push.context = static_cast<uint32_t>(component);
          push.successive_low = dc.successive_low;
          push.compact_header_word_offset = dc.header_word_offset;
          push.compact_token_word_offset = dc.token_word_offset;
          push.compact_refbit_word_offset = dc.histogram_word_offset;
          push.compact_token_capacity = 1;
          break;
        }
        fused_dispatches.push_back({first->width, first->height, push});
        fused_wave64_workgroups += first->push.num_blocks;
      }
    }
    if (dispatches.empty()) {
      slot->ready = true;
      return true;
    }
    if (!EnsureBuffer(slot, &slot->descriptor_buffer,
                      total_output_words * sizeof(uint32_t)) ||
        !EnsureBuffer(
            slot, &slot->compact_buffer,
            std::max<size_t>(total_compact_words, 1) * sizeof(uint32_t)) ||
        !EnsureBuffer(slot, &slot->scan_parameter_buffer,
                      std::max<size_t>(scan_parameters.size() * 20, 1) *
                          sizeof(uint32_t)) ||
        !EnsureBuffer(slot, &slot->frontend_buffer, sizeof(uint32_t)) ||
        !UpdateDescriptors(slot)) {
      slot->cached_results.clear();
      return false;
    }
    if (!scan_parameters.empty()) {
      memcpy(slot->scan_parameter_buffer.mapped, scan_parameters.data(),
             scan_parameters.size() * sizeof(scan_parameters[0]));
    }
    const uint32_t* output =
        static_cast<const uint32_t*>(slot->descriptor_buffer.mapped);
    const uint32_t* compact_output =
        static_cast<const uint32_t*>(slot->compact_buffer.mapped);
    for (const ScanDispatch& dispatch : dispatches) {
      AmdVulkanACResult& result = slot->cached_results[dispatch.scan_index];
      result.words = output + dispatch.push.output_word_offset;
      if (dispatch.compact) {
        result.compact_header =
            compact_output + dispatch.push.compact_header_word_offset;
        result.compact_tokens =
            compact_output + dispatch.push.compact_token_word_offset;
        if (dispatch.push.mode == 1) {
          result.compact_refbits =
              compact_output + dispatch.push.compact_refbit_word_offset;
          result.compact_eobruns =
              compact_output + dispatch.push.compact_eobrun_word_offset;
        }
      }
    }
    for (const DCDispatch& dc : dc_dispatches) {
      AmdVulkanACResult& result = slot->cached_results[dc.scan_index];
      result.compact_header = compact_output + dc.header_word_offset;
      result.compact_tokens = compact_output + dc.token_word_offset;
      result.symbol_histogram = compact_output + dc.histogram_word_offset;
      memset(static_cast<uint32_t*>(slot->compact_buffer.mapped) +
                 dc.histogram_word_offset,
             0, 256 * sizeof(uint32_t));
    }

    if (vkWaitForFences(device_, 1, &slot->fence, VK_TRUE, UINT64_MAX) !=
            VK_SUCCESS ||
        vkResetFences(device_, 1, &slot->fence) != VK_SUCCESS ||
        vkResetCommandBuffer(slot->command_buffer, 0) != VK_SUCCESS) {
      slot->cached_results.clear();
      return false;
    }
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->command_buffer, &begin_info) != VK_SUCCESS) {
      slot->cached_results.clear();
      return false;
    }
    if (slot->query_pool != VK_NULL_HANDLE) {
      vkCmdResetQueryPool(slot->command_buffer, slot->query_pool, 0,
                          kTimestampQueries);
      vkCmdWriteTimestamp(slot->command_buffer,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, slot->query_pool,
                          0);
    }
    VkMemoryBarrier host_to_compute = {};
    host_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_to_compute.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host_to_compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(slot->command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &host_to_compute, 0, nullptr, 0, nullptr);
    vkCmdBindDescriptorSets(slot->command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0,
                            1, &slot->descriptor_set, 0, nullptr);
    if (slot->frontend_prepared) {
      vkCmdBindPipeline(slot->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        frontend_pipeline_);
      for (int c = 0; c < cinfo->num_components; ++c) {
        FrontendPushConstants push = slot->frontend_push[c];
        const uint32_t groups = (push.num_blocks + 7u) / 8u;
        const uint32_t height =
            (groups + push.dispatch_width - 1) / push.dispatch_width;
        vkCmdPushConstants(slot->command_buffer, pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(slot->command_buffer, push.dispatch_width, height, 1);
      }
      VkMemoryBarrier frontend_to_descriptor = {};
      frontend_to_descriptor.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      frontend_to_descriptor.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      frontend_to_descriptor.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(slot->command_buffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &frontend_to_descriptor, 0, nullptr, 0, nullptr);
    }
    if (slot->query_pool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(slot->command_buffer,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          slot->query_pool, 1);
    }
    vkCmdBindPipeline(slot->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_);
    if (!fused_dispatches.empty()) {
      for (const FusedComponentDispatch& dispatch : fused_dispatches) {
        vkCmdPushConstants(slot->command_buffer, pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(dispatch.push), &dispatch.push);
        vkCmdDispatch(slot->command_buffer, dispatch.width, dispatch.height, 1);
      }
    } else {
      for (const ScanDispatch& dispatch : dispatches) {
        vkCmdPushConstants(slot->command_buffer, pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(dispatch.push), &dispatch.push);
        vkCmdDispatch(slot->command_buffer, dispatch.width, dispatch.height, 1);
      }
    }
    if (slot->query_pool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(slot->command_buffer,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          slot->query_pool, 2);
    }
    if (has_ac_compact) {
      VkMemoryBarrier descriptor_to_compact = {};
      descriptor_to_compact.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      descriptor_to_compact.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      descriptor_to_compact.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(slot->command_buffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &descriptor_to_compact, 0, nullptr, 0, nullptr);

      if (slot->query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(slot->command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            slot->query_pool, 3);
      }

      for (const ScanDispatch& dispatch : dispatches) {
        if (!dispatch.compact) continue;
        PushConstants compact_push = dispatch.push;
        compact_push.mode = compact_push.mode == 0 ? 2 : 6;
        vkCmdPushConstants(slot->command_buffer, pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compact_push),
                           &compact_push);
        const uint32_t state_groups = (dispatch.push.num_blocks +
                                       kCompactBlocksPerGroup * kWaveSize - 1) /
                                      (kCompactBlocksPerGroup * kWaveSize);
        vkCmdDispatch(slot->command_buffer, state_groups, 1, 1);
      }
      if (slot->query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(slot->command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            slot->query_pool, 4);
      }

      VkMemoryBarrier state_to_global = {};
      state_to_global.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      state_to_global.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      state_to_global.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(slot->command_buffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &state_to_global, 0, nullptr, 0, nullptr);

      for (const ScanDispatch& dispatch : dispatches) {
        if (!dispatch.compact) continue;
        PushConstants compact_push = dispatch.push;
        compact_push.mode = compact_push.mode == 0 ? 10 : 12;
        vkCmdPushConstants(slot->command_buffer, pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compact_push),
                           &compact_push);
        vkCmdDispatch(slot->command_buffer, 1, 1, 1);
      }
      if (slot->query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(slot->command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            slot->query_pool, 5);
      }

      VkMemoryBarrier global_to_finalize = {};
      global_to_finalize.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      global_to_finalize.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      global_to_finalize.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(slot->command_buffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &global_to_finalize, 0, nullptr, 0, nullptr);

      for (const ScanDispatch& dispatch : dispatches) {
        if (!dispatch.compact) continue;
        PushConstants compact_push = dispatch.push;
        compact_push.mode = compact_push.mode == 0 ? 11 : 13;
        vkCmdPushConstants(slot->command_buffer, pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compact_push),
                           &compact_push);
        const uint32_t state_groups =
            (compact_push.num_blocks + kCompactBlocksPerGroup * kWaveSize - 1) /
            (kCompactBlocksPerGroup * kWaveSize);
        vkCmdDispatch(slot->command_buffer, state_groups, 1, 1);
      }
      if (slot->query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(slot->command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            slot->query_pool, 6);
      }

      VkMemoryBarrier finalize_to_scatter = {};
      finalize_to_scatter.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      finalize_to_scatter.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      finalize_to_scatter.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(slot->command_buffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &finalize_to_scatter, 0, nullptr, 0, nullptr);

      for (const FusedComponentDispatch& dispatch : fused_dispatches) {
        PushConstants compact_push = dispatch.push;
        compact_push.mode = 9;
        const uint32_t compact_groups =
            (compact_push.num_blocks + kWaveSize - 1) / kWaveSize;
        const uint32_t compact_width = std::min(
            compact_groups, properties_.limits.maxComputeWorkGroupCount[0]);
        const uint32_t compact_height =
            (compact_groups + compact_width - 1) / compact_width;
        compact_push.dispatch_width = compact_width;
        vkCmdPushConstants(slot->command_buffer, pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compact_push),
                           &compact_push);
        vkCmdDispatch(slot->command_buffer, compact_width, compact_height, 1);
      }
      if (slot->query_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(slot->command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            slot->query_pool, 7);
      }
    } else if (slot->query_pool != VK_NULL_HANDLE) {
      for (uint32_t query = 3; query <= 7; ++query) {
        vkCmdWriteTimestamp(slot->command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            slot->query_pool, query);
      }
    }
    VkMemoryBarrier compute_to_host = {};
    compute_to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    compute_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(slot->command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &compute_to_host, 0,
                         nullptr, 0, nullptr);
    if (slot->query_pool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(slot->command_buffer,
                          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          slot->query_pool, 8);
    }
    if (vkEndCommandBuffer(slot->command_buffer) != VK_SUCCESS) {
      slot->cached_results.clear();
      return false;
    }
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot->command_buffer;
    slot->submit_time = std::chrono::steady_clock::now();
    if (vkQueueSubmit(queue_, 1, &submit, slot->fence) != VK_SUCCESS) {
      slot->cached_results.clear();
      return false;
    }
    slot->in_flight = true;
    if (TraceEnabled()) {
      fprintf(stderr,
              "jpegli amd-vulkan: submitted %zu AC scans, %zu wave64 "
              "workgroups (%zu descriptor, %zu compact words, fused=%s) "
              "asynchronously\n",
              dispatches.size(),
              fused_dispatches.empty() ? total_wave64_workgroups
                                       : fused_wave64_workgroups,
              total_output_words, total_compact_words,
              fused_dispatches.empty() ? "no" : "yes");
    }
    return true;
  }

 private:
  PipelineSlot* FindSlot(j_compress_ptr cinfo) {
    for (PipelineSlot& slot : slots_) {
      if (slot.cinfo == cinfo) return &slot;
    }
    return nullptr;
  }

  PipelineSlot* FindFreeSlot() {
    for (PipelineSlot& slot : slots_) {
      if (slot.cinfo == nullptr) return &slot;
    }
    return nullptr;
  }

  size_t SlotIndex(const PipelineSlot* slot) const {
    return static_cast<size_t>(slot - slots_.data());
  }

  bool WaitForSlot(PipelineSlot* slot) {
    if (slot->ready) return true;
    if (!slot->in_flight) return false;
    const auto wait_start = std::chrono::steady_clock::now();
    if (vkWaitForFences(device_, 1, &slot->fence, VK_TRUE, UINT64_MAX) !=
        VK_SUCCESS) {
      return false;
    }
    slot->in_flight = false;
    slot->ready = true;
    if (TraceEnabled()) {
      const auto ready_time = std::chrono::steady_clock::now();
      const double wait_ms =
          std::chrono::duration<double, std::milli>(ready_time - wait_start)
              .count();
      const double submit_to_ready_ms =
          std::chrono::duration<double, std::milli>(ready_time -
                                                    slot->submit_time)
              .count();
      double gpu_ms = -1.0;
      std::array<uint64_t, kTimestampQueries> timestamps = {};
      if (slot->query_pool != VK_NULL_HANDLE && timestamp_valid_bits_ != 0 &&
          vkGetQueryPoolResults(device_, slot->query_pool, 0, kTimestampQueries,
                                sizeof(timestamps), timestamps.data(),
                                sizeof(timestamps[0]),
                                VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        const uint64_t mask = timestamp_valid_bits_ >= 64
                                  ? std::numeric_limits<uint64_t>::max()
                                  : (uint64_t{1} << timestamp_valid_bits_) - 1;
        const uint64_t ticks = (timestamps[8] - timestamps[0]) & mask;
        gpu_ms = static_cast<double>(ticks) *
                 properties_.limits.timestampPeriod / 1000000.0;
        fprintf(stderr,
                "jpegli amd-vulkan: gpu stages frontend %.3f, descriptor "
                "%.3f, prepare %.3f, group %.3f, global %.3f, finalize "
                "%.3f, scatter %.3f ms\n",
                static_cast<double>((timestamps[1] - timestamps[0]) & mask) *
                    properties_.limits.timestampPeriod / 1000000.0,
                static_cast<double>((timestamps[2] - timestamps[1]) & mask) *
                    properties_.limits.timestampPeriod / 1000000.0,
                static_cast<double>((timestamps[3] - timestamps[2]) & mask) *
                    properties_.limits.timestampPeriod / 1000000.0,
                static_cast<double>((timestamps[4] - timestamps[3]) & mask) *
                    properties_.limits.timestampPeriod / 1000000.0,
                static_cast<double>((timestamps[5] - timestamps[4]) & mask) *
                    properties_.limits.timestampPeriod / 1000000.0,
                static_cast<double>((timestamps[6] - timestamps[5]) & mask) *
                    properties_.limits.timestampPeriod / 1000000.0,
                static_cast<double>((timestamps[7] - timestamps[6]) & mask) *
                    properties_.limits.timestampPeriod / 1000000.0);
      }
      fprintf(stderr,
              "jpegli amd-vulkan: slot %zu ready, gpu %.3f ms, wait %.3f "
              "ms, submit-to-ready %.3f ms\n",
              SlotIndex(slot), gpu_ms, wait_ms, submit_to_ready_ms);
    }
    return true;
  }

  void ReleaseSlot(PipelineSlot* slot) {
    slot->cinfo = nullptr;
    slot->in_flight = false;
    slot->ready = false;
    slot->frontend_prepared = false;
    slot->cached_results.clear();
  }

  bool Initialize() {
    if (initialization_attempted_) return available_;
    initialization_attempted_ = true;

    VkApplicationInfo application = {};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "jpegli-amd-progressive";
    application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application.pEngineName = "jpegli-direct-amd-vulkan";
    application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    if (vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS) {
      Trace("Vulkan instance creation failed; using CPU tokenizer");
      return false;
    }

    uint32_t physical_device_count = 0;
    if (vkEnumeratePhysicalDevices(instance_, &physical_device_count,
                                   nullptr) != VK_SUCCESS ||
        physical_device_count == 0) {
      Trace("no Vulkan physical devices; using CPU tokenizer");
      ShutdownObjects();
      return false;
    }
    std::vector<VkPhysicalDevice> devices(physical_device_count);
    if (vkEnumeratePhysicalDevices(instance_, &physical_device_count,
                                   devices.data()) != VK_SUCCESS) {
      ShutdownObjects();
      return false;
    }

    for (VkPhysicalDevice candidate : devices) {
      VkPhysicalDeviceProperties candidate_properties;
      vkGetPhysicalDeviceProperties(candidate, &candidate_properties);
      if (candidate_properties.vendorID != kAmdVendorId ||
          candidate_properties.deviceType !=
              VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        continue;
      }

      VkPhysicalDeviceVulkan13Features features13 = {};
      features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
      VkPhysicalDeviceFeatures2 features2 = {};
      features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      features2.pNext = &features13;
      vkGetPhysicalDeviceFeatures2(candidate, &features2);

      VkPhysicalDeviceVulkan13Properties properties13 = {};
      properties13.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
      VkPhysicalDeviceVulkan11Properties properties11 = {};
      properties11.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
      properties11.pNext = &properties13;
      VkPhysicalDeviceProperties2 properties2 = {};
      properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      properties2.pNext = &properties11;
      vkGetPhysicalDeviceProperties2(candidate, &properties2);
      const VkSubgroupFeatureFlags required_subgroup_operations =
          VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
          VK_SUBGROUP_FEATURE_BALLOT_BIT;
      if (!features13.subgroupSizeControl ||
          properties13.minSubgroupSize > kWaveSize ||
          properties13.maxSubgroupSize < kWaveSize ||
          !(properties13.requiredSubgroupSizeStages &
            VK_SHADER_STAGE_COMPUTE_BIT) ||
          !(properties11.subgroupSupportedStages &
            VK_SHADER_STAGE_COMPUTE_BIT) ||
          (properties11.subgroupSupportedOperations &
           required_subgroup_operations) != required_subgroup_operations) {
        continue;
      }

      uint32_t family_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count,
                                               nullptr);
      std::vector<VkQueueFamilyProperties> families(family_count);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count,
                                               families.data());
      uint32_t selected_family = UINT32_MAX;
      for (uint32_t i = 0; i < family_count; ++i) {
        if (!(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
        if (selected_family == UINT32_MAX ||
            !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
          selected_family = i;
        }
        if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
      }
      if (selected_family == UINT32_MAX) continue;

      physical_device_ = candidate;
      properties_ = candidate_properties;
      queue_family_ = selected_family;
      timestamp_valid_bits_ = families[selected_family].timestampValidBits;
      break;
    }
    if (physical_device_ == VK_NULL_HANDLE) {
      Trace(
          "no compatible AMD unified-memory wave64 device; using CPU "
          "tokenizer");
      ShutdownObjects();
      return false;
    }

    const std::string device_name(properties_.deviceName);
    if (device_name.find("8060S") != std::string::npos) {
      tuning_profile_ = AmdTuningProfile::kStrixHalo;
    } else if (device_name.find("780M") != std::string::npos) {
      tuning_profile_ = AmdTuningProfile::kPhoenix;
    } else {
      tuning_profile_ = AmdTuningProfile::kOtherUnifiedAmd;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;
    VkPhysicalDeviceVulkan13Features enabled13 = {};
    enabled13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    enabled13.subgroupSizeControl = VK_TRUE;
    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &enabled13;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (vkCreateDevice(physical_device_, &device_info, nullptr, &device_) !=
        VK_SUCCESS) {
      Trace("AMD Vulkan device creation failed; using CPU tokenizer");
      ShutdownObjects();
      return false;
    }
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    if (!CreatePipelineObjects()) {
      ShutdownObjects();
      return false;
    }
    available_ = true;
    if (TraceEnabled()) {
      const char* profile = tuning_profile_ == AmdTuningProfile::kPhoenix
                                ? "Phoenix/gfx1103"
                            : tuning_profile_ == AmdTuningProfile::kStrixHalo
                                ? "Strix Halo/gfx1151"
                                : "unrecognized AMD unified GPU";
      fprintf(stderr, "jpegli amd-vulkan: %s, profile %s\n",
              properties_.deviceName, profile);
    }
    return true;
  }

  bool CreatePipelineObjects() {
    std::array<VkDescriptorSetLayoutBinding, 5> bindings = {};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr,
                                    &descriptor_set_layout_) != VK_SUCCESS) {
      return false;
    }

    VkPushConstantRange push_range = {};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr,
                               &pipeline_layout_) != VK_SUCCESS) {
      return false;
    }

    VkShaderModuleCreateInfo shader_info = {};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = kAmdVulkanProgressiveSpvSize;
    shader_info.pCode =
        reinterpret_cast<const uint32_t*>(kAmdVulkanProgressiveSpv);
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &shader_info, nullptr, &shader) !=
        VK_SUCCESS) {
      return false;
    }
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_info = {};
    subgroup_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroup_info.requiredSubgroupSize = kWaveSize;
    VkPipelineShaderStageCreateInfo stage = {};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.pNext = &subgroup_info;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = pipeline_layout_;
    const VkResult pipeline_result = vkCreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, shader, nullptr);
    if (pipeline_result != VK_SUCCESS) return false;

    shader_info.codeSize = kAmdVulkanFrontendSpvSize;
    shader_info.pCode =
        reinterpret_cast<const uint32_t*>(kAmdVulkanFrontendSpv);
    shader = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &shader_info, nullptr, &shader) !=
        VK_SUCCESS) {
      return false;
    }
    stage.module = shader;
    const VkResult frontend_pipeline_result =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                 nullptr, &frontend_pipeline_);
    vkDestroyShaderModule(device_, shader, nullptr);
    if (frontend_pipeline_result != VK_SUCCESS) return false;

    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 5 * kPipelineSlots;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = kPipelineSlots;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr,
                               &descriptor_pool_) != VK_SUCCESS) {
      return false;
    }
    VkDescriptorSetAllocateInfo set_info = {};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = descriptor_pool_;
    std::array<VkDescriptorSetLayout, kPipelineSlots> set_layouts;
    set_layouts.fill(descriptor_set_layout_);
    std::array<VkDescriptorSet, kPipelineSlots> descriptor_sets = {};
    set_info.descriptorSetCount = kPipelineSlots;
    set_info.pSetLayouts = set_layouts.data();
    if (vkAllocateDescriptorSets(device_, &set_info, descriptor_sets.data()) !=
        VK_SUCCESS) {
      return false;
    }
    for (size_t i = 0; i < kPipelineSlots; ++i) {
      slots_[i].descriptor_set = descriptor_sets[i];
    }

    VkCommandPoolCreateInfo command_pool_info = {};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                              VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    command_pool_info.queueFamilyIndex = queue_family_;
    if (vkCreateCommandPool(device_, &command_pool_info, nullptr,
                            &command_pool_) != VK_SUCCESS) {
      return false;
    }
    VkCommandBufferAllocateInfo command_info = {};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = command_pool_;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = kPipelineSlots;
    std::array<VkCommandBuffer, kPipelineSlots> command_buffers = {};
    if (vkAllocateCommandBuffers(device_, &command_info,
                                 command_buffers.data()) != VK_SUCCESS) {
      return false;
    }
    for (size_t i = 0; i < kPipelineSlots; ++i) {
      slots_[i].command_buffer = command_buffers[i];
    }
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkQueryPoolCreateInfo query_info = {};
    query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = kTimestampQueries;
    for (PipelineSlot& slot : slots_) {
      if (vkCreateFence(device_, &fence_info, nullptr, &slot.fence) !=
          VK_SUCCESS) {
        return false;
      }
      if (timestamp_valid_bits_ != 0 &&
          vkCreateQueryPool(device_, &query_info, nullptr, &slot.query_pool) !=
              VK_SUCCESS) {
        return false;
      }
    }
    return true;
  }

  uint32_t FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred) const {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);
    uint32_t best_index = UINT32_MAX;
    uint32_t best_score = 0;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if (!(type_bits & (1u << i))) continue;
      const VkMemoryPropertyFlags flags =
          memory_properties.memoryTypes[i].propertyFlags;
      if ((flags & required) != required) continue;
      uint32_t score = 1;
      if ((flags & preferred) == preferred) score += 4;
      if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) score += 2;
      if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) score += 1;
      if (score > best_score) {
        best_score = score;
        best_index = i;
      }
    }
    return best_index;
  }

  bool EnsureBuffer(PipelineSlot* slot, MappedBuffer* target,
                    size_t requested_size, bool prefer_device_local = false) {
    if (target->size >= requested_size) return true;
    DestroyBuffer(target);
    VkDeviceSize allocation_size = std::max<size_t>(requested_size, 4096);
    // Geometric growth avoids reallocating the mapped buffers when successive
    // images in an image-generation batch differ slightly in dimensions.
    allocation_size += allocation_size / 2;
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = allocation_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &buffer_info, nullptr, &target->buffer) !=
        VK_SUCCESS) {
      return false;
    }
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device_, target->buffer, &requirements);
    const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // Both deploy targets are UMA. Cached GTT keeps CPU uploads and dense
    // compact-stream consumption cheap while remaining directly GPU-visible.
    const VkMemoryPropertyFlags preferred =
        prefer_device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                            : VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const uint32_t memory_type =
        FindMemoryType(requirements.memoryTypeBits, required, preferred);
    if (memory_type == UINT32_MAX) {
      DestroyBuffer(target);
      return false;
    }
    VkMemoryAllocateInfo allocation = {};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type;
    if (vkAllocateMemory(device_, &allocation, nullptr, &target->memory) !=
            VK_SUCCESS ||
        vkBindBufferMemory(device_, target->buffer, target->memory, 0) !=
            VK_SUCCESS ||
        vkMapMemory(device_, target->memory, 0, allocation_size, 0,
                    &target->mapped) != VK_SUCCESS) {
      DestroyBuffer(target);
      return false;
    }
    target->size = allocation_size;
    slot->descriptors_dirty = true;
    return true;
  }

  bool UpdateDescriptors(PipelineSlot* slot) {
    if (!slot->descriptors_dirty) return true;
    if (slot->coefficient_buffer.buffer == VK_NULL_HANDLE ||
        slot->descriptor_buffer.buffer == VK_NULL_HANDLE ||
        slot->compact_buffer.buffer == VK_NULL_HANDLE ||
        slot->scan_parameter_buffer.buffer == VK_NULL_HANDLE ||
        slot->frontend_buffer.buffer == VK_NULL_HANDLE) {
      return false;
    }
    std::array<VkDescriptorBufferInfo, 5> buffer_info = {};
    buffer_info[0].buffer = slot->coefficient_buffer.buffer;
    buffer_info[0].range = slot->coefficient_buffer.size;
    buffer_info[1].buffer = slot->descriptor_buffer.buffer;
    buffer_info[1].range = slot->descriptor_buffer.size;
    buffer_info[2].buffer = slot->compact_buffer.buffer;
    buffer_info[2].range = slot->compact_buffer.size;
    buffer_info[3].buffer = slot->scan_parameter_buffer.buffer;
    buffer_info[3].range = slot->scan_parameter_buffer.size;
    buffer_info[4].buffer = slot->frontend_buffer.buffer;
    buffer_info[4].range = slot->frontend_buffer.size;
    std::array<VkWriteDescriptorSet, 5> writes = {};
    for (uint32_t i = 0; i < writes.size(); ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = slot->descriptor_set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffer_info[i];
    }
    vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, nullptr);
    slot->descriptors_dirty = false;
    return true;
  }

  void DestroyBuffer(MappedBuffer* target) {
    if (device_ == VK_NULL_HANDLE) return;
    if (target->mapped != nullptr) vkUnmapMemory(device_, target->memory);
    if (target->buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, target->buffer, nullptr);
    }
    if (target->memory != VK_NULL_HANDLE) {
      vkFreeMemory(device_, target->memory, nullptr);
    }
    *target = {};
  }

  void ShutdownObjects() {
    if (device_ != VK_NULL_HANDLE) {
      vkDeviceWaitIdle(device_);
      for (PipelineSlot& slot : slots_) {
        DestroyBuffer(&slot.frontend_buffer);
        DestroyBuffer(&slot.scan_parameter_buffer);
        DestroyBuffer(&slot.compact_buffer);
        DestroyBuffer(&slot.descriptor_buffer);
        DestroyBuffer(&slot.coefficient_buffer);
        if (slot.query_pool != VK_NULL_HANDLE) {
          vkDestroyQueryPool(device_, slot.query_pool, nullptr);
        }
        if (slot.fence != VK_NULL_HANDLE) {
          vkDestroyFence(device_, slot.fence, nullptr);
        }
        slot.query_pool = VK_NULL_HANDLE;
        slot.fence = VK_NULL_HANDLE;
        slot.command_buffer = VK_NULL_HANDLE;
        slot.descriptor_set = VK_NULL_HANDLE;
        ReleaseSlot(&slot);
      }
      if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
      }
      if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
      }
      if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
      }
      if (frontend_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, frontend_pipeline_, nullptr);
      }
      if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
      }
      if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
      }
      vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    device_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
  }

  void Shutdown() {
    active_slot_ = nullptr;
    available_ = false;
    ShutdownObjects();
  }

  bool initialization_attempted_ = false;
  bool available_ = false;
  PipelineSlot* active_slot_ = nullptr;
  std::array<PipelineSlot, kPipelineSlots> slots_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties_ = {};
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = UINT32_MAX;
  uint32_t timestamp_valid_bits_ = 0;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipeline frontend_pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  AmdTuningProfile tuning_profile_ = AmdTuningProfile::kOtherUnifiedAmd;
};

thread_local AmdVulkanProgressiveTokenizer g_amd_vulkan_tokenizer;

}  // namespace

bool AmdVulkanProgressiveSubmit(j_compress_ptr cinfo) {
  return g_amd_vulkan_tokenizer.SubmitImage(cinfo);
}

bool AmdVulkanFrontendPrepare(j_compress_ptr cinfo) {
  return g_amd_vulkan_tokenizer.PrepareFrontend(cinfo);
}

bool AmdVulkanProgressiveBegin(j_compress_ptr cinfo) {
  return g_amd_vulkan_tokenizer.BeginImage(cinfo);
}

void AmdVulkanProgressiveEnd(j_compress_ptr cinfo) {
  g_amd_vulkan_tokenizer.EndImage(cinfo);
}

bool AmdVulkanTokenizeInitialAC(j_compress_ptr cinfo, int scan_index,
                                int context, AmdVulkanACResult* result) {
  return g_amd_vulkan_tokenizer.TokenizeInitialAC(cinfo, scan_index, context,
                                                  result);
}

bool AmdVulkanTokenizeRefinementAC(j_compress_ptr cinfo, int scan_index,
                                   AmdVulkanACResult* result) {
  return g_amd_vulkan_tokenizer.TokenizeRefinementAC(cinfo, scan_index, result);
}

bool AmdVulkanTokenizeDC(j_compress_ptr cinfo, int scan_index,
                         AmdVulkanACResult* result) {
  return g_amd_vulkan_tokenizer.TokenizeDC(cinfo, scan_index, result);
}

}  // namespace jpegli

#endif  // JPEGLI_ENABLE_AMD_VULKAN
