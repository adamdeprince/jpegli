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

#include "amd_vulkan_progressive_spv.h"
#include "lib/jpegli/encode_internal.h"

namespace jpegli {
namespace {

constexpr uint32_t kAmdVendorId = 0x1002;
constexpr uint32_t kWaveSize = 64;
constexpr size_t kCoefficientWordsPerBlock = DCTSIZE2 / 2;
constexpr size_t kAutoMinComponentBlocks = 4096;
constexpr size_t kPipelineSlots = 2;

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
};
static_assert(sizeof(PushConstants) == 40, "shader push-constant ABI changed");

struct PipelineSlot {
  j_compress_ptr cinfo = nullptr;
  MappedBuffer coefficient_buffer;
  MappedBuffer descriptor_buffer;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool query_pool = VK_NULL_HANDLE;
  bool descriptors_dirty = true;
  bool in_flight = false;
  bool ready = false;
  std::array<size_t, kMaxComponents> component_word_offsets = {};
  std::array<size_t, kMaxComponents> component_num_blocks = {};
  std::vector<AmdVulkanACResult> cached_results;
  std::chrono::steady_clock::time_point submit_time;
};

class AmdVulkanProgressiveTokenizer {
 public:
  ~AmdVulkanProgressiveTokenizer() { Shutdown(); }

  bool SubmitImage(j_compress_ptr cinfo) {
    if (cinfo == nullptr || !cinfo->progressive_mode ||
        cinfo->master == nullptr)
      return false;
    if (FindSlot(cinfo) != nullptr) return true;
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
    if (!Initialize()) return false;
    PipelineSlot* slot = FindFreeSlot();
    if (slot == nullptr) {
      Trace("both pipeline slots are occupied; using CPU tokenizer");
      return false;
    }
    const auto upload_start = std::chrono::steady_clock::now();
    if (cinfo->num_components >
        static_cast<int>(slot->component_word_offsets.size())) {
      return false;
    }

    size_t total_words = 0;
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
    if (total_words > std::numeric_limits<uint32_t>::max()) return false;
    if (!EnsureBuffer(slot, &slot->coefficient_buffer,
                      total_words * sizeof(uint32_t))) {
      return false;
    }

    static_assert(sizeof(coeff_t) == sizeof(JCOEF),
                  "GPU coefficient ABI must match JBLOCK storage");
    auto* destination = static_cast<uint8_t*>(slot->coefficient_buffer.mapped);
    jpeg_comp_master* master = cinfo->master;
    for (int c = 0; c < cinfo->num_components; ++c) {
      const jpeg_component_info& comp = cinfo->comp_info[c];
      uint8_t* component_destination =
          destination + slot->component_word_offsets[c] * sizeof(uint32_t);
      const size_t row_bytes =
          static_cast<size_t>(comp.width_in_blocks) * DCTSIZE2 * sizeof(JCOEF);
      for (JDIMENSION by = 0; by < comp.height_in_blocks; ++by) {
        JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(
            reinterpret_cast<j_common_ptr>(cinfo), master->coeff_buffers[c], by,
            1, FALSE);
        memcpy(component_destination + static_cast<size_t>(by) * row_bytes,
               &blocks[0][0][0], row_bytes);
      }
    }
    slot->cinfo = cinfo;
    slot->ready = false;
    if (TraceEnabled()) {
      const double milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - upload_start)
              .count();
      fprintf(stderr,
              "jpegli amd-vulkan: uploaded %zu coefficient words in %.3f ms\n",
              total_words, milliseconds);
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
    if (slot == nullptr) {
      if (!SubmitImage(cinfo)) return false;
      slot = FindSlot(cinfo);
    }
    if (slot == nullptr) return false;
    if (!WaitForSlot(slot)) {
      available_ = false;
      ReleaseSlot(slot);
      return false;
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
      uint32_t width;
      uint32_t height;
      PushConstants push;
    };
    std::vector<ScanDispatch> dispatches;
    slot->cached_results.assign(cinfo->num_scans, {});
    size_t total_output_words = 0;
    size_t total_wave64_workgroups = 0;
    for (int scan_index = 0; scan_index < cinfo->num_scans; ++scan_index) {
      const jpeg_scan_info& scan = cinfo->scan_info[scan_index];
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
      PushConstants push = {
          static_cast<uint32_t>(slot->component_word_offsets[component]),
          static_cast<uint32_t>(num_blocks),
          static_cast<uint32_t>(scan.Ss),
          static_cast<uint32_t>(scan.Se),
          static_cast<uint32_t>(scan.Al),
          static_cast<uint32_t>(stride_words),
          static_cast<uint32_t>(cinfo->master->ac_ctx_offset[scan_index]),
          width,
          mode,
          static_cast<uint32_t>(total_output_words),
      };
      dispatches.push_back({scan_index, width, height, push});
      slot->cached_results[scan_index].stride_words = stride_words;
      slot->cached_results[scan_index].num_blocks = num_blocks;
      total_output_words += num_blocks * stride_words;
      total_wave64_workgroups += num_blocks;
    }
    if (dispatches.empty()) {
      slot->ready = true;
      return true;
    }
    if (!EnsureBuffer(slot, &slot->descriptor_buffer,
                      total_output_words * sizeof(uint32_t)) ||
        !UpdateDescriptors(slot)) {
      slot->cached_results.clear();
      return false;
    }
    const uint32_t* output =
        static_cast<const uint32_t*>(slot->descriptor_buffer.mapped);
    for (const ScanDispatch& dispatch : dispatches) {
      slot->cached_results[dispatch.scan_index].words =
          output + dispatch.push.output_word_offset;
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
      vkCmdResetQueryPool(slot->command_buffer, slot->query_pool, 0, 2);
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
    vkCmdBindPipeline(slot->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_);
    vkCmdBindDescriptorSets(slot->command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0,
                            1, &slot->descriptor_set, 0, nullptr);
    for (const ScanDispatch& dispatch : dispatches) {
      vkCmdPushConstants(slot->command_buffer, pipeline_layout_,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dispatch.push),
                         &dispatch.push);
      vkCmdDispatch(slot->command_buffer, dispatch.width, dispatch.height, 1);
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
                          slot->query_pool, 1);
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
              "workgroups (%zu descriptor words) asynchronously\n",
              dispatches.size(), total_wave64_workgroups, total_output_words);
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
      std::array<uint64_t, 2> timestamps = {};
      if (slot->query_pool != VK_NULL_HANDLE && timestamp_valid_bits_ != 0 &&
          vkGetQueryPoolResults(device_, slot->query_pool, 0, 2,
                                sizeof(timestamps), timestamps.data(),
                                sizeof(timestamps[0]),
                                VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        const uint64_t mask = timestamp_valid_bits_ >= 64
                                  ? std::numeric_limits<uint64_t>::max()
                                  : (uint64_t{1} << timestamp_valid_bits_) - 1;
        const uint64_t ticks = (timestamps[1] - timestamps[0]) & mask;
        gpu_ms = static_cast<double>(ticks) *
                 properties_.limits.timestampPeriod / 1000000.0;
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
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {};
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

    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 2 * kPipelineSlots;
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
    query_info.queryCount = 2;
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
                    size_t requested_size) {
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
    // Both deploy targets are UMA. CPU-cached GTT is decisively lower latency
    // than mapping the small, CPU-uncached VRAM carveout: the latter makes the
    // descriptor stitching and coefficient upload several times slower than
    // the compute dispatch. The backend requires coherent memory and prefers a
    // cached type on either target.
    const VkMemoryPropertyFlags preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
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
        slot->descriptor_buffer.buffer == VK_NULL_HANDLE) {
      return false;
    }
    std::array<VkDescriptorBufferInfo, 2> buffer_info = {};
    buffer_info[0].buffer = slot->coefficient_buffer.buffer;
    buffer_info[0].range = slot->coefficient_buffer.size;
    buffer_info[1].buffer = slot->descriptor_buffer.buffer;
    buffer_info[1].range = slot->descriptor_buffer.size;
    std::array<VkWriteDescriptorSet, 2> writes = {};
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
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  AmdTuningProfile tuning_profile_ = AmdTuningProfile::kOtherUnifiedAmd;
};

thread_local AmdVulkanProgressiveTokenizer g_amd_vulkan_tokenizer;

}  // namespace

bool AmdVulkanProgressiveSubmit(j_compress_ptr cinfo) {
  return g_amd_vulkan_tokenizer.SubmitImage(cinfo);
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

}  // namespace jpegli

#endif  // JPEGLI_ENABLE_AMD_VULKAN
