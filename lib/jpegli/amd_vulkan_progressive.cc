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

class AmdVulkanProgressiveTokenizer {
 public:
  ~AmdVulkanProgressiveTokenizer() { Shutdown(); }

  bool BeginImage(j_compress_ptr cinfo) {
    active_cinfo_ = nullptr;
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
    const auto upload_start = std::chrono::steady_clock::now();
    if (cinfo->num_components >
        static_cast<int>(component_word_offsets_.size())) {
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
      component_word_offsets_[c] = total_words;
      component_num_blocks_[c] = num_blocks;
      total_words += num_blocks * kCoefficientWordsPerBlock;
    }
    if (total_words > std::numeric_limits<uint32_t>::max()) return false;
    if (!EnsureBuffer(&coefficient_buffer_, total_words * sizeof(uint32_t))) {
      return false;
    }

    static_assert(sizeof(coeff_t) == sizeof(JCOEF),
                  "GPU coefficient ABI must match JBLOCK storage");
    auto* destination = static_cast<uint8_t*>(coefficient_buffer_.mapped);
    jpeg_comp_master* master = cinfo->master;
    for (int c = 0; c < cinfo->num_components; ++c) {
      const jpeg_component_info& comp = cinfo->comp_info[c];
      uint8_t* component_destination =
          destination + component_word_offsets_[c] * sizeof(uint32_t);
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
    active_cinfo_ = cinfo;
    if (TraceEnabled()) {
      const double milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - upload_start)
              .count();
      fprintf(stderr,
              "jpegli amd-vulkan: uploaded %zu coefficient words in %.3f ms\n",
              total_words, milliseconds);
    }
    if (!DispatchAllACScans(cinfo)) {
      // A failure can occur after the fence has been reset. Disable this
      // thread's backend so a later image cannot wait on an unsignaled fence;
      // the encoder transparently recomputes this image on the CPU.
      Trace("AC batch failed; disabling AMD Vulkan tokenizer for this thread");
      available_ = false;
      active_cinfo_ = nullptr;
      return false;
    }
    return true;
  }

  void EndImage(j_compress_ptr cinfo) {
    if (active_cinfo_ == cinfo) active_cinfo_ = nullptr;
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
    if (active_cinfo_ != cinfo || result == nullptr) return false;
    const jpeg_scan_info& scan = cinfo->scan_info[scan_index];
    if (scan.comps_in_scan != 1 || scan.Ss <= 0 ||
        (mode == 0 && scan.Ah != 0) || (mode == 1 && scan.Ah == 0) ||
        mode > 1 || scan.Se < scan.Ss || scan.Se >= DCTSIZE2 || scan.Al < 0 ||
        scan.Al > 15 || context < 0 || context > 255) {
      return false;
    }
    if (scan_index >= 0 &&
        scan_index < static_cast<int>(cached_results_.size()) &&
        cached_results_[scan_index].words != nullptr) {
      *result = cached_results_[scan_index];
      return true;
    }

    const int component = scan.component_index[0];
    const size_t num_blocks = component_num_blocks_[component];
    if (num_blocks == 0) return false;
    const size_t stride_words = 2 + scan.Se - scan.Ss + 1;
    if (num_blocks > std::numeric_limits<uint32_t>::max() ||
        stride_words > std::numeric_limits<uint32_t>::max() ||
        num_blocks > std::numeric_limits<size_t>::max() / stride_words) {
      return false;
    }
    const size_t output_words = num_blocks * stride_words;
    if (!EnsureBuffer(&descriptor_buffer_, output_words * sizeof(uint32_t)) ||
        !UpdateDescriptors()) {
      return false;
    }

    const VkPhysicalDeviceLimits& limits = properties_.limits;
    const uint32_t dispatch_width = static_cast<uint32_t>(
        std::min<size_t>(num_blocks, limits.maxComputeWorkGroupCount[0]));
    const uint32_t dispatch_height = static_cast<uint32_t>(
        (num_blocks + dispatch_width - 1) / dispatch_width);
    if (dispatch_height > limits.maxComputeWorkGroupCount[1]) return false;

    PushConstants push = {
        static_cast<uint32_t>(component_word_offsets_[component]),
        static_cast<uint32_t>(num_blocks),
        static_cast<uint32_t>(scan.Ss),
        static_cast<uint32_t>(scan.Se),
        static_cast<uint32_t>(scan.Al),
        static_cast<uint32_t>(stride_words),
        static_cast<uint32_t>(context),
        dispatch_width,
        mode,
        0,
    };

    const auto start = std::chrono::steady_clock::now();
    if (vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) !=
            VK_SUCCESS ||
        vkResetFences(device_, 1, &fence_) != VK_SUCCESS ||
        vkResetCommandBuffer(command_buffer_, 0) != VK_SUCCESS) {
      return false;
    }

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command_buffer_, &begin_info) != VK_SUCCESS) {
      return false;
    }
    VkMemoryBarrier host_to_compute = {};
    host_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_to_compute.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host_to_compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &host_to_compute, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0,
                            nullptr);
    vkCmdPushConstants(command_buffer_, pipeline_layout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(command_buffer_, dispatch_width, dispatch_height, 1);
    VkMemoryBarrier compute_to_host = {};
    compute_to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    compute_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &compute_to_host, 0,
                         nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) return false;

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    if (vkQueueSubmit(queue_, 1, &submit, fence_) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) !=
            VK_SUCCESS) {
      return false;
    }

    result->words = static_cast<const uint32_t*>(descriptor_buffer_.mapped);
    result->stride_words = stride_words;
    result->num_blocks = num_blocks;
    if (TraceEnabled()) {
      const double milliseconds = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
      fprintf(stderr, "jpegli amd-vulkan: %s AC scan %d, %zu blocks, %.3f ms\n",
              mode == 0 ? "initial" : "refinement", scan_index, num_blocks,
              milliseconds);
    }
    return true;
  }

  bool DispatchAllACScans(j_compress_ptr cinfo) {
    struct ScanDispatch {
      int scan_index;
      uint32_t width;
      uint32_t height;
      PushConstants push;
    };
    std::vector<ScanDispatch> dispatches;
    cached_results_.assign(cinfo->num_scans, {});
    size_t total_output_words = 0;
    for (int scan_index = 0; scan_index < cinfo->num_scans; ++scan_index) {
      const jpeg_scan_info& scan = cinfo->scan_info[scan_index];
      if (scan.Ss <= 0 || scan.comps_in_scan != 1 || scan.Se < scan.Ss ||
          scan.Se >= DCTSIZE2 || scan.Al < 0 || scan.Al > 15) {
        continue;
      }
      const int component = scan.component_index[0];
      const size_t num_blocks = component_num_blocks_[component];
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
          static_cast<uint32_t>(component_word_offsets_[component]),
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
      cached_results_[scan_index].stride_words = stride_words;
      cached_results_[scan_index].num_blocks = num_blocks;
      total_output_words += num_blocks * stride_words;
    }
    if (dispatches.empty()) return true;
    if (!EnsureBuffer(&descriptor_buffer_,
                      total_output_words * sizeof(uint32_t)) ||
        !UpdateDescriptors()) {
      cached_results_.clear();
      return false;
    }
    const uint32_t* output =
        static_cast<const uint32_t*>(descriptor_buffer_.mapped);
    for (const ScanDispatch& dispatch : dispatches) {
      cached_results_[dispatch.scan_index].words =
          output + dispatch.push.output_word_offset;
    }

    const auto start = std::chrono::steady_clock::now();
    if (vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) !=
            VK_SUCCESS ||
        vkResetFences(device_, 1, &fence_) != VK_SUCCESS ||
        vkResetCommandBuffer(command_buffer_, 0) != VK_SUCCESS) {
      cached_results_.clear();
      return false;
    }
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command_buffer_, &begin_info) != VK_SUCCESS) {
      cached_results_.clear();
      return false;
    }
    VkMemoryBarrier host_to_compute = {};
    host_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_to_compute.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host_to_compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &host_to_compute, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0,
                            nullptr);
    for (const ScanDispatch& dispatch : dispatches) {
      vkCmdPushConstants(command_buffer_, pipeline_layout_,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dispatch.push),
                         &dispatch.push);
      vkCmdDispatch(command_buffer_, dispatch.width, dispatch.height, 1);
    }
    VkMemoryBarrier compute_to_host = {};
    compute_to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    compute_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &compute_to_host, 0,
                         nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) {
      cached_results_.clear();
      return false;
    }
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    if (vkQueueSubmit(queue_, 1, &submit, fence_) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) !=
            VK_SUCCESS) {
      cached_results_.clear();
      return false;
    }
    if (TraceEnabled()) {
      const double milliseconds = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
      fprintf(stderr,
              "jpegli amd-vulkan: batched %zu AC scans (%zu descriptor words) "
              "in %.3f ms\n",
              dispatches.size(), total_output_words, milliseconds);
    }
    return true;
  }

 private:
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
    pool_size.descriptorCount = 2;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr,
                               &descriptor_pool_) != VK_SUCCESS) {
      return false;
    }
    VkDescriptorSetAllocateInfo set_info = {};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = descriptor_pool_;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &set_info, &descriptor_set_) !=
        VK_SUCCESS) {
      return false;
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
    command_info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &command_info, &command_buffer_) !=
        VK_SUCCESS) {
      return false;
    }
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    return vkCreateFence(device_, &fence_info, nullptr, &fence_) == VK_SUCCESS;
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

  bool EnsureBuffer(MappedBuffer* target, size_t requested_size) {
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
    descriptors_dirty_ = true;
    return true;
  }

  bool UpdateDescriptors() {
    if (!descriptors_dirty_) return true;
    if (coefficient_buffer_.buffer == VK_NULL_HANDLE ||
        descriptor_buffer_.buffer == VK_NULL_HANDLE) {
      return false;
    }
    std::array<VkDescriptorBufferInfo, 2> buffer_info = {};
    buffer_info[0].buffer = coefficient_buffer_.buffer;
    buffer_info[0].range = coefficient_buffer_.size;
    buffer_info[1].buffer = descriptor_buffer_.buffer;
    buffer_info[1].range = descriptor_buffer_.size;
    std::array<VkWriteDescriptorSet, 2> writes = {};
    for (uint32_t i = 0; i < writes.size(); ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = descriptor_set_;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffer_info[i];
    }
    vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, nullptr);
    descriptors_dirty_ = false;
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
      DestroyBuffer(&descriptor_buffer_);
      DestroyBuffer(&coefficient_buffer_);
      if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
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
    active_cinfo_ = nullptr;
    available_ = false;
    ShutdownObjects();
  }

  bool initialization_attempted_ = false;
  bool available_ = false;
  bool descriptors_dirty_ = true;
  j_compress_ptr active_cinfo_ = nullptr;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties_ = {};
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = UINT32_MAX;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  MappedBuffer coefficient_buffer_;
  MappedBuffer descriptor_buffer_;
  std::array<size_t, kMaxComponents> component_word_offsets_ = {};
  std::array<size_t, kMaxComponents> component_num_blocks_ = {};
  std::vector<AmdVulkanACResult> cached_results_;
  AmdTuningProfile tuning_profile_ = AmdTuningProfile::kOtherUnifiedAmd;
};

thread_local AmdVulkanProgressiveTokenizer g_amd_vulkan_tokenizer;

}  // namespace

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
