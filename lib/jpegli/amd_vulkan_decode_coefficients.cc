// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jpegli/amd_vulkan_decode_coefficients.h"

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

#include "lib/jpegli/common_internal.h"
#include "lib/jpegli/decode_internal.h"
#include "lib/jpegli/decode_stage_profile_internal.h"

#if defined(JPEGLI_ENABLE_AMD_VULKAN)
#include <vulkan/vulkan.h>

#include "amd_vulkan_decode_coefficients_spv.h"
#endif

namespace jpegli {
namespace {

constexpr size_t kAutoMinComponentBlocks = 4096;

enum class DecodeCoefficientMode {
  kOff,
  kCpuEvents,
  kGpu,
  kForceGpu,
};

DecodeCoefficientMode RequestedMode() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_DECODE_COEFFICIENTS");
  if (value == nullptr || strcmp(value, "0") == 0 ||
      strcmp(value, "off") == 0 || strcmp(value, "false") == 0) {
    return DecodeCoefficientMode::kOff;
  }
  if (strcmp(value, "cpu") == 0) {
    return DecodeCoefficientMode::kCpuEvents;
  }
  if (strcmp(value, "force") == 0) {
    return DecodeCoefficientMode::kForceGpu;
  }
  return DecodeCoefficientMode::kGpu;
}

bool TraceEnabled() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_TRACE");
  return value != nullptr && strcmp(value, "0") != 0 &&
         strcmp(value, "off") != 0 && strcmp(value, "false") != 0;
}

void Trace(const char* message) {
  if (TraceEnabled()) {
    fprintf(stderr, "jpegli amd-vulkan decode: %s\n", message);
  }
}

bool CopyFlatCoefficientsToJpegli(j_decompress_ptr cinfo,
                                  const int16_t* coefficients) {
  jpeg_decomp_master* m = cinfo->master;
  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info& comp = cinfo->comp_info[c];
    const size_t component_coefficient_offset =
        m->amd_decode_component_block_offsets_[c] * DCTSIZE2;
    const size_t row_coefficients =
        static_cast<size_t>(comp.width_in_blocks) * DCTSIZE2;
    for (JDIMENSION by = 0; by < comp.height_in_blocks; ++by) {
      JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(
          reinterpret_cast<j_common_ptr>(cinfo), m->coef_arrays[c], by, 1,
          TRUE);
      memcpy(&blocks[0][0][0],
             coefficients + component_coefficient_offset +
                 static_cast<size_t>(by) * row_coefficients,
             row_coefficients * sizeof(int16_t));
    }
  }
  return true;
}

bool ReconstructOnCpu(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  DecodeStageProfileTimer timer(
      m->decode_stage_profile,
      JPEGLI_DECODE_STAGE_PROGRESSIVE_COEFFICIENT_CPU_FALLBACK);
  std::vector<int32_t> coefficients(m->amd_decode_total_coefficients_, 0);
  for (const AmdVulkanDecodeCoefficientEvent& event :
       m->amd_decode_coefficient_events_) {
    if (event.coefficient_index >= coefficients.size()) return false;
    coefficients[event.coefficient_index] += event.delta;
  }
  std::vector<int16_t> packed(coefficients.size());
  for (size_t i = 0; i < coefficients.size(); ++i) {
    if (coefficients[i] < std::numeric_limits<int16_t>::min() ||
        coefficients[i] > std::numeric_limits<int16_t>::max()) {
      return false;
    }
    packed[i] = static_cast<int16_t>(coefficients[i]);
  }
  return CopyFlatCoefficientsToJpegli(cinfo, packed.data());
}

#if defined(JPEGLI_ENABLE_AMD_VULKAN)

constexpr uint32_t kAmdVendorId = 0x1002;
constexpr uint32_t kWorkgroupSize = 256;

struct MappedBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
};

struct PushConstants {
  uint32_t event_count;
  uint32_t coefficient_count;
  uint32_t mode;
  uint32_t base_index;
};
static_assert(sizeof(PushConstants) == 16,
              "GPU decode push-constant ABI changed");

class AmdVulkanCoefficientReconstructor {
 public:
  ~AmdVulkanCoefficientReconstructor() { Shutdown(); }

  bool Available() { return Initialize(); }

  bool Reconstruct(j_decompress_ptr cinfo) {
    if (!Initialize() || cinfo == nullptr || cinfo->master == nullptr) {
      return false;
    }
    jpeg_decomp_master* m = cinfo->master;
    const size_t event_count = m->amd_decode_coefficient_events_.size();
    const size_t coefficient_count = m->amd_decode_total_coefficients_;
    if (event_count > std::numeric_limits<uint32_t>::max() ||
        coefficient_count == 0 ||
        coefficient_count > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    if (m->decode_stage_profile != nullptr) {
      m->decode_stage_profile->progressive_coefficient_events = event_count;
      m->decode_stage_profile->progressive_coefficient_count =
          coefficient_count;
    }
    const size_t packed_words = (coefficient_count + 1) / 2;
    if (!EnsureBuffer(&event_buffer_,
                      event_count * sizeof(AmdVulkanDecodeCoefficientEvent),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      /*prefer_device_local=*/false) ||
        !EnsureBuffer(&accumulator_buffer_, coefficient_count * sizeof(int32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      /*prefer_device_local=*/true) ||
        !EnsureBuffer(&packed_buffer_, packed_words * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      // The existing CPU IDCT reads this buffer immediately.
                      // Prefer cached coherent UMA over uncached device-local
                      // mappings; a future fused GPU renderer can change this.
                      /*prefer_device_local=*/false) ||
        !UpdateDescriptors()) {
      return false;
    }

    {
      DecodeStageProfileTimer timer(
          m->decode_stage_profile,
          JPEGLI_DECODE_STAGE_PROGRESSIVE_COEFFICIENT_EVENT_UPLOAD);
      if (event_count != 0) {
        memcpy(event_buffer_.mapped, m->amd_decode_coefficient_events_.data(),
               event_count * sizeof(AmdVulkanDecodeCoefficientEvent));
      }
    }

    if (vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) !=
            VK_SUCCESS ||
        vkResetFences(device_, 1, &fence_) != VK_SUCCESS ||
        vkResetCommandBuffer(command_buffer_, 0) != VK_SUCCESS) {
      return false;
    }
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command_buffer_, &begin) != VK_SUCCESS) {
      return false;
    }
    if (query_pool_ != VK_NULL_HANDLE) {
      vkCmdResetQueryPool(command_buffer_, query_pool_, 0, 2);
      vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          query_pool_, 0);
    }

    VkMemoryBarrier host_to_compute = {};
    host_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_to_compute.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host_to_compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &host_to_compute, 0, nullptr, 0, nullptr);
    vkCmdFillBuffer(command_buffer_, accumulator_buffer_.buffer, 0,
                    coefficient_count * sizeof(int32_t), 0);
    VkMemoryBarrier fill_to_compute = {};
    fill_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    fill_to_compute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fill_to_compute.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &fill_to_compute, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0,
                            nullptr);
    if (!RecordDispatches(/*mode=*/0, static_cast<uint32_t>(event_count),
                          static_cast<uint32_t>(coefficient_count))) {
      vkEndCommandBuffer(command_buffer_);
      return false;
    }
    VkMemoryBarrier apply_to_pack = {};
    apply_to_pack.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    apply_to_pack.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    apply_to_pack.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &apply_to_pack, 0, nullptr, 0, nullptr);
    if (!RecordDispatches(/*mode=*/1, static_cast<uint32_t>(packed_words),
                          static_cast<uint32_t>(coefficient_count))) {
      vkEndCommandBuffer(command_buffer_);
      return false;
    }
    VkMemoryBarrier compute_to_host = {};
    compute_to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    compute_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &compute_to_host, 0,
                         nullptr, 0, nullptr);
    if (query_pool_ != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          query_pool_, 1);
    }
    if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) return false;

    const auto submit_start = std::chrono::steady_clock::now();
    {
      DecodeStageProfileTimer timer(
          m->decode_stage_profile,
          JPEGLI_DECODE_STAGE_PROGRESSIVE_COEFFICIENT_GPU_RECONSTRUCTION);
      VkSubmitInfo submit = {};
      submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command_buffer_;
      if (vkQueueSubmit(queue_, 1, &submit, fence_) != VK_SUCCESS ||
          vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) !=
              VK_SUCCESS) {
        return false;
      }
    }

    double device_ms = 0.0;
    if (query_pool_ != VK_NULL_HANDLE) {
      uint64_t timestamps[2] = {};
      if (vkGetQueryPoolResults(device_, query_pool_, 0, 2, sizeof(timestamps),
                                timestamps, sizeof(timestamps[0]),
                                VK_QUERY_RESULT_64_BIT |
                                    VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        device_ms = static_cast<double>(timestamps[1] - timestamps[0]) *
                    properties_.limits.timestampPeriod / 1.0e6;
      }
    }
    if (m->decode_stage_profile != nullptr) {
      m->decode_stage_profile->progressive_coefficient_gpu_device_ns =
          static_cast<uint64_t>(device_ms * 1.0e6 + 0.5);
    }

    {
      DecodeStageProfileTimer timer(
          m->decode_stage_profile,
          JPEGLI_DECODE_STAGE_PROGRESSIVE_COEFFICIENT_READBACK);
      static_assert(sizeof(JCOEF) == sizeof(int16_t),
                    "GPU coefficient output requires the libjpeg int16 ABI");
      if (!CopyFlatCoefficientsToJpegli(
              cinfo, static_cast<const int16_t*>(packed_buffer_.mapped))) {
        return false;
      }
    }
    if (TraceEnabled()) {
      const double submit_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - submit_start)
              .count();
      fprintf(stderr,
              "jpegli amd-vulkan decode: %zu events, %zu coefficients, "
              "device %.3f ms, submit/wait plus readback %.3f ms\n",
              event_count, coefficient_count, device_ms, submit_ms);
    }
    return true;
  }

 private:
  bool RecordDispatches(uint32_t mode, uint32_t item_count,
                        uint32_t coefficient_count) {
    if (item_count == 0) return true;
    const uint64_t max_items =
        static_cast<uint64_t>(properties_.limits.maxComputeWorkGroupCount[0]) *
        kWorkgroupSize;
    if (max_items == 0) return false;
    uint64_t base = 0;
    while (base < item_count) {
      const uint32_t count = static_cast<uint32_t>(
          std::min<uint64_t>(item_count - base, max_items));
      const uint32_t groups = (count + kWorkgroupSize - 1) / kWorkgroupSize;
      PushConstants push = {mode == 0 ? item_count : 0, coefficient_count, mode,
                            static_cast<uint32_t>(base)};
      vkCmdPushConstants(command_buffer_, pipeline_layout_,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
      vkCmdDispatch(command_buffer_, groups, 1, 1);
      base += count;
    }
    return true;
  }

  bool Initialize() {
    if (initialization_attempted_) return available_;
    initialization_attempted_ = true;

    VkApplicationInfo application = {};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "jpegli-amd-progressive-decode";
    application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application.pEngineName = "jpegli-direct-amd-vulkan";
    application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    if (vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS) {
      Trace("Vulkan instance creation failed");
      return false;
    }

    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance_, &count, nullptr) != VK_SUCCESS ||
        count == 0) {
      Trace("no Vulkan physical device");
      ShutdownObjects();
      return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance_, &count, devices.data()) !=
        VK_SUCCESS) {
      ShutdownObjects();
      return false;
    }
    for (VkPhysicalDevice candidate : devices) {
      VkPhysicalDeviceProperties candidate_properties = {};
      vkGetPhysicalDeviceProperties(candidate, &candidate_properties);
      if (candidate_properties.vendorID != kAmdVendorId) continue;
      VkPhysicalDeviceMemoryProperties memory = {};
      vkGetPhysicalDeviceMemoryProperties(candidate, &memory);
      bool has_coherent_uma = false;
      for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = memory.memoryTypes[i].propertyFlags;
        const VkMemoryPropertyFlags required =
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((flags & required) == required) has_coherent_uma = true;
      }
      if (!has_coherent_uma) continue;

      uint32_t family_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count,
                                               nullptr);
      std::vector<VkQueueFamilyProperties> families(family_count);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count,
                                               families.data());
      uint32_t selected = UINT32_MAX;
      for (uint32_t i = 0; i < family_count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
        if (selected == UINT32_MAX ||
            (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
          selected = i;
        }
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) break;
      }
      if (selected == UINT32_MAX) continue;
      physical_device_ = candidate;
      properties_ = candidate_properties;
      queue_family_ = selected;
      timestamp_valid_bits_ = families[selected].timestampValidBits;
      break;
    }
    if (physical_device_ == VK_NULL_HANDLE) {
      Trace("no compatible coherent-memory AMD Vulkan device");
      ShutdownObjects();
      return false;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (vkCreateDevice(physical_device_, &device_info, nullptr, &device_) !=
        VK_SUCCESS) {
      Trace("Vulkan device creation failed");
      ShutdownObjects();
      return false;
    }
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    if (!CreateObjects()) {
      ShutdownObjects();
      return false;
    }
    available_ = true;
    if (TraceEnabled()) {
      fprintf(stderr, "jpegli amd-vulkan decode: selected %s\n",
              properties_.deviceName);
    }
    return true;
  }

  bool CreateObjects() {
    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_layout_info = {};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = bindings.size();
    set_layout_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &set_layout_info, nullptr,
                                    &descriptor_set_layout_) != VK_SUCCESS) {
      return false;
    }
    VkPushConstantRange push_range = {};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
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
    shader_info.codeSize = kAmdVulkanDecodeCoefficientsSpvSize;
    shader_info.pCode =
        reinterpret_cast<const uint32_t*>(kAmdVulkanDecodeCoefficientsSpv);
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &shader_info, nullptr, &shader) !=
        VK_SUCCESS) {
      return false;
    }
    VkPipelineShaderStageCreateInfo stage = {};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
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
    pool_size.descriptorCount = 3;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr,
                               &descriptor_pool_) != VK_SUCCESS) {
      return false;
    }
    VkDescriptorSetAllocateInfo descriptor_info = {};
    descriptor_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_info.descriptorPool = descriptor_pool_;
    descriptor_info.descriptorSetCount = 1;
    descriptor_info.pSetLayouts = &descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &descriptor_info, &descriptor_set_) !=
        VK_SUCCESS) {
      return false;
    }

    VkCommandPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool.queueFamilyIndex = queue_family_;
    if (vkCreateCommandPool(device_, &pool, nullptr, &command_pool_) !=
        VK_SUCCESS) {
      return false;
    }
    VkCommandBufferAllocateInfo command = {};
    command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command.commandPool = command_pool_;
    command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &command, &command_buffer_) !=
        VK_SUCCESS) {
      return false;
    }
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(device_, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
      return false;
    }
    if (timestamp_valid_bits_ != 0) {
      VkQueryPoolCreateInfo query = {};
      query.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      query.queryType = VK_QUERY_TYPE_TIMESTAMP;
      query.queryCount = 2;
      if (vkCreateQueryPool(device_, &query, nullptr, &query_pool_) !=
          VK_SUCCESS) {
        return false;
      }
    }
    return true;
  }

  uint32_t FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred) const {
    VkPhysicalDeviceMemoryProperties memory = {};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory);
    uint32_t best = UINT32_MAX;
    uint32_t best_score = 0;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
      if ((type_bits & (1u << i)) == 0) continue;
      const VkMemoryPropertyFlags flags = memory.memoryTypes[i].propertyFlags;
      if ((flags & required) != required) continue;
      uint32_t score = 1;
      if ((flags & preferred) == preferred) score += 4;
      if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) score += 2;
      if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) score += 1;
      if (score > best_score) {
        best = i;
        best_score = score;
      }
    }
    return best;
  }

  bool EnsureBuffer(MappedBuffer* target, size_t requested_size,
                    VkBufferUsageFlags usage, bool prefer_device_local) {
    if (target->size >= requested_size) return true;
    DestroyBuffer(target);
    VkDeviceSize allocation_size = std::max<size_t>(requested_size, 4096);
    allocation_size += allocation_size / 2;
    VkBufferCreateInfo buffer = {};
    buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer.size = allocation_size;
    buffer.usage = usage;
    buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &buffer, nullptr, &target->buffer) !=
        VK_SUCCESS) {
      return false;
    }
    VkMemoryRequirements requirements = {};
    vkGetBufferMemoryRequirements(device_, target->buffer, &requirements);
    const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
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
    descriptors_dirty_ = true;
    return true;
  }

  bool UpdateDescriptors() {
    if (!descriptors_dirty_) return true;
    std::array<VkDescriptorBufferInfo, 3> buffers = {};
    buffers[0] = {event_buffer_.buffer, 0, event_buffer_.size};
    buffers[1] = {accumulator_buffer_.buffer, 0, accumulator_buffer_.size};
    buffers[2] = {packed_buffer_.buffer, 0, packed_buffer_.size};
    std::array<VkWriteDescriptorSet, 3> writes = {};
    for (uint32_t i = 0; i < writes.size(); ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = descriptor_set_;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffers[i];
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
      DestroyBuffer(&packed_buffer_);
      DestroyBuffer(&accumulator_buffer_);
      DestroyBuffer(&event_buffer_);
      if (query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_, query_pool_, nullptr);
      }
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
    physical_device_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
  }

  void Shutdown() {
    available_ = false;
    ShutdownObjects();
  }

  bool initialization_attempted_ = false;
  bool available_ = false;
  bool descriptors_dirty_ = true;
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
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
  MappedBuffer event_buffer_;
  MappedBuffer accumulator_buffer_;
  MappedBuffer packed_buffer_;
};

thread_local AmdVulkanCoefficientReconstructor g_coefficient_reconstructor;

#endif  // JPEGLI_ENABLE_AMD_VULKAN

}  // namespace

bool AmdVulkanDecodeCoefficientsPrepare(j_decompress_ptr cinfo) {
  if (cinfo == nullptr || cinfo->master == nullptr) return false;
  jpeg_decomp_master* m = cinfo->master;
  const DecodeCoefficientMode mode = RequestedMode();
  if (mode == DecodeCoefficientMode::kOff || !cinfo->progressive_mode ||
      cinfo->buffered_image != FALSE || m->streaming_mode_ ||
      cinfo->num_components <= 0 || cinfo->num_components > kMaxComponents) {
    return false;
  }

  size_t total_blocks = 0;
  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info& comp = cinfo->comp_info[c];
    const size_t component_blocks =
        static_cast<size_t>(comp.width_in_blocks) * comp.height_in_blocks;
    m->amd_decode_component_block_offsets_[c] = total_blocks;
    if (component_blocks > std::numeric_limits<size_t>::max() - total_blocks) {
      return false;
    }
    total_blocks += component_blocks;
  }
  if (total_blocks == 0 ||
      total_blocks > std::numeric_limits<uint32_t>::max() / DCTSIZE2 ||
      (mode != DecodeCoefficientMode::kForceGpu &&
       mode != DecodeCoefficientMode::kCpuEvents &&
       total_blocks < kAutoMinComponentBlocks)) {
    return false;
  }

  bool use_gpu = false;
  if (mode != DecodeCoefficientMode::kCpuEvents) {
#if defined(JPEGLI_ENABLE_AMD_VULKAN)
    use_gpu = g_coefficient_reconstructor.Available();
#endif
    if (!use_gpu) {
      Trace("GPU coefficient backend unavailable; using stock decoder");
      return false;
    }
  }

  m->amd_decode_total_blocks_ = total_blocks;
  m->amd_decode_total_coefficients_ = total_blocks * DCTSIZE2;
  m->amd_decode_significant_.assign(total_blocks, 0);
  m->amd_decode_negative_.assign(total_blocks, 0);
  m->amd_decode_coefficient_events_.clear();
  const size_t reserve_events =
      std::min<size_t>(m->amd_decode_total_coefficients_, total_blocks * 8);
  m->amd_decode_coefficient_events_.reserve(reserve_events);
  m->amd_decode_coefficients_gpu_ = use_gpu;
  m->amd_decode_coefficients_reconstructed_ = false;
  m->amd_decode_coefficients_active_ = true;
  if (TraceEnabled()) {
    fprintf(stderr,
            "jpegli amd-vulkan decode: recording progressive events for %zu "
            "blocks (%s reconstruction)\n",
            total_blocks, use_gpu ? "GPU" : "CPU control");
  }
  return true;
}

bool AmdVulkanDecodeCoefficientsFinish(j_decompress_ptr cinfo) {
  if (cinfo == nullptr || cinfo->master == nullptr) return false;
  jpeg_decomp_master* m = cinfo->master;
  if (!m->amd_decode_coefficients_active_ ||
      m->amd_decode_coefficients_reconstructed_) {
    return true;
  }
  bool reconstructed = false;
  if (m->decode_stage_profile != nullptr) {
    m->decode_stage_profile->progressive_coefficient_events =
        m->amd_decode_coefficient_events_.size();
    m->decode_stage_profile->progressive_coefficient_count =
        m->amd_decode_total_coefficients_;
  }
#if defined(JPEGLI_ENABLE_AMD_VULKAN)
  if (m->amd_decode_coefficients_gpu_) {
    reconstructed = g_coefficient_reconstructor.Reconstruct(cinfo);
  }
#endif
  if (!reconstructed) {
    reconstructed = ReconstructOnCpu(cinfo);
    if (m->amd_decode_coefficients_gpu_) {
      Trace("GPU reconstruction failed; used exact CPU event fallback");
    }
  }
  if (!reconstructed) return false;
  m->amd_decode_coefficients_reconstructed_ = true;
  m->amd_decode_coefficients_active_ = false;
  return true;
}

}  // namespace jpegli
