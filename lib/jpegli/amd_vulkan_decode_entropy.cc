// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jpegli/amd_vulkan_decode_entropy.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include "lib/jpegli/common_internal.h"
#include "lib/jpegli/decode_internal.h"
#include "lib/jpegli/decode_stage_profile_internal.h"
#include "lib/jpegli/huffman.h"

#if defined(JPEGLI_ENABLE_AMD_VULKAN)
#include <vulkan/vulkan.h>

#include "amd_vulkan_decode_entropy_spv.h"
#endif

namespace jpegli {
namespace {

constexpr size_t kPackedSegmentWords = 8;

enum class ParallelEntropyMode {
  kOff,
  kCpuIndependent,
  kGpuIndependent,
  kCpuRestart,
  kGpuRestart,
};

struct PreparedEntropyWork {
  // Two words per task: first segment and number of scan segments.
  std::vector<uint32_t> tasks;
  // Eight words per ordered scan segment. See BuildPreparedWork().
  std::vector<uint32_t> segments;
  std::vector<uint32_t> luts;
  size_t coefficient_count = 0;
};

ParallelEntropyMode RequestedMode() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_DECODE_ENTROPY");
  if (value == nullptr || strcmp(value, "0") == 0 ||
      strcmp(value, "off") == 0 || strcmp(value, "false") == 0) {
    return ParallelEntropyMode::kOff;
  }
  if (strcmp(value, "cpu-independent") == 0) {
    return ParallelEntropyMode::kCpuIndependent;
  }
  if (strcmp(value, "cpu-restart") == 0) {
    return ParallelEntropyMode::kCpuRestart;
  }
  if (strcmp(value, "restart") == 0) {
    return ParallelEntropyMode::kGpuRestart;
  }
  return ParallelEntropyMode::kGpuIndependent;
}

bool IsGpuMode(ParallelEntropyMode mode) {
  return mode == ParallelEntropyMode::kGpuIndependent ||
         mode == ParallelEntropyMode::kGpuRestart;
}

bool RequiresRestarts(ParallelEntropyMode mode) {
  return mode == ParallelEntropyMode::kCpuRestart ||
         mode == ParallelEntropyMode::kGpuRestart;
}

bool TraceEnabled() {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_TRACE");
  return value != nullptr && strcmp(value, "0") != 0 &&
         strcmp(value, "off") != 0 && strcmp(value, "false") != 0;
}

void Trace(const char* message) {
  if (TraceEnabled()) {
    fprintf(stderr, "jpegli amd-vulkan entropy: %s\n", message);
  }
}

uint32_t PackHuffmanEntry(const HuffmanTableEntry& entry) {
  return static_cast<uint32_t>(entry.bits) |
         (static_cast<uint32_t>(entry.value) << 16);
}

bool CopyFlatCoefficientsToJpegli(j_decompress_ptr cinfo,
                                  const int16_t* coefficients) {
  jpeg_decomp_master* m = cinfo->master;
  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info& comp = cinfo->comp_info[c];
    const size_t component_offset =
        m->amd_decode_entropy_component_block_offsets_[c] * DCTSIZE2;
    const size_t row_coefficients =
        static_cast<size_t>(comp.width_in_blocks) * DCTSIZE2;
    for (JDIMENSION by = 0; by < comp.height_in_blocks; ++by) {
      JBLOCKARRAY blocks = (*cinfo->mem->access_virt_barray)(
          reinterpret_cast<j_common_ptr>(cinfo), m->coef_arrays[c], by, 1,
          TRUE);
      memcpy(&blocks[0][0][0],
             coefficients + component_offset +
                 static_cast<size_t>(by) * row_coefficients,
             row_coefficients * sizeof(int16_t));
    }
  }
  return true;
}

bool BuildPreparedWork(j_decompress_ptr cinfo, PreparedEntropyWork* work) {
  jpeg_decomp_master* m = cinfo->master;
  const size_t scan_count = m->amd_decode_entropy_scans_.size();
  if (scan_count == 0 || scan_count > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  work->coefficient_count = m->amd_decode_entropy_total_coefficients_;
  std::vector<uint32_t> dc_offsets(scan_count);
  std::vector<uint32_t> ac_offsets(scan_count);
  for (size_t i = 0; i < scan_count; ++i) {
    const AmdVulkanDecodeEntropyScan& scan = m->amd_decode_entropy_scans_[i];
    if (scan.dc_lut.size() != kJpegHuffmanLutSize ||
        scan.ac_lut.size() != kJpegHuffmanLutSize ||
        work->luts.size() >
            std::numeric_limits<uint32_t>::max() - 2 * kJpegHuffmanLutSize) {
      return false;
    }
    dc_offsets[i] = static_cast<uint32_t>(work->luts.size());
    work->luts.insert(work->luts.end(), scan.dc_lut.begin(), scan.dc_lut.end());
    ac_offsets[i] = static_cast<uint32_t>(work->luts.size());
    work->luts.insert(work->luts.end(), scan.ac_lut.begin(), scan.ac_lut.end());
  }

  std::vector<std::vector<size_t>> component_scans(cinfo->num_components);
  for (size_t i = 0; i < scan_count; ++i) {
    const uint32_t component = m->amd_decode_entropy_scans_[i].component;
    if (component >= component_scans.size()) return false;
    component_scans[component].push_back(i);
  }
  for (int c = 0; c < cinfo->num_components; ++c) {
    const std::vector<size_t>& scan_indices = component_scans[c];
    if (scan_indices.empty()) return false;
    const AmdVulkanDecodeEntropyScan& first =
        m->amd_decode_entropy_scans_[scan_indices[0]];
    const size_t segment_count = first.segments.size();
    if (segment_count == 0) return false;
    for (size_t scan_index : scan_indices) {
      const AmdVulkanDecodeEntropyScan& scan =
          m->amd_decode_entropy_scans_[scan_index];
      if (scan.total_blocks != first.total_blocks ||
          scan.segments.size() != segment_count) {
        return false;
      }
      if (m->amd_decode_entropy_require_restarts_) {
        if (scan.restart_interval == 0 || segment_count < 2) return false;
      } else if (scan.restart_interval != 0 || segment_count != 1) {
        return false;
      }
      for (size_t segment_index = 0; segment_index < segment_count;
           ++segment_index) {
        if (scan.segments[segment_index].block_start !=
                first.segments[segment_index].block_start ||
            scan.segments[segment_index].block_count !=
                first.segments[segment_index].block_count) {
          return false;
        }
      }
    }

    for (size_t segment_index = 0; segment_index < segment_count;
         ++segment_index) {
      const size_t ordered_first = work->segments.size() / kPackedSegmentWords;
      if (ordered_first > std::numeric_limits<uint32_t>::max() ||
          scan_indices.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
      }
      work->tasks.push_back(static_cast<uint32_t>(ordered_first));
      work->tasks.push_back(static_cast<uint32_t>(scan_indices.size()));
      for (size_t scan_index : scan_indices) {
        const AmdVulkanDecodeEntropyScan& scan =
            m->amd_decode_entropy_scans_[scan_index];
        const AmdVulkanDecodeEntropySegment& segment =
            scan.segments[segment_index];
        const uint64_t block_base =
            m->amd_decode_entropy_component_block_offsets_[c] +
            segment.block_start;
        if (block_base > std::numeric_limits<uint32_t>::max()) return false;
        const uint32_t params =
            scan.ss | (scan.se << 8) | (scan.ah << 16) | (scan.al << 24);
        work->segments.insert(
            work->segments.end(),
            {segment.data_offset, segment.data_size,
             static_cast<uint32_t>(block_base), segment.block_count, params,
             dc_offsets[scan_index], ac_offsets[scan_index], 0});
      }
    }
  }
  return !work->tasks.empty();
}

class CapturedBitReader {
 public:
  CapturedBitReader(const std::vector<uint8_t>& bytes, uint32_t offset,
                    uint32_t size)
      : bytes_(bytes), pos_(offset), end_(static_cast<size_t>(offset) + size) {
    if (end_ > bytes_.size()) ok_ = false;
  }

  uint32_t ReadBits(uint32_t count) {
    if (count == 0) return 0;
    if (count > 16 || !Fill(count)) {
      ok_ = false;
      return 0;
    }
    const uint32_t value =
        (bits_ >> (bits_left_ - count)) & ((uint32_t{1} << count) - 1);
    bits_left_ -= count;
    consumed_bits_ += count;
    if (consumed_bits_ > loaded_valid_bits_) ok_ = false;
    return value;
  }

  uint32_t PeekBits(uint32_t count) {
    if (count == 0) return 0;
    if (count > 16 || !Fill(count)) {
      ok_ = false;
      return 0;
    }
    return (bits_ >> (bits_left_ - count)) & ((uint32_t{1} << count) - 1);
  }

  void ConsumeBits(uint32_t count) {
    if (!Fill(count)) {
      ok_ = false;
      return;
    }
    bits_left_ -= count;
    consumed_bits_ += count;
    if (consumed_bits_ > loaded_valid_bits_) ok_ = false;
  }

  bool ok() const { return ok_; }

 private:
  bool Fill(uint32_t count) {
    while (ok_ && bits_left_ < count) {
      if (pos_ >= end_) {
        // Jpegli's ordinary bit reader peeks through the following marker and
        // supplies zeroes. A final short Huffman code may therefore require a
        // wider lookup peek than the physical entropy payload contains. These
        // virtual bits are legal to peek but never legal to consume.
        bits_ <<= 8;
        bits_left_ += 8;
      } else {
        const uint32_t byte = bytes_[pos_++];
        if (byte == 0xff) {
          if (pos_ >= end_ || bytes_[pos_] != 0) {
            ok_ = false;
            break;
          }
          ++pos_;
        }
        if (bits_left_ > 24) {
          ok_ = false;
          break;
        }
        bits_ = (bits_ << 8) | byte;
        bits_left_ += 8;
        loaded_valid_bits_ += 8;
      }
    }
    return ok_;
  }

  const std::vector<uint8_t>& bytes_;
  size_t pos_;
  size_t end_;
  uint32_t bits_ = 0;
  uint32_t bits_left_ = 0;
  uint64_t loaded_valid_bits_ = 0;
  uint64_t consumed_bits_ = 0;
  bool ok_ = true;
};

uint32_t ReadSymbol(const PreparedEntropyWork& work, uint32_t lut_offset,
                    CapturedBitReader* reader) {
  const uint32_t root_value = reader->PeekBits(8);
  if (!reader->ok() || lut_offset + root_value >= work.luts.size()) {
    return kJpegHuffmanAlphabetSize;
  }
  const uint32_t root_index = lut_offset + root_value;
  const uint32_t root = work.luts[root_index];
  const uint32_t root_bits = root & 0xff;
  if (root_bits <= kJpegHuffmanRootTableBits) {
    reader->ConsumeBits(root_bits);
    return root >> 16;
  }
  reader->ConsumeBits(kJpegHuffmanRootTableBits);
  const uint32_t table_bits = root_bits - kJpegHuffmanRootTableBits;
  const uint32_t low = reader->PeekBits(table_bits);
  const uint64_t second_index =
      static_cast<uint64_t>(root_index) + (root >> 16) + low;
  if (!reader->ok() || second_index >= work.luts.size()) {
    return kJpegHuffmanAlphabetSize;
  }
  const uint32_t second = work.luts[second_index];
  reader->ConsumeBits(second & 0xff);
  return second >> 16;
}

int HuffExtend(uint32_t value, uint32_t bits) {
  const int half = 1 << (bits - 1);
  return value >= static_cast<uint32_t>(half)
             ? static_cast<int>(value)
             : static_cast<int>(value) - (1 << bits) + 1;
}

bool DecodeInitialBlock(const PreparedEntropyWork& work, uint32_t params,
                        uint32_t dc_lut, uint32_t ac_lut, int* last_dc,
                        int* eobrun, CapturedBitReader* reader,
                        int32_t* coefficients) {
  int ss = params & 0xff;
  const int se = (params >> 8) & 0xff;
  const int al = (params >> 24) & 0xff;
  const int amplitude = 1 << al;
  const bool eobrun_allowed = ss > 0;
  if (ss == 0) {
    const uint32_t symbol = ReadSymbol(work, dc_lut, reader);
    if (symbol >= kJpegDCAlphabetSize) return false;
    int difference = 0;
    if (symbol > 0) {
      difference = HuffExtend(reader->ReadBits(symbol), symbol);
    }
    *last_dc += difference;
    const int value = *last_dc * amplitude;
    if (value < std::numeric_limits<int16_t>::min() ||
        value > std::numeric_limits<int16_t>::max()) {
      return false;
    }
    coefficients[0] = value;
    ++ss;
  }
  if (ss > se) return reader->ok();
  if (*eobrun > 0) {
    --(*eobrun);
    return true;
  }
  for (int k = ss; k <= se; ++k) {
    const uint32_t symbol = ReadSymbol(work, ac_lut, reader);
    if (symbol >= kJpegHuffmanAlphabetSize) return false;
    const int run = symbol >> 4;
    const int size = symbol & 15;
    if (size > 0) {
      k += run;
      if (k > se || size + al >= kJpegDCAlphabetSize) return false;
      const int value = HuffExtend(reader->ReadBits(size), size) * amplitude;
      coefficients[kJPEGNaturalOrder[k]] = value;
    } else if (run == 15) {
      k += 15;
    } else {
      *eobrun = 1 << run;
      if (run > 0) {
        if (!eobrun_allowed) return false;
        *eobrun += reader->ReadBits(run);
      }
      break;
    }
  }
  --(*eobrun);
  return reader->ok();
}

bool RefineBlock(const PreparedEntropyWork& work, uint32_t params,
                 uint32_t ac_lut, int* eobrun, CapturedBitReader* reader,
                 int32_t* coefficients) {
  int ss = params & 0xff;
  const int se = (params >> 8) & 0xff;
  const int al = (params >> 24) & 0xff;
  const int p1 = 1 << al;
  const int m1 = -p1;
  const bool eobrun_allowed = ss > 0;
  if (ss == 0) {
    coefficients[0] |= reader->ReadBits(1) * p1;
    ++ss;
  }
  if (ss > se) return reader->ok();
  int k = ss;
  bool in_zero_run = false;
  if (*eobrun <= 0) {
    for (; k <= se; ++k) {
      const uint32_t symbol = ReadSymbol(work, ac_lut, reader);
      if (symbol >= kJpegHuffmanAlphabetSize) return false;
      int run = symbol >> 4;
      int value_size = symbol & 15;
      int new_value = 0;
      if (value_size != 0) {
        if (value_size != 1) return false;
        new_value = reader->ReadBits(1) ? p1 : m1;
        in_zero_run = false;
      } else {
        if (run != 15) {
          *eobrun = 1 << run;
          if (run > 0) {
            if (!eobrun_allowed) return false;
            *eobrun += reader->ReadBits(run);
          }
          break;
        }
        in_zero_run = true;
      }
      do {
        const int coefficient = kJPEGNaturalOrder[k];
        int32_t current = coefficients[coefficient];
        if (current != 0) {
          if (reader->ReadBits(1) && (current & p1) == 0) {
            current += current >= 0 ? p1 : m1;
            coefficients[coefficient] = current;
          }
        } else if (--run < 0) {
          break;
        }
        ++k;
      } while (k <= se);
      if (new_value != 0) {
        if (k > se) return false;
        coefficients[kJPEGNaturalOrder[k]] = new_value;
      }
    }
  }
  if (in_zero_run) return false;
  if (*eobrun > 0) {
    for (; k <= se; ++k) {
      const int coefficient = kJPEGNaturalOrder[k];
      int32_t current = coefficients[coefficient];
      if (current != 0 && reader->ReadBits(1) && (current & p1) == 0) {
        current += current >= 0 ? p1 : m1;
        coefficients[coefficient] = current;
      }
    }
  }
  --(*eobrun);
  return reader->ok();
}

bool DecodeTask(const PreparedEntropyWork& work,
                const std::vector<uint8_t>& bytes, size_t task_index,
                std::vector<int32_t>* coefficients) {
  const uint32_t first_segment = work.tasks[2 * task_index];
  const uint32_t segment_count = work.tasks[2 * task_index + 1];
  for (uint32_t scan_in_task = 0; scan_in_task < segment_count;
       ++scan_in_task) {
    const size_t segment_index = first_segment + scan_in_task;
    const size_t base = segment_index * kPackedSegmentWords;
    if (base + kPackedSegmentWords > work.segments.size()) return false;
    const uint32_t data_offset = work.segments[base];
    const uint32_t data_size = work.segments[base + 1];
    const uint32_t block_base = work.segments[base + 2];
    const uint32_t block_count = work.segments[base + 3];
    const uint32_t params = work.segments[base + 4];
    const uint32_t dc_lut = work.segments[base + 5];
    const uint32_t ac_lut = work.segments[base + 6];
    if (static_cast<uint64_t>(block_base + block_count) * DCTSIZE2 >
        coefficients->size()) {
      if (TraceEnabled()) {
        fprintf(stderr,
                "jpegli amd-vulkan entropy: task %zu segment %u has invalid "
                "block range %u+%u\n",
                task_index, scan_in_task, block_base, block_count);
      }
      return false;
    }
    CapturedBitReader reader(bytes, data_offset, data_size);
    int last_dc = 0;
    int eobrun = -1;
    for (uint32_t block = 0; block < block_count; ++block) {
      int32_t* output = &(*coefficients)[(block_base + block) * DCTSIZE2];
      const int ah = (params >> 16) & 0xff;
      const bool ok =
          ah == 0 ? DecodeInitialBlock(work, params, dc_lut, ac_lut, &last_dc,
                                       &eobrun, &reader, output)
                  : RefineBlock(work, params, ac_lut, &eobrun, &reader, output);
      if (!ok) {
        if (TraceEnabled()) {
          fprintf(stderr,
                  "jpegli amd-vulkan entropy: task %zu segment %u failed at "
                  "block %u/%u params=%02x/%02x/%02x/%02x reader=%d "
                  "eobrun=%d\n",
                  task_index, scan_in_task, block, block_count, params & 0xff,
                  (params >> 8) & 0xff, (params >> 16) & 0xff,
                  (params >> 24) & 0xff, reader.ok(), eobrun);
        }
        return false;
      }
    }
    if (eobrun > 0 || !reader.ok()) {
      if (TraceEnabled()) {
        fprintf(stderr,
                "jpegli amd-vulkan entropy: task %zu segment %u ended with "
                "reader=%d eobrun=%d\n",
                task_index, scan_in_task, reader.ok(), eobrun);
      }
      return false;
    }
  }
  return true;
}

size_t RequestedCpuThreads(size_t task_count) {
  const char* value = std::getenv("JPEGLI_AMD_VULKAN_DECODE_ENTROPY_THREADS");
  if (value != nullptr) {
    char* end = nullptr;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (end != value && *end == 0 && parsed > 0) {
      return std::min<size_t>(task_count, parsed);
    }
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  return std::min<size_t>(task_count, hardware == 0 ? 1 : hardware);
}

bool ReconstructOnCpu(j_decompress_ptr cinfo, const PreparedEntropyWork& work) {
  jpeg_decomp_master* m = cinfo->master;
  DecodeStageProfileTimer timer(m->decode_stage_profile,
                                JPEGLI_DECODE_STAGE_PARALLEL_ENTROPY_CPU_PARSE);
  std::vector<int32_t> coefficients(work.coefficient_count, 0);
  const size_t task_count = work.tasks.size() / 2;
  const size_t thread_count = RequestedCpuThreads(task_count);
  std::atomic<size_t> next_task{0};
  std::atomic<bool> ok{true};
  auto worker = [&]() {
    while (ok.load(std::memory_order_relaxed)) {
      const size_t task = next_task.fetch_add(1, std::memory_order_relaxed);
      if (task >= task_count) break;
      if (!DecodeTask(work, m->amd_decode_entropy_bytes_, task,
                      &coefficients)) {
        ok.store(false, std::memory_order_relaxed);
      }
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(thread_count > 0 ? thread_count - 1 : 0);
  for (size_t i = 1; i < thread_count; ++i) threads.emplace_back(worker);
  worker();
  for (std::thread& thread : threads) thread.join();
  if (!ok.load(std::memory_order_relaxed)) return false;
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
constexpr uint32_t kWorkgroupSize = 32;
constexpr uint32_t kRequiredSubgroupSize = 32;

struct MappedBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
};

struct PushConstants {
  uint32_t item_count;
  uint32_t coefficient_count;
  uint32_t mode;
  uint32_t base_index;
};
static_assert(sizeof(PushConstants) == 16,
              "GPU entropy push-constant ABI changed");

class AmdVulkanEntropyDecoder {
 public:
  ~AmdVulkanEntropyDecoder() { Shutdown(); }

  bool Available() { return Initialize(); }

  bool Reconstruct(j_decompress_ptr cinfo, const PreparedEntropyWork& work) {
    if (!Initialize() || cinfo == nullptr || cinfo->master == nullptr) {
      return false;
    }
    jpeg_decomp_master* m = cinfo->master;
    const size_t task_count = work.tasks.size() / 2;
    const size_t coefficient_bytes = work.coefficient_count * sizeof(int16_t);
    const size_t entropy_words =
        (m->amd_decode_entropy_bytes_.size() + sizeof(uint32_t) - 1) /
        sizeof(uint32_t);
    if (task_count == 0 || task_count > std::numeric_limits<uint32_t>::max() ||
        work.coefficient_count == 0 ||
        work.coefficient_count > std::numeric_limits<uint32_t>::max() ||
        entropy_words > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    if (!EnsureBuffer(&entropy_buffer_, entropy_words * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      /*prefer_device_local=*/false) ||
        !EnsureBuffer(&task_buffer_, work.tasks.size() * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      /*prefer_device_local=*/false) ||
        !EnsureBuffer(&segment_buffer_, work.segments.size() * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      /*prefer_device_local=*/false) ||
        !EnsureBuffer(&lut_buffer_, work.luts.size() * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      /*prefer_device_local=*/false) ||
        !EnsureBuffer(&coefficient_buffer_, coefficient_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      /*prefer_device_local=*/true) ||
        !EnsureBuffer(&packed_buffer_, coefficient_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      /*prefer_device_local=*/false) ||
        !EnsureBuffer(&status_buffer_, task_count * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      /*prefer_device_local=*/false) ||
        !UpdateDescriptors()) {
      return false;
    }

    {
      DecodeStageProfileTimer timer(
          m->decode_stage_profile, JPEGLI_DECODE_STAGE_PARALLEL_ENTROPY_UPLOAD);
      // Both deployment targets are little-endian x86 APUs. The shader's byte
      // extractor deliberately matches a raw memcpy into uint32 words.
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
      return false;
#else
      memset(entropy_buffer_.mapped, 0, entropy_words * sizeof(uint32_t));
      memcpy(entropy_buffer_.mapped, m->amd_decode_entropy_bytes_.data(),
             m->amd_decode_entropy_bytes_.size());
#endif
      memcpy(task_buffer_.mapped, work.tasks.data(),
             work.tasks.size() * sizeof(uint32_t));
      memcpy(segment_buffer_.mapped, work.segments.data(),
             work.segments.size() * sizeof(uint32_t));
      memcpy(lut_buffer_.mapped, work.luts.data(),
             work.luts.size() * sizeof(uint32_t));
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
    vkCmdFillBuffer(command_buffer_, coefficient_buffer_.buffer, 0,
                    coefficient_bytes, 0);
    vkCmdFillBuffer(command_buffer_, status_buffer_.buffer, 0,
                    task_count * sizeof(uint32_t), 0);
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
    if (!RecordDispatches(/*mode=*/0, static_cast<uint32_t>(task_count),
                          static_cast<uint32_t>(work.coefficient_count))) {
      vkEndCommandBuffer(command_buffer_);
      return false;
    }
    VkMemoryBarrier parse_to_copy = {};
    parse_to_copy.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    parse_to_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    parse_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &parse_to_copy,
                         0, nullptr, 0, nullptr);
    VkBufferCopy coefficient_copy = {0, 0, coefficient_bytes};
    vkCmdCopyBuffer(command_buffer_, coefficient_buffer_.buffer,
                    packed_buffer_.buffer, 1, &coefficient_copy);
    VkMemoryBarrier copy_to_host = {};
    copy_to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    copy_to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copy_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &copy_to_host, 0,
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
          JPEGLI_DECODE_STAGE_PARALLEL_ENTROPY_GPU_PARSE);
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
        uint64_t delta = timestamps[1] - timestamps[0];
        if (timestamp_valid_bits_ < 64) {
          delta &= (uint64_t{1} << timestamp_valid_bits_) - 1;
        }
        device_ms = static_cast<double>(delta) *
                    properties_.limits.timestampPeriod / 1.0e6;
      }
    }
    if (m->decode_stage_profile != nullptr) {
      m->decode_stage_profile->parallel_entropy_gpu_device_ns =
          static_cast<uint64_t>(device_ms * 1.0e6 + 0.5);
    }

    {
      DecodeStageProfileTimer timer(
          m->decode_stage_profile,
          JPEGLI_DECODE_STAGE_PARALLEL_ENTROPY_READBACK);
      const uint32_t* status =
          static_cast<const uint32_t*>(status_buffer_.mapped);
      for (size_t task = 0; task < task_count; ++task) {
        if (status[task] != 0) {
          if (TraceEnabled()) {
            fprintf(stderr,
                    "jpegli amd-vulkan entropy: GPU task %zu returned status "
                    "%u\n",
                    task, status[task]);
          }
          return false;
        }
      }
      static_assert(sizeof(JCOEF) == sizeof(int16_t),
                    "GPU entropy output requires the libjpeg int16 ABI");
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
              "jpegli amd-vulkan entropy: %zu tasks, %zu entropy bytes, "
              "device %.3f ms, submit/wait plus readback %.3f ms\n",
              task_count, m->amd_decode_entropy_bytes_.size(), device_ms,
              submit_ms);
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
      PushConstants push = {item_count, coefficient_count, mode,
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
    application.pApplicationName = "jpegli-amd-progressive-entropy-decode";
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
      if (candidate_properties.vendorID != kAmdVendorId ||
          candidate_properties.deviceType !=
              VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        continue;
      }
      VkPhysicalDevice16BitStorageFeatures storage_16 = {};
      storage_16.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
      VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_control = {};
      subgroup_control.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
      storage_16.pNext = &subgroup_control;
      VkPhysicalDeviceFeatures2 features = {};
      features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      features.pNext = &storage_16;
      vkGetPhysicalDeviceFeatures2(candidate, &features);
      if (!features.features.shaderInt16 ||
          !storage_16.storageBuffer16BitAccess ||
          !subgroup_control.subgroupSizeControl) {
        continue;
      }
      VkPhysicalDeviceSubgroupSizeControlProperties subgroup_properties = {};
      subgroup_properties.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
      VkPhysicalDeviceProperties2 properties_2 = {};
      properties_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      properties_2.pNext = &subgroup_properties;
      vkGetPhysicalDeviceProperties2(candidate, &properties_2);
      if (subgroup_properties.minSubgroupSize > kRequiredSubgroupSize ||
          subgroup_properties.maxSubgroupSize < kRequiredSubgroupSize ||
          (subgroup_properties.requiredSubgroupSizeStages &
           VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
        continue;
      }
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
      Trace("no compatible coherent-memory AMD integrated Vulkan device");
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
    VkPhysicalDeviceFeatures enabled_features = {};
    enabled_features.shaderInt16 = VK_TRUE;
    VkPhysicalDevice16BitStorageFeatures storage_16 = {};
    storage_16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage_16.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_control = {};
    subgroup_control.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    subgroup_control.subgroupSizeControl = VK_TRUE;
    storage_16.pNext = &subgroup_control;
    device_info.pEnabledFeatures = &enabled_features;
    device_info.pNext = &storage_16;
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
      fprintf(stderr, "jpegli amd-vulkan entropy: selected %s\n",
              properties_.deviceName);
    }
    return true;
  }

  bool CreateObjects() {
    std::array<VkDescriptorSetLayoutBinding, 7> bindings = {};
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
    shader_info.codeSize = kAmdVulkanDecodeEntropySpvSize;
    shader_info.pCode =
        reinterpret_cast<const uint32_t*>(kAmdVulkanDecodeEntropySpv);
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
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size = {};
    subgroup_size.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroup_size.requiredSubgroupSize = kRequiredSubgroupSize;
    stage.pNext = &subgroup_size;
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
    pool_size.descriptorCount = bindings.size();
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
    const std::array<MappedBuffer*, 7> sources = {
        &entropy_buffer_,     &task_buffer_,   &segment_buffer_, &lut_buffer_,
        &coefficient_buffer_, &packed_buffer_, &status_buffer_};
    std::array<VkDescriptorBufferInfo, 7> buffers = {};
    std::array<VkWriteDescriptorSet, 7> writes = {};
    for (uint32_t i = 0; i < writes.size(); ++i) {
      buffers[i] = {sources[i]->buffer, 0, sources[i]->size};
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
      DestroyBuffer(&status_buffer_);
      DestroyBuffer(&packed_buffer_);
      DestroyBuffer(&coefficient_buffer_);
      DestroyBuffer(&lut_buffer_);
      DestroyBuffer(&segment_buffer_);
      DestroyBuffer(&task_buffer_);
      DestroyBuffer(&entropy_buffer_);
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
  MappedBuffer entropy_buffer_;
  MappedBuffer task_buffer_;
  MappedBuffer segment_buffer_;
  MappedBuffer lut_buffer_;
  MappedBuffer coefficient_buffer_;
  MappedBuffer packed_buffer_;
  MappedBuffer status_buffer_;
};

thread_local AmdVulkanEntropyDecoder g_entropy_decoder;

bool GpuAvailable() { return g_entropy_decoder.Available(); }
bool ReconstructOnGpu(j_decompress_ptr cinfo, const PreparedEntropyWork& work) {
  return g_entropy_decoder.Reconstruct(cinfo, work);
}

#else

bool GpuAvailable() { return false; }
bool ReconstructOnGpu(j_decompress_ptr, const PreparedEntropyWork&) {
  return false;
}

#endif

}  // namespace

bool AmdVulkanDecodeEntropyPrepare(j_decompress_ptr cinfo) {
  if (cinfo == nullptr || cinfo->master == nullptr) return false;
  jpeg_decomp_master* m = cinfo->master;
  const ParallelEntropyMode mode = RequestedMode();
  if (mode == ParallelEntropyMode::kOff || !cinfo->progressive_mode ||
      cinfo->buffered_image != FALSE || m->streaming_mode_ ||
      cinfo->num_components <= 0 || cinfo->num_components > kMaxComponents) {
    return false;
  }
  if (IsGpuMode(mode) && !GpuAvailable()) {
    Trace("GPU entropy backend unavailable; using stock decoder");
    return false;
  }
  size_t total_blocks = 0;
  for (int c = 0; c < cinfo->num_components; ++c) {
    const jpeg_component_info& comp = cinfo->comp_info[c];
    const size_t component_blocks =
        static_cast<size_t>(comp.width_in_blocks) * comp.height_in_blocks;
    m->amd_decode_entropy_component_block_offsets_[c] = total_blocks;
    if (component_blocks > std::numeric_limits<size_t>::max() - total_blocks) {
      return false;
    }
    total_blocks += component_blocks;
  }
  if (total_blocks == 0 ||
      total_blocks > std::numeric_limits<uint32_t>::max() / DCTSIZE2) {
    return false;
  }
  m->amd_decode_entropy_total_coefficients_ = total_blocks * DCTSIZE2;
  m->amd_decode_entropy_bytes_.clear();
  m->amd_decode_entropy_scans_.clear();
  m->amd_decode_entropy_gpu_ = IsGpuMode(mode);
  m->amd_decode_entropy_require_restarts_ = RequiresRestarts(mode);
  m->amd_decode_entropy_finished_ = false;
  m->amd_decode_entropy_active_ = true;
  if (TraceEnabled()) {
    fprintf(stderr,
            "jpegli amd-vulkan entropy: capturing %s progressive ranges for "
            "%s parsing\n",
            m->amd_decode_entropy_require_restarts_ ? "restart"
                                                    : "independent-scan",
            m->amd_decode_entropy_gpu_ ? "GPU" : "CPU control");
  }
  return true;
}

bool AmdVulkanDecodeEntropyActive(const jpeg_decomp_master* m) {
  return m != nullptr && m->amd_decode_entropy_active_;
}

int AmdVulkanDecodeEntropyCaptureScan(j_decompress_ptr cinfo,
                                      const uint8_t* data, size_t len,
                                      size_t* pos, size_t* bit_pos) {
  jpeg_decomp_master* m = cinfo->master;
  if (!m->amd_decode_entropy_active_ || data == nullptr || pos == nullptr ||
      bit_pos == nullptr || cinfo->comps_in_scan != 1) {
    return kNeedMoreInput;
  }
  size_t cursor = 0;
  size_t segment_start = 0;
  std::vector<std::pair<size_t, size_t>> ranges;
  int expected_restart_marker = 0;
  size_t marker_start = len;
  while (cursor < len) {
    if (data[cursor] != 0xff) {
      ++cursor;
      continue;
    }
    const size_t ff_start = cursor;
    while (cursor < len && data[cursor] == 0xff) ++cursor;
    if (cursor == len) break;
    const uint8_t marker = data[cursor];
    if (marker == 0) {
      ++cursor;
      continue;
    }
    if (marker >= 0xd0 && marker <= 0xd7) {
      if (marker != 0xd0 + expected_restart_marker) return kNeedMoreInput;
      ranges.emplace_back(segment_start, ff_start);
      expected_restart_marker = (expected_restart_marker + 1) & 7;
      ++cursor;
      segment_start = cursor;
      continue;
    }
    marker_start = ff_start;
    ranges.emplace_back(segment_start, marker_start);
    break;
  }
  if (marker_start == len) {
    *pos = 0;
    *bit_pos = 0;
    return kNeedMoreInput;
  }

  const jpeg_component_info* comp = cinfo->cur_comp_info[0];
  const uint64_t total_blocks64 =
      static_cast<uint64_t>(comp->width_in_blocks) * comp->height_in_blocks;
  if (total_blocks64 == 0 ||
      total_blocks64 > std::numeric_limits<uint32_t>::max()) {
    return kNeedMoreInput;
  }
  const uint32_t total_blocks = static_cast<uint32_t>(total_blocks64);
  const uint32_t restart_interval = cinfo->restart_interval;
  const size_t expected_segments =
      restart_interval == 0
          ? 1
          : (static_cast<size_t>(total_blocks) + restart_interval - 1) /
                restart_interval;
  if (ranges.size() != expected_segments) return kNeedMoreInput;

  AmdVulkanDecodeEntropyScan scan = {};
  scan.component = comp->component_index;
  scan.ss = cinfo->Ss;
  scan.se = cinfo->Se;
  scan.ah = cinfo->Ah;
  scan.al = cinfo->Al;
  scan.restart_interval = restart_interval;
  scan.total_blocks = total_blocks;
  scan.dc_lut.reserve(kJpegHuffmanLutSize);
  scan.ac_lut.reserve(kJpegHuffmanLutSize);
  const HuffmanTableEntry* dc =
      &m->dc_huff_lut_[comp->dc_tbl_no * kJpegHuffmanLutSize];
  const HuffmanTableEntry* ac =
      &m->ac_huff_lut_[comp->ac_tbl_no * kJpegHuffmanLutSize];
  for (size_t i = 0; i < kJpegHuffmanLutSize; ++i) {
    scan.dc_lut.push_back(PackHuffmanEntry(dc[i]));
    scan.ac_lut.push_back(PackHuffmanEntry(ac[i]));
  }
  uint32_t block_start = 0;
  for (const auto& range : ranges) {
    const size_t range_size = range.second - range.first;
    if (m->amd_decode_entropy_bytes_.size() >
            std::numeric_limits<uint32_t>::max() - range_size ||
        range_size > std::numeric_limits<uint32_t>::max()) {
      return kNeedMoreInput;
    }
    const uint32_t data_offset =
        static_cast<uint32_t>(m->amd_decode_entropy_bytes_.size());
    m->amd_decode_entropy_bytes_.insert(m->amd_decode_entropy_bytes_.end(),
                                        data + range.first,
                                        data + range.second);
    const uint32_t block_count =
        restart_interval == 0
            ? total_blocks
            : std::min<uint32_t>(restart_interval, total_blocks - block_start);
    scan.segments.push_back({data_offset, static_cast<uint32_t>(range_size),
                             block_start, block_count});
    block_start += block_count;
  }
  if (block_start != total_blocks) return kNeedMoreInput;
  m->amd_decode_entropy_scans_.push_back(std::move(scan));
  cinfo->input_iMCU_row = cinfo->total_iMCU_rows;
  *pos = marker_start;
  *bit_pos = 0;
  return JPEG_SCAN_COMPLETED;
}

bool AmdVulkanDecodeEntropyFinish(j_decompress_ptr cinfo) {
  if (cinfo == nullptr || cinfo->master == nullptr) return false;
  jpeg_decomp_master* m = cinfo->master;
  if (!m->amd_decode_entropy_active_ || m->amd_decode_entropy_finished_) {
    return true;
  }
  PreparedEntropyWork work;
  if (!BuildPreparedWork(cinfo, &work)) {
    Trace("captured scans could not be formed into aligned tasks");
    return false;
  }
  const uint64_t task_count = work.tasks.size() / 2;
  const uint64_t segment_count = work.segments.size() / kPackedSegmentWords;
  if (m->decode_stage_profile != nullptr) {
    m->decode_stage_profile->parallel_entropy_tasks = task_count;
    m->decode_stage_profile->parallel_entropy_segments = segment_count;
    m->decode_stage_profile->parallel_entropy_bytes =
        m->amd_decode_entropy_bytes_.size();
  }
  bool reconstructed = false;
  if (m->amd_decode_entropy_gpu_) {
    reconstructed = ReconstructOnGpu(cinfo, work);
  }
  if (!reconstructed) {
    reconstructed = ReconstructOnCpu(cinfo, work);
    if (m->amd_decode_entropy_gpu_) {
      Trace("GPU parse failed; used captured CPU replay");
    }
  }
  if (!reconstructed) {
    Trace("captured entropy replay failed");
    return false;
  }
  if (TraceEnabled()) {
    fprintf(stderr,
            "jpegli amd-vulkan entropy: reconstructed %zu coefficients from "
            "%llu tasks, %llu scan segments, and %zu entropy bytes\n",
            work.coefficient_count, static_cast<unsigned long long>(task_count),
            static_cast<unsigned long long>(segment_count),
            m->amd_decode_entropy_bytes_.size());
  }
  m->amd_decode_entropy_finished_ = true;
  m->amd_decode_entropy_active_ = false;
  return true;
}

}  // namespace jpegli
