// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jpegli/apple_metal.h"

#include <setjmp.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "lib/jpegli/common.h"
#include "lib/jpegli/decode.h"
#include "lib/jpegli/encode.h"
#include "lib/jpegli/testing.h"

namespace jpegli {
namespace {

struct TestErrorManager {
  jpeg_error_mgr pub;
  jmp_buf jump_buffer;
};

struct TestProgressManager {
  jpeg_progress_mgr pub = {};
  int calls = 0;

  TestProgressManager() { pub.progress_monitor = ProgressMonitor; }

  static void ProgressMonitor(j_common_ptr cinfo) {
    auto* progress = reinterpret_cast<TestProgressManager*>(cinfo->progress);
    ++progress->calls;
  }
};

struct SuspendingSource {
  SuspendingSource(const std::vector<uint8_t>& bytes, size_t chunk_size)
      : bytes(bytes), chunk_size(chunk_size) {
    pub.init_source = InitSource;
    pub.fill_input_buffer = FillInputBuffer;
    pub.skip_input_data = SkipInputData;
    pub.resync_to_restart = jpegli_resync_to_restart;
    pub.term_source = TermSource;
  }

  bool LoadNextChunk() {
    if (position >= bytes.size()) return false;
    const size_t buffered = pub.bytes_in_buffer;
    if (buffered != 0) {
      memmove(buffer.data(), pub.next_input_byte, buffered);
    }
    const size_t count = std::min(chunk_size, bytes.size() - position);
    buffer.resize(buffered + count);
    memcpy(buffer.data() + buffered, bytes.data() + position, count);
    pub.next_input_byte = buffer.data();
    pub.bytes_in_buffer = buffered + count;
    position += count;
    return true;
  }

  jpeg_source_mgr pub = {};
  const std::vector<uint8_t>& bytes;
  std::vector<uint8_t> buffer;
  size_t chunk_size;
  size_t position = 0;

  static void InitSource(j_decompress_ptr cinfo) {
    auto* source = reinterpret_cast<SuspendingSource*>(cinfo->src);
    source->pub.next_input_byte = nullptr;
    source->pub.bytes_in_buffer = 0;
    source->position = 0;
  }

  static boolean FillInputBuffer(j_decompress_ptr cinfo) { return FALSE; }

  static void SkipInputData(j_decompress_ptr cinfo,
                            long byte_count /* NOLINT */) {
    auto* source = reinterpret_cast<SuspendingSource*>(cinfo->src);
    if (byte_count <= 0) return;
    const size_t buffered = source->pub.bytes_in_buffer;
    if (static_cast<size_t>(byte_count) <= buffered) {
      source->pub.next_input_byte += byte_count;
      source->pub.bytes_in_buffer -= byte_count;
      return;
    }
    const size_t remaining = static_cast<size_t>(byte_count) - buffered;
    source->pub.bytes_in_buffer = 0;
    source->position =
        std::min(source->bytes.size(), source->position + remaining);
  }

  static void TermSource(j_decompress_ptr cinfo) {}
};

void TestErrorExit(j_common_ptr cinfo) {
  TestErrorManager* error = reinterpret_cast<TestErrorManager*>(cinfo->err);
  longjmp(error->jump_buffer, 1);
}

enum class TestJpegColorSpace { kGrayscale, kYCbCr, kRGB, kCMYK, kYCCK };

std::vector<uint8_t> MakePixels(size_t width, size_t height,
                                TestJpegColorSpace color_space) {
  const size_t channels =
      color_space == TestJpegColorSpace::kGrayscale
          ? 1
          : (color_space == TestJpegColorSpace::kCMYK ||
                     color_space == TestJpegColorSpace::kYCCK
                 ? 4
                 : 3);
  std::vector<uint8_t> pixels(width * height * channels);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      if (channels == 1) {
        pixels[y * width + x] =
            static_cast<uint8_t>((x * 13 + y * 29 + (x * y) / 7) & 255);
      } else {
        const size_t pos = (y * width + x) * channels;
        pixels[pos + 0] =
            static_cast<uint8_t>((x * 11 + y * 3 + (x ^ y)) & 255);
        pixels[pos + 1] =
            static_cast<uint8_t>((x * 5 + y * 17 + (x * y) / 11) & 255);
        pixels[pos + 2] =
            static_cast<uint8_t>((x * 23 + y * 7 + (x + y) / 3) & 255);
        if (channels == 4) {
          pixels[pos + 3] =
              static_cast<uint8_t>((x * 7 + y * 19 + (x * y) / 13) & 255);
        }
      }
    }
  }
  return pixels;
}

bool EncodeTestJpeg(size_t width, size_t height, int quality,
                    const char* sampling, bool progressive,
                    TestJpegColorSpace color_space, unsigned int restart,
                    std::vector<uint8_t>* encoded) {
  jpeg_compress_struct cinfo = {};
  TestErrorManager jerr = {};
  unsigned char* compressed = nullptr;
  unsigned long compressed_size = 0;
  volatile bool created = false;
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = TestErrorExit;
  if (setjmp(jerr.jump_buffer)) {
    if (created) jpegli_destroy_compress(&cinfo);
    free(compressed);
    return false;
  }
  jpegli_create_compress(&cinfo);
  created = true;
  jpegli_mem_dest(&cinfo, &compressed, &compressed_size);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components =
      color_space == TestJpegColorSpace::kGrayscale
          ? 1
          : (color_space == TestJpegColorSpace::kCMYK ||
                     color_space == TestJpegColorSpace::kYCCK
                 ? 4
                 : 3);
  cinfo.in_color_space =
      color_space == TestJpegColorSpace::kGrayscale
          ? JCS_GRAYSCALE
          : (cinfo.input_components == 4 ? JCS_CMYK : JCS_RGB);
  jpegli_set_defaults(&cinfo);
  if (color_space == TestJpegColorSpace::kRGB) {
    jpegli_set_colorspace(&cinfo, JCS_RGB);
  } else if (color_space == TestJpegColorSpace::kYCCK) {
    jpegli_set_colorspace(&cinfo, JCS_YCCK);
  }
  jpegli_set_quality(&cinfo, quality, TRUE);
  jpegli_set_progressive_level(&cinfo, progressive ? 2 : 0);
  if (color_space == TestJpegColorSpace::kYCbCr) {
    if (strcmp(sampling, "444") == 0) {
      cinfo.comp_info[0].h_samp_factor = 1;
      cinfo.comp_info[0].v_samp_factor = 1;
    } else if (strcmp(sampling, "422") == 0) {
      cinfo.comp_info[0].h_samp_factor = 2;
      cinfo.comp_info[0].v_samp_factor = 1;
    } else if (strcmp(sampling, "440") == 0) {
      cinfo.comp_info[0].h_samp_factor = 1;
      cinfo.comp_info[0].v_samp_factor = 2;
    } else if (strcmp(sampling, "411") == 0) {
      cinfo.comp_info[0].h_samp_factor = 4;
      cinfo.comp_info[0].v_samp_factor = 1;
    } else {
      cinfo.comp_info[0].h_samp_factor = 2;
      cinfo.comp_info[0].v_samp_factor = 2;
    }
    cinfo.comp_info[1].h_samp_factor = 1;
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[2].h_samp_factor = 1;
    cinfo.comp_info[2].v_samp_factor = 1;
  }
  cinfo.restart_interval = restart;
  std::vector<uint8_t> pixels = MakePixels(width, height, color_space);
  const size_t stride = width * cinfo.input_components;
  jpegli_start_compress(&cinfo, TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW row = pixels.data() + cinfo.next_scanline * stride;
    if (jpegli_write_scanlines(&cinfo, &row, 1) != 1) return false;
  }
  jpegli_finish_compress(&cinfo);
  encoded->assign(compressed, compressed + compressed_size);
  jpegli_destroy_compress(&cinfo);
  free(compressed);
  return true;
}

bool SetLumaQuantizationValue(uint8_t value, std::vector<uint8_t>* encoded) {
  if (encoded == nullptr || encoded->size() < 4 || (*encoded)[0] != 0xff ||
      (*encoded)[1] != 0xd8) {
    return false;
  }
  size_t pos = 2;
  while (pos + 4 <= encoded->size()) {
    if ((*encoded)[pos] != 0xff) return false;
    const uint8_t marker = (*encoded)[pos + 1];
    pos += 2;
    if (marker == 0xd9 || marker == 0xda) return false;
    if (marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd7)) {
      continue;
    }
    const size_t length =
        (static_cast<size_t>((*encoded)[pos]) << 8) | (*encoded)[pos + 1];
    if (length < 2 || length > encoded->size() - pos) return false;
    const size_t end = pos + length;
    if (marker == 0xdb) {
      size_t table_pos = pos + 2;
      while (table_pos < end) {
        const uint8_t table_info = (*encoded)[table_pos++];
        const size_t bytes_per_value = (table_info >> 4) == 0 ? 1 : 2;
        const size_t table_bytes = DCTSIZE2 * bytes_per_value;
        if (table_bytes > end - table_pos) return false;
        if ((table_info & 15) == 0 && bytes_per_value == 1) {
          std::fill(encoded->begin() + table_pos,
                    encoded->begin() + table_pos + DCTSIZE2, value);
          return true;
        }
        table_pos += table_bytes;
      }
    }
    pos = end;
  }
  return false;
}

bool DecodeTestJpeg(const std::vector<uint8_t>& encoded,
                    JpegliAppleMetalMode mode, bool direct,
                    std::vector<uint8_t>* pixels, JpegliAppleMetalStats* stats,
                    unsigned int scale_denom = 1, bool fancy_upsampling = true,
                    size_t crop_x = 0, size_t crop_width = 0,
                    J_COLOR_SPACE output_color_space = JCS_EXT_RGBA,
                    JpegliDataType output_data_type = JPEGLI_TYPE_UINT8,
                    JpegliAppleMetalMode entropy_mode =
                        JPEGLI_APPLE_METAL_AUTO) {
  jpeg_decompress_struct cinfo = {};
  TestErrorManager jerr = {};
  JpegliAppleMetalOutput output = {};
  volatile bool created = false;
  volatile bool exported = false;
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = TestErrorExit;
  if (setjmp(jerr.jump_buffer)) {
    if (exported) jpegli_apple_metal_release_output(&output);
    if (created) jpegli_destroy_decompress(&cinfo);
    return false;
  }
  jpegli_create_decompress(&cinfo);
  created = true;
  jpegli_apple_metal_set_mode(&cinfo, mode);
  jpegli_apple_metal_set_entropy_mode(&cinfo, entropy_mode);
  jpegli_mem_src(&cinfo, encoded.data(), encoded.size());
  if (jpegli_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) return false;
  cinfo.out_color_space = output_color_space;
  cinfo.scale_num = 1;
  cinfo.scale_denom = scale_denom;
  cinfo.do_fancy_upsampling = fancy_upsampling ? TRUE : FALSE;
  jpegli_set_output_format(&cinfo, output_data_type, JPEGLI_NATIVE_ENDIAN);
  if (direct) {
    if (!jpegli_start_decompress_to_apple_metal(&cinfo, &output)) return false;
    exported = true;
    pixels->resize(output.width * output.height * 4);
    const uint8_t* source = static_cast<const uint8_t*>(output.buffer_contents);
    for (size_t y = 0; y < output.height; ++y) {
      memcpy(pixels->data() + y * output.width * 4,
             source + y * output.row_bytes, output.width * 4);
    }
  } else {
    if (!jpegli_start_decompress(&cinfo)) return false;
    if (crop_width != 0) {
      JDIMENSION xoffset = crop_x;
      JDIMENSION width = crop_width;
      jpegli_crop_scanline(&cinfo, &xoffset, &width);
    }
    const size_t bytes_per_sample = jpegli_bytes_per_sample(output_data_type);
    pixels->resize(static_cast<size_t>(cinfo.output_width) *
                   cinfo.output_height * cinfo.output_components *
                   bytes_per_sample);
    const size_t stride = static_cast<size_t>(cinfo.output_width) *
                          cinfo.output_components * bytes_per_sample;
    while (cinfo.output_scanline < cinfo.output_height) {
      JSAMPROW row = pixels->data() + cinfo.output_scanline * stride;
      if (jpegli_read_scanlines(&cinfo, &row, 1) != 1) return false;
    }
  }
  if (!jpegli_finish_decompress(&cinfo)) return false;
  jpegli_apple_metal_get_stats(&cinfo, stats);
  if (exported) jpegli_apple_metal_release_output(&output);
  jpegli_destroy_decompress(&cinfo);
  return true;
}

bool DecodeTestJpegWithEntropyMode(
    const std::vector<uint8_t>& encoded, JpegliAppleMetalMode mode,
    bool direct, JpegliAppleMetalMode entropy_mode,
    std::vector<uint8_t>* pixels, JpegliAppleMetalStats* stats) {
  return DecodeTestJpeg(encoded, mode, direct, pixels, stats,
                        /*scale_denom=*/1, /*fancy_upsampling=*/true,
                        /*crop_x=*/0, /*crop_width=*/0, JCS_EXT_RGBA,
                        JPEGLI_TYPE_UINT8, entropy_mode);
}

bool DecodeSuspended(const std::vector<uint8_t>& encoded,
                     JpegliAppleMetalMode mode, std::vector<uint8_t>* pixels,
                     JpegliAppleMetalStats* stats, bool direct = false,
                     unsigned int scale_denom = 1) {
  jpeg_decompress_struct cinfo = {};
  TestErrorManager jerr = {};
  JpegliAppleMetalOutput output = {};
  SuspendingSource source(encoded, 97);
  volatile bool created = false;
  volatile bool exported = false;
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = TestErrorExit;
  if (setjmp(jerr.jump_buffer)) {
    if (exported) jpegli_apple_metal_release_output(&output);
    if (created) jpegli_destroy_decompress(&cinfo);
    return false;
  }
  jpegli_create_decompress(&cinfo);
  created = true;
  jpegli_apple_metal_set_mode(&cinfo, mode);
  cinfo.src = &source.pub;
  while (jpegli_read_header(&cinfo, TRUE) == JPEG_SUSPENDED) {
    if (!source.LoadNextChunk()) return false;
  }
  cinfo.out_color_space = JCS_EXT_RGBA;
  cinfo.scale_num = 1;
  cinfo.scale_denom = scale_denom;
  if (direct) {
    while (!jpegli_start_decompress_to_apple_metal(&cinfo, &output)) {
      if (!source.LoadNextChunk()) return false;
    }
    exported = true;
    pixels->resize(output.width * output.height * 4);
    const uint8_t* source_pixels =
        static_cast<const uint8_t*>(output.buffer_contents);
    for (size_t y = 0; y < output.height; ++y) {
      memcpy(pixels->data() + y * output.width * 4,
             source_pixels + y * output.row_bytes, output.width * 4);
    }
  } else {
    while (!jpegli_start_decompress(&cinfo)) {
      if (!source.LoadNextChunk()) return false;
    }
    const size_t stride = static_cast<size_t>(cinfo.output_width) * 4;
    pixels->resize(stride * cinfo.output_height);
    while (cinfo.output_scanline < cinfo.output_height) {
      JSAMPROW row = pixels->data() + cinfo.output_scanline * stride;
      if (jpegli_read_scanlines(&cinfo, &row, 1) == 0 &&
          !source.LoadNextChunk()) {
        return false;
      }
    }
  }
  while (!jpegli_finish_decompress(&cinfo)) {
    if (!source.LoadNextChunk()) return false;
  }
  jpegli_apple_metal_get_stats(&cinfo, stats);
  if (exported) jpegli_apple_metal_release_output(&output);
  jpegli_destroy_decompress(&cinfo);
  return true;
}

TEST(AppleMetalTest, ExactBaselineProgressiveSamplingQualityAndRestart) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  for (int quality : {50, 95, 100}) {
    for (const char* sampling : {"444", "440", "422", "420"}) {
      for (bool progressive : {false, true}) {
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(EncodeTestJpeg(257, 193, quality, sampling, progressive,
                                   TestJpegColorSpace::kYCbCr,
                                   /*restart=*/7, &encoded));
        std::vector<uint8_t> cpu;
        std::vector<uint8_t> metal;
        std::vector<uint8_t> direct;
        JpegliAppleMetalStats cpu_stats = {};
        JpegliAppleMetalStats metal_stats = {};
        JpegliAppleMetalStats direct_stats = {};
        ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false,
                                   &cpu, &cpu_stats));
        ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, false,
                                   &metal, &metal_stats));
        ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, true,
                                   &direct, &direct_stats));
        EXPECT_EQ(cpu, metal) << sampling << " q" << quality;
        EXPECT_EQ(cpu, direct) << sampling << " q" << quality;
        EXPECT_EQ(1, metal_stats.used_metal);
        EXPECT_EQ(0, metal_stats.used_gpu_entropy)
            << sampling << " q" << quality << " progressive=" << progressive;
        EXPECT_EQ(0, metal_stats.direct_output);
        EXPECT_EQ(0u, metal_stats.coefficient_copy_ns);
        EXPECT_GT(metal_stats.unified_coefficient_bytes, 0u);
        EXPECT_EQ(strcmp(sampling, "440") == 0 ? 0 : 1,
                  metal_stats.fused_pipeline);
        if (strcmp(sampling, "444") == 0) {
          EXPECT_EQ(0u, metal_stats.float_plane_bytes);
        } else {
          EXPECT_GT(metal_stats.float_plane_bytes, 0u);
        }
        EXPECT_EQ(1, direct_stats.used_metal);
        EXPECT_EQ(0, direct_stats.used_gpu_entropy);
        EXPECT_EQ(1, direct_stats.direct_output);
      }
    }
  }
}

TEST(AppleMetalTest, ExactSelfSynchronizingBaselineEntropy) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  for (int quality : {25, 90, 100}) {
    for (const char* sampling : {"444", "422", "420"}) {
      std::vector<uint8_t> encoded;
      ASSERT_TRUE(EncodeTestJpeg(257, 193, quality, sampling,
                                 /*progressive=*/false,
                                 TestJpegColorSpace::kYCbCr,
                                 /*restart=*/0, &encoded));
      std::vector<uint8_t> cpu;
      std::vector<uint8_t> metal;
      std::vector<uint8_t> direct;
      JpegliAppleMetalStats stats = {};
      ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED,
                                 false, &cpu, &stats));
      ASSERT_TRUE(DecodeTestJpegWithEntropyMode(
          encoded, JPEGLI_APPLE_METAL_FORCE, false,
          JPEGLI_APPLE_METAL_FORCE, &metal, &stats));
      EXPECT_EQ(cpu, metal) << sampling << " q" << quality;
      EXPECT_EQ(1, stats.used_gpu_entropy);
      ASSERT_TRUE(DecodeTestJpegWithEntropyMode(
          encoded, JPEGLI_APPLE_METAL_FORCE, true,
          JPEGLI_APPLE_METAL_FORCE, &direct, &stats));
      EXPECT_EQ(cpu, direct) << sampling << " q" << quality;
      EXPECT_EQ(1, stats.used_gpu_entropy);
    }
  }

  for (TestJpegColorSpace color_space :
       {TestJpegColorSpace::kGrayscale, TestJpegColorSpace::kRGB}) {
    std::vector<uint8_t> encoded;
    ASSERT_TRUE(EncodeTestJpeg(259, 195, 92, "444",
                               /*progressive=*/false, color_space,
                               /*restart=*/0, &encoded));
    std::vector<uint8_t> cpu;
    std::vector<uint8_t> metal;
    JpegliAppleMetalStats stats = {};
    ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false,
                               &cpu, &stats));
    ASSERT_TRUE(DecodeTestJpegWithEntropyMode(
        encoded, JPEGLI_APPLE_METAL_FORCE, false,
        JPEGLI_APPLE_METAL_FORCE, &metal, &stats));
    EXPECT_EQ(cpu, metal);
    EXPECT_EQ(color_space == TestJpegColorSpace::kGrayscale ? 1 : 0,
              stats.used_gpu_entropy);
  }
}

TEST(AppleMetalTest, EntropyModeRoundTrips) {
  jpeg_decompress_struct cinfo = {};
  jpeg_error_mgr jerr = {};
  cinfo.err = jpegli_std_error(&jerr);
  jpegli_create_decompress(&cinfo);
  EXPECT_EQ(JPEGLI_APPLE_METAL_AUTO,
            jpegli_apple_metal_get_entropy_mode(&cinfo));
  jpegli_apple_metal_set_entropy_mode(&cinfo, JPEGLI_APPLE_METAL_FORCE);
  EXPECT_EQ(JPEGLI_APPLE_METAL_FORCE,
            jpegli_apple_metal_get_entropy_mode(&cinfo));
  jpegli_apple_metal_set_entropy_mode(&cinfo,
                                      JPEGLI_APPLE_METAL_DISABLED);
  EXPECT_EQ(JPEGLI_APPLE_METAL_DISABLED,
            jpegli_apple_metal_get_entropy_mode(&cinfo));
  jpegli_destroy_decompress(&cinfo);
}

TEST(AppleMetalTest, EntropyClassifierUsesPixelsDensityAndQuantization) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  const size_t pixels = static_cast<size_t>(1537) * 1025;
  std::vector<uint8_t> selected_density;
  std::vector<uint8_t> excessive_density;
  std::vector<uint8_t> coarse_quantization;
  std::vector<uint8_t> direct_sparse;
  std::vector<uint8_t> low_density;
  ASSERT_TRUE(EncodeTestJpeg(1537, 1025, 75, "420",
                             /*progressive=*/false,
                             TestJpegColorSpace::kYCbCr,
                             /*restart=*/0, &selected_density));
  ASSERT_TRUE(SetLumaQuantizationValue(1, &selected_density));
  ASSERT_TRUE(EncodeTestJpeg(1537, 1025, 100, "420",
                             /*progressive=*/false,
                             TestJpegColorSpace::kYCbCr,
                             /*restart=*/0, &excessive_density));
  coarse_quantization = selected_density;
  ASSERT_TRUE(SetLumaQuantizationValue(8, &coarse_quantization));
  ASSERT_TRUE(EncodeTestJpeg(1537, 1025, 40, "420",
                             /*progressive=*/false,
                             TestJpegColorSpace::kYCbCr,
                             /*restart=*/0, &direct_sparse));
  ASSERT_TRUE(SetLumaQuantizationValue(1, &direct_sparse));
  ASSERT_TRUE(EncodeTestJpeg(1537, 1025, 1, "420",
                             /*progressive=*/false,
                             TestJpegColorSpace::kYCbCr,
                             /*restart=*/0, &low_density));
  ASSERT_GE(selected_density.size() * 32, pixels * 7);
  ASSERT_LE(selected_density.size() * 16, pixels * 11);
  ASSERT_GT(excessive_density.size() * 16, pixels * 11);
  ASSERT_EQ(selected_density.size(), coarse_quantization.size());
  ASSERT_GE(direct_sparse.size() * 32, pixels * 7);
  ASSERT_LT(direct_sparse.size() * 8, pixels * 3);
  ASSERT_LT(low_density.size() * 32, pixels * 7);

  std::vector<uint8_t> cpu;
  std::vector<uint8_t> metal;
  JpegliAppleMetalStats stats = {};
  ASSERT_TRUE(DecodeTestJpeg(selected_density, JPEGLI_APPLE_METAL_DISABLED,
                             false, &cpu, &stats));
  jpegli_apple_metal_release_cached_resources(JPEGLI_APPLE_METAL_RELEASE_ALL);
  ASSERT_TRUE(DecodeTestJpeg(selected_density, JPEGLI_APPLE_METAL_FORCE, false,
                             &metal, &stats));
  EXPECT_EQ(cpu, metal);
  EXPECT_EQ(0, stats.used_gpu_entropy);
  ASSERT_TRUE(DecodeTestJpeg(selected_density, JPEGLI_APPLE_METAL_FORCE, false,
                             &metal, &stats));
  EXPECT_EQ(cpu, metal);
  EXPECT_EQ(1, stats.used_gpu_entropy);

  ASSERT_TRUE(DecodeTestJpeg(coarse_quantization,
                             JPEGLI_APPLE_METAL_DISABLED, false, &cpu,
                             &stats));
  ASSERT_TRUE(DecodeTestJpeg(coarse_quantization, JPEGLI_APPLE_METAL_FORCE,
                             false, &metal, &stats));
  EXPECT_EQ(cpu, metal);
  EXPECT_EQ(0, stats.used_gpu_entropy);

  ASSERT_TRUE(DecodeTestJpeg(excessive_density,
                             JPEGLI_APPLE_METAL_DISABLED, false, &cpu,
                             &stats));
  ASSERT_TRUE(DecodeTestJpeg(excessive_density, JPEGLI_APPLE_METAL_FORCE,
                             false, &metal, &stats));
  EXPECT_EQ(cpu, metal);
  EXPECT_EQ(0, stats.used_gpu_entropy);

  ASSERT_TRUE(DecodeTestJpeg(direct_sparse, JPEGLI_APPLE_METAL_DISABLED,
                             false, &cpu, &stats));
  ASSERT_TRUE(DecodeTestJpeg(direct_sparse, JPEGLI_APPLE_METAL_FORCE, true,
                             &metal, &stats));
  EXPECT_EQ(cpu, metal);
  EXPECT_EQ(0, stats.used_gpu_entropy);

  ASSERT_TRUE(DecodeTestJpeg(low_density, JPEGLI_APPLE_METAL_DISABLED, false,
                             &cpu, &stats));
  ASSERT_TRUE(DecodeTestJpeg(low_density, JPEGLI_APPLE_METAL_FORCE, false,
                             &metal, &stats));
  EXPECT_EQ(cpu, metal);
  EXPECT_EQ(0, stats.used_gpu_entropy);
}

TEST(AppleMetalTest, ExactGrayscaleAndRgbJpeg) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  for (TestJpegColorSpace color_space :
       {TestJpegColorSpace::kGrayscale, TestJpegColorSpace::kRGB}) {
    std::vector<uint8_t> encoded;
    ASSERT_TRUE(EncodeTestJpeg(131, 99, 88, "444", true, color_space,
                               /*restart=*/5, &encoded));
    std::vector<uint8_t> cpu;
    std::vector<uint8_t> metal;
    JpegliAppleMetalStats stats = {};
    ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false,
                               &cpu, &stats));
    ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, false, &metal,
                               &stats));
    EXPECT_EQ(cpu, metal);
    EXPECT_EQ(1, stats.used_metal);
    EXPECT_EQ(0, stats.used_gpu_entropy);
    EXPECT_EQ(1, stats.fused_pipeline);
    EXPECT_EQ(0u, stats.float_plane_bytes);
  }
}

TEST(AppleMetalTest, UnsupportedFormatsUseExactCpuFallback) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  for (TestJpegColorSpace color_space :
       {TestJpegColorSpace::kCMYK, TestJpegColorSpace::kYCCK}) {
    std::vector<uint8_t> encoded;
    ASSERT_TRUE(EncodeTestJpeg(137, 103, 88, "444", true, color_space,
                               /*restart=*/5, &encoded));
    std::vector<uint8_t> cpu;
    std::vector<uint8_t> fallback;
    JpegliAppleMetalStats stats = {};
    ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false,
                               &cpu, &stats, 1, true, 0, 0, JCS_CMYK));
    ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, false,
                               &fallback, &stats, 1, true, 0, 0, JCS_CMYK));
    EXPECT_EQ(cpu, fallback);
    EXPECT_EQ(0, stats.used_metal);
    EXPECT_NE('\0', stats.fallback_reason[0]);
  }

  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(263, 197, 90, "411", false,
                             TestJpegColorSpace::kYCbCr, 3, &encoded));
  std::vector<uint8_t> cpu;
  std::vector<uint8_t> fallback;
  JpegliAppleMetalStats stats = {};
  ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false, &cpu,
                             &stats));
  ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, false,
                             &fallback, &stats));
  EXPECT_EQ(cpu, fallback);
  EXPECT_EQ(0, stats.used_metal);
  EXPECT_NE(nullptr, strstr(stats.fallback_reason, "sampling factors"));

  for (JpegliDataType data_type : {JPEGLI_TYPE_UINT16, JPEGLI_TYPE_FLOAT}) {
    ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false,
                               &cpu, &stats, 1, true, 0, 0, JCS_EXT_RGBA,
                               data_type));
    ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, false,
                               &fallback, &stats, 1, true, 0, 0, JCS_EXT_RGBA,
                               data_type));
    EXPECT_EQ(cpu, fallback);
    EXPECT_EQ(0, stats.used_metal);
    EXPECT_NE(nullptr, strstr(stats.fallback_reason, "8-bit output"));
  }
}

TEST(AppleMetalTest, RawOutputFallsBackBeforeReconstruction) {
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(129, 97, 84, "420", false,
                             TestJpegColorSpace::kYCbCr, 0, &encoded));
  jpeg_decompress_struct cinfo = {};
  TestErrorManager jerr = {};
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = TestErrorExit;
  ASSERT_EQ(0, setjmp(jerr.jump_buffer));
  jpegli_create_decompress(&cinfo);
  jpegli_mem_src(&cinfo, encoded.data(), encoded.size());
  ASSERT_EQ(JPEG_HEADER_OK, jpegli_read_header(&cinfo, TRUE));
  cinfo.raw_data_out = TRUE;
  jpegli_apple_metal_set_mode(&cinfo, JPEGLI_APPLE_METAL_FORCE);
  ASSERT_TRUE(jpegli_start_decompress(&cinfo));
  JpegliAppleMetalStats stats = {};
  jpegli_apple_metal_get_stats(&cinfo, &stats);
  EXPECT_EQ(0, stats.used_metal);
  EXPECT_NE(nullptr,
            strstr(stats.fallback_reason,
                   jpegli_apple_metal_is_available() ? "raw" : "not enabled"));
  jpegli_abort_decompress(&cinfo);
  jpegli_destroy_decompress(&cinfo);
}

TEST(AppleMetalTest, UnavailableDirectEndpointLeavesCpuDecodeUsable) {
  if (jpegli_apple_metal_is_available()) GTEST_SKIP();
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(129, 97, 84, "420", false,
                             TestJpegColorSpace::kYCbCr, 0, &encoded));
  jpeg_decompress_struct cinfo = {};
  TestErrorManager jerr = {};
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = TestErrorExit;
  ASSERT_EQ(0, setjmp(jerr.jump_buffer));
  jpegli_create_decompress(&cinfo);
  jpegli_mem_src(&cinfo, encoded.data(), encoded.size());
  ASSERT_EQ(JPEG_HEADER_OK, jpegli_read_header(&cinfo, TRUE));
  cinfo.out_color_space = JCS_EXT_RGBA;
  JpegliAppleMetalOutput output = {};
  EXPECT_FALSE(jpegli_start_decompress_to_apple_metal_command_buffer(
      &cinfo, reinterpret_cast<void*>(1), reinterpret_cast<void*>(2), &output));
  EXPECT_EQ(nullptr, output.private_handle);
  EXPECT_FALSE(jpegli_start_decompress_to_apple_metal(&cinfo, &output));
  EXPECT_EQ(nullptr, output.private_handle);
  ASSERT_TRUE(jpegli_start_decompress(&cinfo));
  std::vector<uint8_t> row(static_cast<size_t>(cinfo.output_width) * 4);
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row_pointer = row.data();
    ASSERT_EQ(1u, jpegli_read_scanlines(&cinfo, &row_pointer, 1));
  }
  ASSERT_TRUE(jpegli_finish_decompress(&cinfo));
  jpegli_destroy_decompress(&cinfo);
}

TEST(AppleMetalTest, ExactWithoutFancyUpsampling) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(255, 191, 82, "420", false,
                             TestJpegColorSpace::kYCbCr, 11, &encoded));
  std::vector<uint8_t> cpu;
  std::vector<uint8_t> metal;
  JpegliAppleMetalStats stats = {};
  ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false, &cpu,
                             &stats, 1, false));
  ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, false, &metal,
                             &stats, 1, false));
  EXPECT_EQ(cpu, metal);
  EXPECT_EQ(1, stats.used_metal);
  EXPECT_EQ(0u, stats.coefficient_copy_ns);
  EXPECT_EQ(0u, stats.float_plane_bytes);
}

TEST(AppleMetalTest, DirectScaledOutputUsesExactCpuFallback) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(259, 195, 87, "422", true,
                             TestJpegColorSpace::kYCbCr, 5, &encoded));
  std::vector<uint8_t> cpu;
  std::vector<uint8_t> direct;
  JpegliAppleMetalStats cpu_stats = {};
  JpegliAppleMetalStats direct_stats = {};
  ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false, &cpu,
                             &cpu_stats, 2));
  ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, true, &direct,
                             &direct_stats, 2));
  EXPECT_EQ(cpu, direct);
  EXPECT_EQ(0, direct_stats.used_metal);
  EXPECT_EQ(1, direct_stats.direct_output);
  EXPECT_NE(nullptr, strstr(direct_stats.fallback_reason, "scaled IDCT"));
}

TEST(AppleMetalTest, CroppedScanlineOutputRemainsExact) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(521, 389, 92, "420", false,
                             TestJpegColorSpace::kYCbCr, 0, &encoded));
  std::vector<uint8_t> cpu;
  std::vector<uint8_t> metal;
  JpegliAppleMetalStats stats = {};
  ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false, &cpu,
                             &stats, 1, true, 17, 301));
  ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, false, &metal,
                             &stats, 1, true, 17, 301));
  EXPECT_EQ(cpu, metal);
  EXPECT_EQ(1, stats.used_metal);
}

TEST(AppleMetalTest, ScanlineProgressAndImcuStateRemainCompatible) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(259, 195, 90, "420", false,
                             TestJpegColorSpace::kYCbCr, 0, &encoded));
  jpeg_decompress_struct cinfo = {};
  TestErrorManager jerr = {};
  TestProgressManager progress;
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = TestErrorExit;
  ASSERT_EQ(0, setjmp(jerr.jump_buffer));
  jpegli_create_decompress(&cinfo);
  cinfo.progress = &progress.pub;
  jpegli_mem_src(&cinfo, encoded.data(), encoded.size());
  ASSERT_EQ(JPEG_HEADER_OK, jpegli_read_header(&cinfo, TRUE));
  cinfo.out_color_space = JCS_EXT_RGBA;
  jpegli_apple_metal_set_mode(&cinfo, JPEGLI_APPLE_METAL_FORCE);
  ASSERT_TRUE(jpegli_start_decompress(&cinfo));
  std::vector<uint8_t> row(static_cast<size_t>(cinfo.output_width) * 4);
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row_pointer = row.data();
    ASSERT_EQ(1u, jpegli_read_scanlines(&cinfo, &row_pointer, 1));
  }
  EXPECT_GT(progress.calls, 0);
  EXPECT_EQ(cinfo.total_iMCU_rows, cinfo.output_iMCU_row);
  EXPECT_EQ(0u, jpegli_read_scanlines(&cinfo, nullptr, 1));
  EXPECT_EQ(1, progress.pub.completed_passes);
  EXPECT_EQ(0u, jpegli_read_scanlines(&cinfo, nullptr, 1));
  EXPECT_EQ(1, progress.pub.completed_passes);
  ASSERT_TRUE(jpegli_finish_decompress(&cinfo));
  jpegli_destroy_decompress(&cinfo);
}

TEST(AppleMetalTest, AutoCrossoverKeepsSmallImagesOnCpu) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  const size_t saved_crossover = jpegli_apple_metal_get_crossover_pixels();
  jpegli_apple_metal_set_crossover_pixels(480000);
  for (const auto& dimensions :
       {std::pair<size_t, size_t>{600, 600}, {800, 700}}) {
    std::vector<uint8_t> encoded;
    ASSERT_TRUE(EncodeTestJpeg(dimensions.first, dimensions.second, 90, "420",
                               false, TestJpegColorSpace::kYCbCr, 0, &encoded));
    std::vector<uint8_t> cpu;
    std::vector<uint8_t> automatic;
    JpegliAppleMetalStats stats = {};
    ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false,
                               &cpu, &stats));
    ASSERT_TRUE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_AUTO, false,
                               &automatic, &stats));
    EXPECT_EQ(cpu, automatic);
    EXPECT_EQ(dimensions.first * dimensions.second >= 480000,
              stats.used_metal != 0);
  }
  jpegli_apple_metal_set_crossover_pixels(saved_crossover);
}

TEST(AppleMetalTest, MalformedHeaderIsRejectedInCpuAndForcedModes) {
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(65, 49, 80, "444", false,
                             TestJpegColorSpace::kYCbCr, 0, &encoded));
  ASSERT_GE(encoded.size(), 2u);
  encoded[0] = 0;
  std::vector<uint8_t> pixels;
  JpegliAppleMetalStats stats = {};
  EXPECT_FALSE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_DISABLED, false,
                              &pixels, &stats));
  EXPECT_FALSE(DecodeTestJpeg(encoded, JPEGLI_APPLE_METAL_FORCE, false, &pixels,
                              &stats));
}

TEST(AppleMetalTest, TruncatedEntropyHasMatchingCpuAndMetalBehavior) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(267, 199, 84, "420", true,
                             TestJpegColorSpace::kYCbCr, 9, &encoded));
  for (size_t removed : {size_t{1}, size_t{17}, encoded.size() / 4}) {
    ASSERT_GT(encoded.size(), removed);
    std::vector<uint8_t> truncated(encoded.begin(), encoded.end() - removed);
    std::vector<uint8_t> cpu;
    std::vector<uint8_t> metal;
    JpegliAppleMetalStats stats = {};
    const bool cpu_ok = DecodeTestJpeg(truncated, JPEGLI_APPLE_METAL_DISABLED,
                                       false, &cpu, &stats);
    const bool metal_ok = DecodeTestJpeg(truncated, JPEGLI_APPLE_METAL_FORCE,
                                         false, &metal, &stats);
    EXPECT_EQ(cpu_ok, metal_ok) << "removed=" << removed;
    if (cpu_ok && metal_ok) EXPECT_EQ(cpu, metal) << "removed=" << removed;
  }
}

TEST(AppleMetalTest, ForcedGpuEntropyFallsBackOnTruncatedBaseline) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(1027, 769, 92, "420",
                             /*progressive=*/false,
                             TestJpegColorSpace::kYCbCr,
                             /*restart=*/0, &encoded));
  for (size_t removed : {size_t{1}, size_t{17}, encoded.size() / 4}) {
    ASSERT_GT(encoded.size(), removed);
    std::vector<uint8_t> truncated(encoded.begin(), encoded.end() - removed);
    std::vector<uint8_t> cpu;
    std::vector<uint8_t> metal;
    JpegliAppleMetalStats stats = {};
    const bool cpu_ok = DecodeTestJpegWithEntropyMode(
        truncated, JPEGLI_APPLE_METAL_DISABLED, false,
        JPEGLI_APPLE_METAL_DISABLED, &cpu, &stats);
    const bool metal_ok = DecodeTestJpegWithEntropyMode(
        truncated, JPEGLI_APPLE_METAL_FORCE, false,
        JPEGLI_APPLE_METAL_FORCE, &metal, &stats);
    EXPECT_EQ(cpu_ok, metal_ok) << "removed=" << removed;
    EXPECT_EQ(0, stats.used_gpu_entropy) << "removed=" << removed;
    if (cpu_ok && metal_ok) EXPECT_EQ(cpu, metal) << "removed=" << removed;
  }
}

TEST(AppleMetalTest, BaselineAndProgressiveInputSuspensionRemainExact) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  for (bool progressive : {false, true}) {
    std::vector<uint8_t> encoded;
    ASSERT_TRUE(EncodeTestJpeg(263, 197, 89, "420", progressive,
                               TestJpegColorSpace::kYCbCr, 13, &encoded));
    std::vector<uint8_t> cpu;
    std::vector<uint8_t> metal;
    std::vector<uint8_t> direct;
    JpegliAppleMetalStats stats = {};
    ASSERT_TRUE(
        DecodeSuspended(encoded, JPEGLI_APPLE_METAL_DISABLED, &cpu, &stats));
    ASSERT_TRUE(
        DecodeSuspended(encoded, JPEGLI_APPLE_METAL_FORCE, &metal, &stats));
    ASSERT_TRUE(DecodeSuspended(encoded, JPEGLI_APPLE_METAL_FORCE, &direct,
                                &stats, true));
    EXPECT_EQ(cpu, metal);
    EXPECT_EQ(cpu, direct);
    EXPECT_EQ(1, stats.used_metal);
    EXPECT_EQ(1, stats.direct_output);
  }
}

TEST(AppleMetalTest, SuspendedDirectCpuFallbackResumesWithoutRestarting) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  for (bool progressive : {false, true}) {
    std::vector<uint8_t> encoded;
    ASSERT_TRUE(EncodeTestJpeg(517, 389, 86, "420", progressive,
                               TestJpegColorSpace::kYCbCr, 7, &encoded));
    std::vector<uint8_t> cpu;
    std::vector<uint8_t> direct;
    JpegliAppleMetalStats stats = {};
    ASSERT_TRUE(DecodeSuspended(encoded, JPEGLI_APPLE_METAL_DISABLED, &cpu,
                                &stats, false, 2));
    ASSERT_TRUE(DecodeSuspended(encoded, JPEGLI_APPLE_METAL_FORCE, &direct,
                                &stats, true, 2));
    EXPECT_EQ(cpu, direct);
    EXPECT_EQ(0, stats.used_metal);
    EXPECT_EQ(1, stats.direct_output);
    EXPECT_NE(nullptr, strstr(stats.fallback_reason, "scaled IDCT"));
  }
}

TEST(AppleMetalTest, BufferedProgressiveFallsBackAndAbortReleasesState) {
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(129, 97, 90, "420", true,
                             TestJpegColorSpace::kYCbCr, 3, &encoded));
  jpeg_decompress_struct cinfo = {};
  TestErrorManager jerr = {};
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = TestErrorExit;
  ASSERT_EQ(0, setjmp(jerr.jump_buffer));
  jpegli_create_decompress(&cinfo);
  jpegli_mem_src(&cinfo, encoded.data(), encoded.size());
  ASSERT_EQ(JPEG_HEADER_OK, jpegli_read_header(&cinfo, TRUE));
  cinfo.out_color_space = JCS_EXT_RGBA;
  cinfo.buffered_image = TRUE;
  jpegli_apple_metal_set_mode(&cinfo, JPEGLI_APPLE_METAL_FORCE);
  ASSERT_TRUE(jpegli_start_decompress(&cinfo));
  EXPECT_EQ(0, jpegli_apple_metal_was_used(&cinfo));
  JpegliAppleMetalStats stats = {};
  jpegli_apple_metal_get_stats(&cinfo, &stats);
  EXPECT_NE(nullptr, strstr(stats.fallback_reason,
                            jpegli_apple_metal_is_available() ? "buffered-image"
                                                              : "not enabled"));
  JpegliAppleMetalOutput output = {};
  EXPECT_FALSE(jpegli_start_decompress_to_apple_metal(&cinfo, &output));
  EXPECT_EQ(nullptr, output.private_handle);
  jpegli_abort_decompress(&cinfo);
  jpegli_destroy_decompress(&cinfo);
  jpegli_apple_metal_release_cached_resources(JPEGLI_APPLE_METAL_RELEASE_ALL);
}

TEST(AppleMetalTest, DirectOutputSurvivesDecoderDestroyAndResourceRelease) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(127, 95, 91, "420", false,
                             TestJpegColorSpace::kYCbCr, 0, &encoded));
  jpeg_decompress_struct cinfo = {};
  TestErrorManager jerr = {};
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = TestErrorExit;
  ASSERT_EQ(0, setjmp(jerr.jump_buffer));
  jpegli_create_decompress(&cinfo);
  jpegli_mem_src(&cinfo, encoded.data(), encoded.size());
  ASSERT_EQ(JPEG_HEADER_OK, jpegli_read_header(&cinfo, TRUE));
  cinfo.out_color_space = JCS_EXT_RGBA;
  JpegliAppleMetalOutput output = {};
  ASSERT_TRUE(jpegli_start_decompress_to_apple_metal(&cinfo, &output));
  ASSERT_NE(nullptr, output.buffer_contents);
  const uint8_t first = static_cast<const uint8_t*>(output.buffer_contents)[0];
  jpegli_destroy_decompress(&cinfo);
  jpegli_apple_metal_release_cached_resources(JPEGLI_APPLE_METAL_RELEASE_ALL);
  EXPECT_EQ(first, static_cast<const uint8_t*>(output.buffer_contents)[0]);
  jpegli_apple_metal_release_output(&output);
  EXPECT_EQ(nullptr, output.private_handle);
}

TEST(AppleMetalTest, AbortReleasesActiveMetalDecoderState) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(EncodeTestJpeg(513, 385, 91, "420", false,
                             TestJpegColorSpace::kYCbCr, 0, &encoded));
  jpeg_decompress_struct cinfo = {};
  TestErrorManager jerr = {};
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = TestErrorExit;
  ASSERT_EQ(0, setjmp(jerr.jump_buffer));
  jpegli_create_decompress(&cinfo);
  jpegli_mem_src(&cinfo, encoded.data(), encoded.size());
  ASSERT_EQ(JPEG_HEADER_OK, jpegli_read_header(&cinfo, TRUE));
  cinfo.out_color_space = JCS_EXT_RGBA;
  jpegli_apple_metal_set_mode(&cinfo, JPEGLI_APPLE_METAL_FORCE);
  ASSERT_TRUE(jpegli_start_decompress(&cinfo));
  ASSERT_EQ(1, jpegli_apple_metal_was_used(&cinfo));
  jpegli_abort_decompress(&cinfo);
  jpegli_destroy_decompress(&cinfo);
  jpegli_apple_metal_release_cached_resources(JPEGLI_APPLE_METAL_RELEASE_ALL);
}

}  // namespace
}  // namespace jpegli
