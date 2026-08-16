// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#import <Metal/Metal.h>

#include "lib/jpegli/apple_metal.h"

#include <setjmp.h>
#include <stdlib.h>

#include <cstdint>
#include <cstring>
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

void TestErrorExit(j_common_ptr cinfo) {
  TestErrorManager* error = reinterpret_cast<TestErrorManager*>(cinfo->err);
  longjmp(error->jump_buffer, 1);
}

std::vector<uint8_t> Encode(size_t width, size_t height, bool progressive, bool subsampled) {
  jpeg_compress_struct cinfo = {};
  TestErrorManager error = {};
  unsigned char* compressed = nullptr;
  unsigned long compressed_size = 0;
  cinfo.err = jpegli_std_error(&error.pub);
  error.pub.error_exit = TestErrorExit;
  if (setjmp(error.jump_buffer)) {
    jpegli_destroy_compress(&cinfo);
    free(compressed);
    return {};
  }
  jpegli_create_compress(&cinfo);
  jpegli_mem_dest(&cinfo, &compressed, &compressed_size);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpegli_set_defaults(&cinfo);
  jpegli_set_quality(&cinfo, 83, TRUE);
  jpegli_set_progressive_level(&cinfo, progressive ? 2 : 0);
  cinfo.comp_info[0].h_samp_factor = subsampled ? 2 : 1;
  cinfo.comp_info[0].v_samp_factor = subsampled ? 2 : 1;
  cinfo.comp_info[1].h_samp_factor = 1;
  cinfo.comp_info[1].v_samp_factor = 1;
  cinfo.comp_info[2].h_samp_factor = 1;
  cinfo.comp_info[2].v_samp_factor = 1;
  std::vector<uint8_t> pixels(width * height * 3);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      const size_t p = (y * width + x) * 3;
      pixels[p + 0] = static_cast<uint8_t>((11 * x + 7 * y) & 255);
      pixels[p + 1] = static_cast<uint8_t>((3 * x + 19 * y + x * y) & 255);
      pixels[p + 2] = static_cast<uint8_t>((23 * x + 5 * y) & 255);
    }
  }
  jpegli_start_compress(&cinfo, TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW row = pixels.data() + cinfo.next_scanline * width * 3;
    if (jpegli_write_scanlines(&cinfo, &row, 1) != 1) return {};
  }
  jpegli_finish_compress(&cinfo);
  std::vector<uint8_t> result(compressed, compressed + compressed_size);
  jpegli_destroy_compress(&cinfo);
  free(compressed);
  return result;
}

std::vector<uint8_t> DecodeCpu(const std::vector<uint8_t>& jpeg) {
  jpeg_decompress_struct cinfo = {};
  TestErrorManager error = {};
  cinfo.err = jpegli_std_error(&error.pub);
  error.pub.error_exit = TestErrorExit;
  if (setjmp(error.jump_buffer)) {
    jpegli_destroy_decompress(&cinfo);
    return {};
  }
  jpegli_create_decompress(&cinfo);
  jpegli_mem_src(&cinfo, jpeg.data(), jpeg.size());
  if (jpegli_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) return {};
  cinfo.out_color_space = JCS_EXT_RGBA;
  jpegli_apple_metal_set_mode(&cinfo, JPEGLI_APPLE_METAL_DISABLED);
  if (!jpegli_start_decompress(&cinfo)) return {};
  std::vector<uint8_t> pixels(cinfo.output_width * cinfo.output_height * 4);
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row =
        pixels.data() + static_cast<size_t>(cinfo.output_scanline) * cinfo.output_width * 4;
    if (jpegli_read_scanlines(&cinfo, &row, 1) != 1) return {};
  }
  if (!jpegli_finish_decompress(&cinfo)) return {};
  jpegli_destroy_decompress(&cinfo);
  return pixels;
}

TEST(AppleMetalCommandBufferTest, ExactCallerOwnedTextureBeforeCommitAndDecoderDestroy) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  constexpr size_t kWidth = 173;
  constexpr size_t kHeight = 119;
  for (bool progressive : {false, true}) {
    for (bool subsampled : {false, true}) {
      const std::vector<uint8_t> jpeg = Encode(kWidth, kHeight, progressive, subsampled);
      ASSERT_FALSE(jpeg.empty());
      const std::vector<uint8_t> expected = DecodeCpu(jpeg);
      ASSERT_EQ(kWidth * kHeight * 4, expected.size());

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      MTLTextureDescriptor* descriptor =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                             width:kWidth
                                                            height:kHeight
                                                         mipmapped:NO];
      descriptor.storageMode = MTLStorageModeShared;
      descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
      ASSERT_NE(nil, command);
      ASSERT_NE(nil, texture);

      jpeg_decompress_struct cinfo = {};
      TestErrorManager error = {};
      cinfo.err = jpegli_std_error(&error.pub);
      error.pub.error_exit = TestErrorExit;
      ASSERT_EQ(0, setjmp(error.jump_buffer));
      jpegli_create_decompress(&cinfo);
      jpegli_mem_src(&cinfo, jpeg.data(), jpeg.size());
      ASSERT_EQ(JPEG_HEADER_OK, jpegli_read_header(&cinfo, TRUE));
      cinfo.out_color_space = JCS_EXT_RGBA;
      jpegli_apple_metal_set_mode(&cinfo, JPEGLI_APPLE_METAL_FORCE);
      JpegliAppleMetalOutput output = {};
      ASSERT_TRUE(jpegli_start_decompress_to_apple_metal_command_buffer(
          &cinfo, (__bridge void*)command, (__bridge void*)texture, &output));
      EXPECT_EQ((__bridge void*)texture, output.texture);
      EXPECT_EQ(nullptr, output.buffer);
      EXPECT_EQ(nullptr, output.buffer_contents);
      EXPECT_EQ(0u, output.row_bytes);
      JpegliAppleMetalStats stats = {};
      jpegli_apple_metal_get_stats(&cinfo, &stats);
      EXPECT_EQ(1, stats.used_metal);
      EXPECT_EQ(1, stats.caller_command_buffer);
      EXPECT_EQ(1, stats.fused_pipeline);

      // This deliberately destroys decoder-owned state before submission. The
      // output handle must keep every encoded source resource alive.
      ASSERT_TRUE(jpegli_finish_decompress(&cinfo));
      jpegli_destroy_decompress(&cinfo);
      const bool release_before_commit = progressive && subsampled;
      if (release_before_commit) {
        // A normal retained-reference command buffer must remain valid even if
        // the optional JPEGli retention handle is released before submission.
        jpegli_apple_metal_release_output(&output);
      }
      [command commit];
      [command waitUntilCompleted];
      ASSERT_EQ(MTLCommandBufferStatusCompleted, command.status);
      std::vector<uint8_t> actual(kWidth * kHeight * 4);
      [texture getBytes:actual.data()
            bytesPerRow:kWidth * 4
             fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
            mipmapLevel:0];
      EXPECT_EQ(expected, actual);
      if (!release_before_commit) jpegli_apple_metal_release_output(&output);
    }
  }
}

TEST(AppleMetalCommandBufferTest, InvalidTextureLeavesCpuFallbackUsable) {
  if (!jpegli_apple_metal_is_available()) GTEST_SKIP();
  constexpr size_t kWidth = 91;
  constexpr size_t kHeight = 67;
  const std::vector<uint8_t> jpeg =
      Encode(kWidth, kHeight, /*progressive=*/true, /*subsampled=*/true);
  const std::vector<uint8_t> expected = DecodeCpu(jpeg);
  ASSERT_FALSE(expected.empty());

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                         width:kWidth
                                                        height:kHeight
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderWrite;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];

  jpeg_decompress_struct cinfo = {};
  TestErrorManager error = {};
  cinfo.err = jpegli_std_error(&error.pub);
  error.pub.error_exit = TestErrorExit;
  ASSERT_EQ(0, setjmp(error.jump_buffer));
  jpegli_create_decompress(&cinfo);
  jpegli_mem_src(&cinfo, jpeg.data(), jpeg.size());
  ASSERT_EQ(JPEG_HEADER_OK, jpegli_read_header(&cinfo, TRUE));
  cinfo.out_color_space = JCS_EXT_RGBA;
  jpegli_apple_metal_set_mode(&cinfo, JPEGLI_APPLE_METAL_FORCE);
  JpegliAppleMetalOutput output = {};
  EXPECT_FALSE(jpegli_start_decompress_to_apple_metal_command_buffer(
      &cinfo, (__bridge void*)command, (__bridge void*)texture, &output));
  JpegliAppleMetalStats stats = {};
  jpegli_apple_metal_get_stats(&cinfo, &stats);
  EXPECT_NE(nullptr, strstr(stats.fallback_reason, "caller texture"));

  std::vector<uint8_t> actual(kWidth * kHeight * 4);
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row = actual.data() + static_cast<size_t>(cinfo.output_scanline) * kWidth * 4;
    ASSERT_EQ(1u, jpegli_read_scanlines(&cinfo, &row, 1));
  }
  EXPECT_EQ(expected, actual);
  ASSERT_TRUE(jpegli_finish_decompress(&cinfo));
  jpegli_destroy_decompress(&cinfo);
}

}  // namespace
}  // namespace jpegli
