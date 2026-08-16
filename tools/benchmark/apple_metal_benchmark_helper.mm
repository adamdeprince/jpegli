// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#import <Metal/Metal.h>

#include "tools/benchmark/apple_metal_benchmark_helper.h"

#include <new>

struct JpegliAppleMetalBenchmarkTarget {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLCommandBuffer> command;
  id<MTLTexture> texture;
  size_t width;
  size_t height;
};

JpegliAppleMetalBenchmarkTarget* JpegliAppleMetalBenchmarkTargetCreate(size_t width,
                                                                       size_t height) {
  @autoreleasepool {
    JpegliAppleMetalBenchmarkTarget* target = new (std::nothrow) JpegliAppleMetalBenchmarkTarget();
    if (target == nullptr) return nullptr;
    target->device = MTLCreateSystemDefaultDevice();
    target->queue = [target->device newCommandQueue];
    target->command = [target->queue commandBuffer];
    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    target->texture = [target->device newTextureWithDescriptor:descriptor];
    target->width = width;
    target->height = height;
    if (target->device == nil || target->queue == nil || target->command == nil ||
        target->texture == nil) {
      delete target;
      return nullptr;
    }
    return target;
  }
}

void* JpegliAppleMetalBenchmarkCommandBuffer(JpegliAppleMetalBenchmarkTarget* target) {
  return target == nullptr ? nullptr : (__bridge void*)target->command;
}

void* JpegliAppleMetalBenchmarkTexture(JpegliAppleMetalBenchmarkTarget* target) {
  return target == nullptr ? nullptr : (__bridge void*)target->texture;
}

bool JpegliAppleMetalBenchmarkCommitWaitAndRead(JpegliAppleMetalBenchmarkTarget* target,
                                                uint8_t* rgba, size_t rgba_row_bytes) {
  if (target == nullptr) return false;
  [target->command commit];
  [target->command waitUntilCompleted];
  if (target->command.status != MTLCommandBufferStatusCompleted) return false;
  if (rgba != nullptr) {
    [target->texture getBytes:rgba
                  bytesPerRow:rgba_row_bytes
                   fromRegion:MTLRegionMake2D(0, 0, target->width, target->height)
                  mipmapLevel:0];
  }
  return true;
}

void JpegliAppleMetalBenchmarkTargetDestroy(JpegliAppleMetalBenchmarkTarget* target) {
  delete target;
}
