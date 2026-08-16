# Experimental Apple Metal decoder

JPEGli can optionally reconstruct decoded JPEG coefficient planes with direct
Apple Metal. Marker parsing, Huffman/entropy decoding, restart handling, EOB
state, and progressive scan state remain on the CPU. Metal performs adaptive
dequantization, the 8x8 IDCT, chroma upsampling, YCbCr-to-RGB conversion, and
RGBA8 output after the required scans are complete.

This backend targets low latency for one consumer image. It does not create
workers, pipeline images, maintain multiple command streams, or schedule
concurrent decoder jobs. The reconstruction kernels use one command buffer and
two compute encoders so IDCT work is not redundantly repeated for every output
pixel. This measured faster than separate submissions on Apple M4.

## Building

The option is disabled by default and is rejected on non-Apple platforms:

```sh
cmake -S . -B build-metal \
  -DCMAKE_BUILD_TYPE=Release \
  -DJPEGLI_ENABLE_APPLE_METAL=ON
cmake --build build-metal --target jpegli-static
```

With the option off, `apple_metal.h` and its capability/policy symbols remain
available, but `jpegli_apple_metal_is_available()` returns zero and all normal
decode APIs remain CPU-only.

When a full Xcode Metal toolchain is present, the build embeds a precompiled
`metallib` by default. This avoids compiling shader source during the first
consumer decode. Command Line Tools-only builds automatically retain the lazy
runtime compiler fallback. The precompiled path can also be disabled
explicitly with `-DJPEGLI_APPLE_METAL_PRECOMPILE_SHADERS=OFF`, which is useful
for validating that fallback.

## Runtime selection

The standard scanline API defaults to `JPEGLI_APPLE_METAL_AUTO`. AUTO uses the
CPU below 786,432 output pixels, the conservative measured M4 p50/p95
crossover across baseline and progressive inputs, and uses Metal for eligible
larger images. Applications can select
`JPEGLI_APPLE_METAL_DISABLED` or `JPEGLI_APPLE_METAL_FORCE` after creating the
decompressor and before starting decompression. FORCE bypasses the size
threshold, but never bypasses a correctness or format check.

The accelerated reconstruction path currently accepts full-scale uint8
`JCS_EXT_RGBA` output from grayscale, RGB, or YCbCr JPEGs whose component
ratios cover 4:4:4, 4:2:2, or 4:2:0. It handles sequential and progressive
coding, fancy or box chroma upsampling, odd dimensions, and restart intervals.
Scaled IDCT, raw output, color quantization, CMYK/YCCK, unusual sampling
factors, and incremental buffered-image output transparently retain the CPU
renderer. Progressive entropy and scan-state processing always remain on the
CPU; buffered progressive output therefore preserves existing incremental
behavior.

JPEGli's entropy decoder stores each coefficient block in natural 8x8 order,
so this backend does not need a separate inverse-zigzag pass. Dequantization is
fused into the IDCT kernel, while upsampling and color conversion share the
second kernel.

## Direct texture output

The experimental endpoint returns both a shared `MTLBuffer` and an RGBA8Unorm
texture view. The example is Objective-C++:

```objc
#import <Metal/Metal.h>

#include <jpegli/apple_metal.h>
#include <jpegli/common.h>
#include <jpegli/decode.h>

jpeg_decompress_struct cinfo = {};
jpeg_error_mgr error = {};
cinfo.err = jpegli_std_error(&error);
jpegli_create_decompress(&cinfo);
jpegli_mem_src(&cinfo, jpeg_bytes, jpeg_size);
jpegli_read_header(&cinfo, TRUE);
cinfo.out_color_space = JCS_EXT_RGBA;

JpegliAppleMetalOutput output = {};
if (jpegli_start_decompress_to_apple_metal(&cinfo, &output)) {
  id<MTLTexture> texture = (__bridge id<MTLTexture>)output.texture;
  // Encode display/Core Image/other GPU use of texture here.
  jpegli_finish_decompress(&cinfo);
  jpegli_destroy_decompress(&cinfo);

  // output owns the texture and buffer independently of cinfo's lifetime.
  jpegli_apple_metal_release_output(&output);
} else {
  jpegli_destroy_decompress(&cinfo);
}
```

The call has the same suspension contract as `jpegli_start_decompress()` and
may be retried as source bytes arrive. It completes GPU reconstruction before
returning success. Eligible images stay in shared Apple unified memory without
a CPU pixel copy. Unsupported reconstruction modes use the authoritative CPU
renderer and, when a Metal device exists, upload the resulting RGBA bytes into
the returned shared allocation. Check `jpegli_apple_metal_was_used()` or
`JpegliAppleMetalStats.used_metal` to distinguish those cases.

`buffered_image` is the one direct-endpoint exception: it represents
incremental output rather than one completed image, so the direct call returns
`FALSE` without disturbing buffered state. Continue with
`jpegli_start_output()` and the existing CPU scanline API.

The caller must release every successful output. It is safe to finish, abort,
or destroy the decompressor first because the output retains its own Metal
objects.

## Resource and application lifecycle

Device, command queue, shader library, and pipelines are initialized lazily.
Only one scratch-buffer set is cached. Scratch larger than 96 MiB is not kept,
and a decode whose total Metal working set would exceed 512 MiB falls back to
the CPU. Output buffers are never part of the scratch cache.

Applications can respond to memory pressure or lifecycle transitions with:

```c
jpegli_apple_metal_release_cached_resources(
    JPEGLI_APPLE_METAL_RELEASE_SCRATCH);
// Or release scratch plus the inexpensive device/queue/pipeline context:
jpegli_apple_metal_release_cached_resources(JPEGLI_APPLE_METAL_RELEASE_ALL);
```

Active decoders and exported outputs remain valid while retained global state
is released.

## Correctness and profiling

The Metal shader uses precise floating-point compilation and bit-identical
constants/operation ordering for JPEGli's current float IDCT and adaptive
dequantization. Tests require byte-for-byte equality with CPU JPEGli; there is
no tolerance path. Unsupported cases fall back instead of accepting a pixel
error.

Configure the benchmark with Metal and run:

```sh
build-metal/tools/jpegli_decode_benchmark IMAGE_OR_DIRECTORY \
  --apple_metal_benchmark --iterations=25 --warmups=3 \
  --quality=90 --chroma_subsampling=420 --progressive_level=0 \
  --csv=metal-baseline.csv
```

The measured M4 results and methodology are recorded in
[apple_metal_results.md](apple_metal_results.md).

The CSV includes cold/warm ready and total latency, process CPU time and the
macOS per-process energy counter to ready, MP/s, CPU entropy time, Metal
initialization, coefficient analysis/copy, command encoding/submission, GPU
IDCT, GPU upsampling/color conversion, requested CPU output copy, and CPU,
Metal-buffer, and combined retained memory. Direct texture readiness excludes
benchmark-only validation readback. Energy is reported as unavailable when the
host kernel does not expose the current `RUSAGE_INFO_V6` counter. It is a
whole-process counter sampled around each serial decode. Its update granularity
is too coarse for some short decodes (including occasional zero deltas), so it
is raw diagnostic telemetry rather than a component-level power measurement.

M4 is the validated target. M5 has not yet been measured for crossover or
bit-exact shader behavior. A build made without the full Xcode Metal toolchain
uses lazy runtime shader compilation, which increases the first process decode
cost. Devices without timestamp counter support still decode correctly, but
the two GPU stage fields remain unavailable.
