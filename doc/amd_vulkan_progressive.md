# AMD Vulkan progressive tokenization

This experimental backend accelerates progressive JPEG AC scan processing on
AMD unified-memory GPUs. It is deliberately a direct Vulkan implementation,
not a generic GPU abstraction. The two deployment targets are:

- Phoenix / Radeon 780M (`gfx1103`, RDNA 3)
- Strix Halo / Radeon 8060S (`gfx1151`, RDNA 3.5)

The runtime selects only an integrated AMD Vulkan device that supports a
required 64-lane compute subgroup, subgroup ballots and arithmetic, and
host-visible coherent storage buffers. Unsupported devices retain the existing
CPU tokenizer. Phoenix is validated on the current machine; Strix Halo support
is capability-selected and still needs measurement on 8060S hardware.

## Build

The build requires Vulkan 1.3 headers/loader and `glslc`:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DJPEGLI_ENABLE_AMD_VULKAN=ON
cmake --build build --target jpeg
```

`glslc` compiles `amd_vulkan_progressive.comp` at build time. CMake embeds the
resulting SPIR-V in `libjpegli-static`; an installed shader file is not needed.

## Runtime policy

`JPEGLI_AMD_VULKAN_PROGRESSIVE` controls the backend:

- unset or `1`: automatic AMD GPU selection with the latency cutoff
- `0`, `off`, or `false`: CPU tokenizer
- `force`: GPU path regardless of image size, intended for testing

`JPEGLI_AMD_VULKAN_TRACE=1` prints device selection, coefficient upload,
batched dispatch, and CPU stitching timings.

Automatic mode keeps images below 4,096 total component blocks on the CPU.
On Phoenix this is approximately the conservative crossover measured between
256x256 and 512x512 4:2:0 inputs. The threshold avoids Vulkan initialization,
upload and dispatch for small images; it intentionally does not add a content
classifier to the encode path.

## Pipeline

The encoder uploads all quantized coefficient planes once into a persistently
mapped, host-cached coherent storage buffer. All initial and refinement AC
scans are then recorded into one command buffer and completed with one fence.
The Vulkan device, queue, pipeline, command buffer, fence and grow-only buffers
are retained per encoder thread across images.

One required AMD wave64 workgroup owns one 8x8 block:

1. Each lane loads one coefficient.
2. Subgroup ballots classify significant and future-significant coefficients.
3. Initial scans use subgroup prefix sums to place zero-run and coefficient
   tokens directly into a fixed-stride block descriptor.
4. Refinement scans compact ordered events containing coefficient position,
   new/existing state, sign and correction bit.
5. The CPU stitches cross-block EOB runs and consumes refinement events using
   Jpegli's existing state-machine semantics.
6. Existing Huffman optimization, entropy bit packing and bitstream framing
   remain on the CPU.

The fixed token/event representation is integer-only and preserves exact token
order. Adding restart markers or changing the scan script is not required.

## Memory and latency choices

Phoenix exposes a small device-local heap whose CPU mapping is uncached. Using
that heap made host stitching dominate the encode. The backend therefore
prefers host-cached, host-coherent system memory for both coefficient and
descriptor buffers. Both target APUs access this memory directly, without a
PCIe staging copy.

The implementation batches scans rather than submitting per scan, row, or
block. It retains independent descriptor regions for every scan so batching
does not alter scan ordering or token state.

## Current scope

- Initial progressive AC token formation: GPU
- Progressive AC refinement classification/compaction: GPU
- Cross-block EOB/refinement state: CPU
- Progressive DC scans: CPU
- Huffman optimization and entropy/bitstream writing: CPU

The next large opportunity is moving refinement EOB state reconciliation to a
segmented GPU scan. That is an algorithmic extension; it is not hidden behind
the current backend interface.
