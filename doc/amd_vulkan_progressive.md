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
batched dispatch, Vulkan timestamp-query duration, host fence-wait time, and
CPU stitching timings.

`JPEGLI_AMD_VULKAN_THROUGHPUT=1` selects the server-throughput shader schedule.
The ordinary latency path dispatches each AC scan independently. Throughput
mode groups the AC scans by component, loads each block's 64 coefficients once,
and emits every initial/refinement descriptor for that component from the same
wave64 workgroup. It does not change token layout, CPU stitching, entropy
coding, or output bytes. Leaving the variable unset or setting it to `0`,
`off`, or `false` retains the latency schedule.

Automatic mode keeps images below 4,096 total component blocks on the CPU.
On Phoenix this is approximately the conservative crossover measured between
256x256 and 512x512 4:2:0 inputs. The threshold avoids Vulkan initialization,
upload and dispatch for small images; it intentionally does not add a content
classifier to the encode path.

## Pipeline

The encoder uploads all quantized coefficient planes once into a persistently
mapped, host-cached coherent storage buffer. All initial and refinement AC
scans are then recorded into one command buffer and completed with one fence.
The Vulkan device, queue, and pipeline are retained per encoder thread. Two
pipeline slots each retain independent coefficient and descriptor buffers,
descriptor set, command buffer, fence, and timestamp query pool.

One required AMD wave64 workgroup owns one 8x8 block. In latency mode it owns
that block for one AC scan; in throughput mode it owns the block for all AC
scans of one component:

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

The fused schedule keeps an eight-word parameter record per AC scan in a small
coherent storage buffer. It reduces coefficient reads and wave launches by the
number of AC scans per component without adding another Vulkan submission or a
generic scheduling layer. The default Jpegli progressive script has four AC
scans per component, so a representative 4 MP 4:2:0 encode falls from 393,216
to 98,304 wave64 workgroups.

### Split-phase endpoint

`<jpegli_pipeline.h>` adds one experimental function:

```c
boolean jpegli_pipeline_submit(j_compress_ptr cinfo);
```

Call it after all input rows have been written and before
`jpeg_finish_compress()`. A successful call copies the final quantized
coefficients and queues GPU work without waiting. The existing finish function
waits only when it reaches that image, stitches the GPU descriptors, and runs
Huffman optimization, entropy encoding, and bit packing on the calling CPU
thread. A false result is a transparent fallback: normal finish remains valid.

The compressor must be submitted, finished or aborted on the same thread. The
endpoint accepts ordinary progressive encodes whose quantization is final; it
rejects coefficient-transcode state and PSNR-target mode because those paths
can still modify coefficients during finish.

A latency-oriented two-slot schedule is:

```cpp
prepare(slot[0], source.Next());
jpegli_pipeline_submit(&slot[0].cinfo);  // GPU image n
prepare(slot[1], source.Next());         // CPU image n + 1
jpegli_pipeline_submit(&slot[1].cinfo);  // queue GPU image n + 1
jpeg_finish_compress(&slot[0].cinfo);    // CPU entropy n; GPU runs n + 1
```

The application owns the streaming source and output sink. This avoids adding
a virtual source abstraction, buffer copy, worker thread, or scheduler to the
latency path. The accompanying C++ benchmark uses a generator-style `Next()`
source and alternates the two compressor objects.

## Memory and latency choices

Phoenix exposes a small device-local heap whose CPU mapping is uncached. Using
that heap made host stitching dominate the encode. The backend therefore
prefers host-cached, host-coherent system memory for both coefficient and
descriptor buffers. Both target APUs access this memory directly, without a
PCIe staging copy.

The implementation batches scans rather than submitting per scan, row, or
block. It retains independent descriptor regions for every scan so batching
does not alter scan ordering or token state.

## Phoenix capacity snapshot

On the current Radeon 780M / RADV Phoenix machine, quality-90 runs over the 30
decoded CLIC images measured the following. The source sequence was repeated
in memory; PNG decode is outside the timing. GPU duration comes from Vulkan
timestamp queries and busy percentage from amdgpu's sysfs counter.

| Concurrent pipeline processes | Aggregate images/s | Images/s per process | Median GPU batch | Mean GPU busy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 48.89 | 48.89 | 1.803 ms | 7.4% |
| 2 | 89.94 | 44.97 | 3.257 ms | 12.9% |
| 4 | 151.75 | 37.94 | 4.485 ms | 19.9% |

One stream therefore consumes roughly 7--9% GPU duty cycle end to end, not the
whole GPU. Each batch contains enough one-wave workgroups to occupy all CUs,
so a simultaneously resident compute job stretches batch latency even though
there is substantial idle time between JPEG batches. Two JPEG jobs lose about
8% per-job throughput while gaining 1.84x aggregate throughput; four lose about
22% per job while gaining 3.10x aggregate throughput. A sustained image-model
kernel should be expected to contend during the JPEG batch rather than run
with literally zero slowdown. The 8060S is expected to have more headroom but
still needs the same measurement on Strix Halo hardware.

## Current scope

- Initial progressive AC token formation: GPU
- Progressive AC refinement classification/compaction: GPU
- Cross-block EOB/refinement state: CPU
- Progressive DC scans: CPU
- Huffman optimization and entropy/bitstream writing: CPU

The next large opportunity is moving refinement EOB state reconciliation to a
segmented GPU scan. That is an algorithmic extension; it is not hidden behind
the current backend interface.
