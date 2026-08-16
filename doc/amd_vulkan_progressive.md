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

## Progressive decoder coefficient experiment

The decoder has a separate, opt-in coefficient-reconstruction experiment:

- `JPEGLI_AMD_VULKAN_DECODE_COEFFICIENTS=force` enables the GPU path.
- `JPEGLI_AMD_VULKAN_DECODE_COEFFICIENTS=cpu` uses the same event-forming
  entropy parser but reconstructs on the CPU as an architecture control.
- unset, `0`, `off`, or `false` leaves the original decoder loop unchanged.

The serial CPU Huffman parser keeps one significance and sign mask per block.
Initial values and asserted refinement bits become eight-byte
`(coefficient_index, signed_delta)` events. Since each successive-approximation
bit contributes a distinct signed power of two, the events commute. A single
GPU submission clears an int32 coefficient plane, atomically accumulates all
events from all scans, and packs pairs to Jpegli's int16 coefficient ABI.

The event input and packed output use host-cached coherent storage; the GPU-only
int32 accumulator prefers device-local coherent storage. The decoder falls back
to exact CPU event reconstruction if device initialization or submission fails.
Buffered-image progressive output remains on the stock path because it can
render incomplete scans before EOI.

This experiment is disabled by default because the measured end-to-end result
is negative. On the 86.25 MP quality-75 4:2:0 corpus, full-resolution
progressive latency was 581.85 ms on the stock path, 767.08 ms with CPU events,
and 718.13 ms with GPU events. The GPU reduced separated event finalization from
133.38 to 65.34 ms, but event formation, upload, synchronization, and readback
left the complete decoder 23.4% slower than stock. Even subtracting all readback
leaves it 21.2% slower. The retained measurements are therefore evidence that
coefficient reconstruction alone is too narrow a decoder GPU boundary.

## Parallel progressive entropy experiment

The decoder also has an opt-in parser that moves the complete fused Huffman,
run-length and coefficient-reconstruction loop onto independent tasks. It has
two schedules:

- one task per component when the file has no restart markers; and
- one task per component/block interval when every progressive scan has
  aligned restart markers.

Each task processes its coefficient interval through that component's scans in
progressive order. This preserves DC prediction, EOB runs and refinement
dependencies inside the task while making coefficient ranges independent
between tasks. No coefficient atomics or inter-task barriers are needed.

`JPEGLI_AMD_VULKAN_DECODE_ENTROPY` selects the schedule:

- `independent`: GPU component chains without restart markers;
- `restart`: GPU restart-segment chains;
- `cpu-independent`: matching CPU architecture control; and
- `cpu-restart`: threaded CPU restart-segment parser.

Unset, `0`, `off` or `false` preserves the stock loop.
`JPEGLI_AMD_VULKAN_DECODE_ENTROPY_THREADS` sets the CPU worker count.
`JPEGLI_AMD_VULKAN_TRACE=1` reports tasks, segments, entropy bytes, the Vulkan
timestamp and host wait/readback duration.

The GPU path requires an integrated coherent-memory AMD device, native 16-bit
storage and a controllable 32-lane subgroup. The shader writes int16
coefficients directly, forces RDNA wave32 for unrelated branch-heavy Huffman
streams, and uses one submission per image. These capabilities are present on
the measured Phoenix / Radeon 780M and are part of the Strix Halo / Radeon
8060S target profile. The latter still requires on-device validation.

The independent schedule exposes only three tasks for the default 4:2:0 scan
script and is not useful: a representative 2.8 MP image took 1,534.72 ms of GPU
device time. Restart interval 16 exposed enough work to reach 223.77 MP/s over
the 30-image/four-scale latency matrix, 1.338x stock without restarts, but made
the JPEG corpus 18.45% larger. At interval 32 the GPU no longer beat stock.

The threaded CPU schedule is the stronger latency result on Phoenix. Restart
interval 64 and 16 logical threads reached 308.47 MP/s, 1.845x stock without
restarts, for 4.77% more bytes. Eight threads reached 294.61 MP/s. This consumes
most of the host CPU and is therefore a latency result, not a claim about
multi-request server throughput.

The experimental encoder tool accepts `--restart_interval=N` and
`--restart_in_rows=N` to create aligned inputs. Files without restart markers
cannot acquire restart independence at decode time.

## Encoder current scope

- Initial progressive AC token formation: GPU
- Progressive AC refinement classification/compaction: GPU
- Cross-block EOB/refinement state: CPU
- Progressive DC scans: CPU
- Huffman optimization and entropy/bitstream writing: CPU

The next large opportunity is moving refinement EOB state reconciliation to a
segmented GPU scan. That is an algorithmic extension; it is not hidden behind
the current backend interface.
