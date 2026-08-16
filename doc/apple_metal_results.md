# Apple M4 Metal decoder results

These measurements were taken on an Apple M4 Max (40-core GPU, 128 GiB) with
macOS 26.5.1 in High Power Mode. The input corpus was the 30-image CLIC 2025
test set (1.90--4.19 megapixels per image). JPEG encoding was excluded from all
decode timings. Each path decoded one image at a time, serially, with three
warmups and 25 measured repetitions. Paths were rotated between repetitions to
reduce ordering bias.

The timed output format was RGBA8. The JPEGs were encoded by JPEGli at quality
90 with 4:2:0 chroma. `ready` ends when CPU scanlines or the direct Metal
texture can be consumed; `total` also includes finish, destroy, and exported
object release. Every timed Metal result was checked byte-for-byte against CPU
JPEGli.

## Single-image latency

Warm baseline JPEG:

| Decoder path | Ready p50 | Ready p95 | Total p50 | MP/s | Process CPU p50 |
|---|---:|---:|---:|---:|---:|
| CPU JPEGli | 9.418 ms | 12.532 ms | 9.423 ms | 301.1 | 9.418 ms |
| Metal, CPU scanlines | 5.351 ms | 7.894 ms | 5.357 ms | 516.8 | 4.394 ms |
| Direct Metal texture | 4.671 ms | 7.156 ms | 4.697 ms | 585.4 | 3.729 ms |

Relative to single-thread CPU JPEGli, the pooled p50 improvement is 1.76x
(43.2% less latency) for Metal-to-CPU output and 2.02x (50.4% less latency)
for direct texture output. The median of the 30 per-image paired speedups is
1.70x and 1.93x, respectively.

Warm progressive JPEG:

| Decoder path | Ready p50 | Ready p95 | Total p50 | MP/s | Process CPU p50 |
|---|---:|---:|---:|---:|---:|
| CPU JPEGli | 18.465 ms | 26.214 ms | 18.471 ms | 152.5 | 18.458 ms |
| Metal, CPU scanlines | 13.794 ms | 21.643 ms | 13.802 ms | 192.6 | 12.790 ms |
| Direct Metal texture | 13.127 ms | 20.921 ms | 13.146 ms | 201.2 | 12.189 ms |

Here the pooled p50 improvement is 1.34x (25.3%) for Metal-to-CPU and 1.41x
(28.9%) for direct texture output. Entropy and progressive scan processing
remain on the CPU, so final reconstruction is a smaller share of total time.

Cold measurements release the JPEGli process-global Metal context before each
Metal decode. They are deliberately stricter than ordinary application reuse,
although they do not flush macOS driver caches or start a new process:

| Coding | Path | Cold ready p50 | Cold ready p95 |
|---|---|---:|---:|
| Baseline | CPU | 9.628 ms | 12.623 ms |
| Baseline | Metal, CPU scanlines | 9.239 ms | 11.628 ms |
| Baseline | Direct texture | 8.411 ms | 10.724 ms |
| Progressive | CPU | 18.302 ms | 25.823 ms |
| Progressive | Metal, CPU scanlines | 17.351 ms | 25.210 ms |
| Progressive | Direct texture | 17.402 ms | 25.737 ms |

An isolated first-process probe measured 2.258 ms of Metal initialization with
the embedded precompiled library. The same probe with runtime shader source
compilation measured 42.315 ms. Subsequent full context recreations had a
baseline median of 0.388 ms. Builds without a full Xcode Metal toolchain still
use the slower runtime compiler fallback.

## Where time goes

The median warm Metal-to-CPU stages over all images and repetitions were:

| Stage | Baseline | Progressive |
|---|---:|---:|
| CPU entropy/coefficient decode | 2.945 ms | 11.501 ms |
| Coefficient analysis | 0.282 ms | 0.283 ms |
| Coefficient copy to shared buffer | 0.092 ms | 0.093 ms |
| Metal command encoding | 0.006 ms | 0.010 ms |
| Submission/wait overhead | 0.507 ms | 0.528 ms |
| GPU dequantization + IDCT | 0.323 ms | 0.326 ms |
| GPU upsampling + color conversion | 0.378 ms | 0.378 ms |
| CPU scanline copy | 0.596 ms | 0.592 ms |
| Complete Metal reconstruction | 1.614 ms | 1.651 ms |

The CPU-only stage profile attributed 34.7% of baseline time to entropy,
16.3% to IDCT, 15.5% to upsampling, 13.9% to color conversion, and 15.3% to
pixel output. For progressive files, CPU entropy processing dominated at
66.0%.

## Crossover and memory

The serial size sweep covered 64x64 through 4096x3072 at representative
qualities, baseline/progressive coding, and 4:4:4/4:2:0 sampling. In the final
quality-90 4:2:0 run, Metal-to-CPU beat CPU at both p50 and p95 from 480,000
pixels for baseline files and 786,432 pixels for progressive files. AUTO
therefore uses the conservative 786,432-pixel threshold. FORCE and direct
texture output bypass the threshold but retain all correctness and working-set
checks.

On the CLIC set, baseline CPU streaming retained a median 0.69 MiB (0.77 MiB
maximum), while progressive CPU decoding retained 8.43 MiB (12.68 MiB
maximum). The Metal decoder retained 43.43 MiB median and 64.77 MiB maximum,
including the CPU coefficient store, shared scratch buffers, and output. Only
one scratch set is cached; sets above 96 MiB are released and a decode above
the 512 MiB Metal working-set limit falls back before selecting full-image
coefficient storage.

## CPU decoder and perceptual comparison

For the same quality-90 4:2:0 files, the sum of the 30 per-image median decode
times was:

| Coding | CPU JPEGli | TurboJPEG | Apple ImageIO |
|---|---:|---:|---:|
| Baseline | 288.3 ms / 299.2 MP/s | 143.4 ms / 601.6 MP/s | 214.4 ms / 402.2 MP/s |
| Progressive | 567.1 ms / 152.1 MP/s | 445.1 ms / 193.8 MP/s | 537.4 ms / 160.5 MP/s |

Thus the original CPU JPEGli decoder did not beat TurboJPEG on latency.
JPEGli's reconstruction differs perceptually because of adaptive
dequantization. Mean scores over the 30 source images are below; higher
SSIMULACRA2 and lower Butteraugli are better.

| Quality | Sampling | CPU JPEGli SSIM2 / BA | TurboJPEG SSIM2 / BA | ImageIO SSIM2 / BA |
|---:|:---:|---:|---:|---:|
| 50 | 4:2:0 | 60.171 / 4.692 | 60.631 / 4.670 | 59.895 / 5.308 |
| 50 | 4:4:4 | 63.787 / 4.109 | 64.520 / 3.979 | 64.408 / 3.985 |
| 75 | 4:2:0 | 71.984 / 3.508 | 71.774 / 3.478 | 70.788 / 4.463 |
| 75 | 4:4:4 | 74.849 / 2.843 | 74.690 / 2.731 | 74.706 / 2.730 |
| 90 | 4:2:0 | 81.996 / 2.671 | 81.295 / 2.650 | 79.726 / 3.813 |
| 90 | 4:4:4 | 84.687 / 1.643 | 84.015 / 1.654 | 83.919 / 1.679 |
| 95 | 4:2:0 | 86.673 / 2.318 | 86.027 / 2.343 | 83.923 / 3.603 |
| 95 | 4:4:4 | 89.389 / 1.054 | 88.524 / 1.147 | 88.316 / 1.189 |

There is no universal quality winner at low quality: TurboJPEG leads several
quality-50 and Butteraugli cases, while JPEGli generally leads SSIMULACRA2 at
quality 75 and above and both metrics at high-quality 4:4:4. The Metal output
is pixel-identical to CPU JPEGli and therefore has the same perceptual scores.

The benchmark also records macOS `ri_energy_nj` deltas for every serial
decode. That process counter updates too coarsely for many sub-20-ms samples,
producing zeros and delayed attribution between rotated paths. The raw values
are retained for audit, but this run does not claim a path-level energy win.
Process CPU time is stable and shows the offload benefit; defensible component
energy measurement still requires external power instrumentation or a longer
non-latency workload.
