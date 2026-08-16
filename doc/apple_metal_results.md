# Apple M4 Metal decoder results

The paper-style experimental systems characterization, including methodology,
latency, cold start, stage time, crossover, memory, energy, exactness,
perceptual results, and threats to validity, is available as LaTeX source in
[paper/apple_metal_decoder_characterization.tex](paper/apple_metal_decoder_characterization.tex).

These measurements were taken on an Apple M4 Max (40-core GPU, 128 GiB) with
macOS 26.5.1 in High Power Mode. The input corpus was the 30-image CLIC 2025
test set (1.90--4.19 megapixels per image). JPEG encoding was excluded from all
decode timings. Each path decoded one image at a time, serially, with three
warmups and 25 measured repetitions. The four paths rotated between
repetitions to reduce ordering bias. No images were pipelined or in flight
concurrently.

The timed output format was RGBA8. JPEGli encoded the inputs at quality 90,
using both 4:2:0 and 4:4:4 and both baseline and progressive scans. `ready`
ends when CPU scanlines or a completed Metal texture can be consumed; `total`
also includes finish, destroy, and exported-object release. The caller-command
path additionally records when JPEGli has only finished encoding GPU work,
before the benchmark helper commits and waits. Every cold output and all timed
Metal modes were checked byte-for-byte against CPU JPEGli. No tolerance was
accepted. The caller-command interval conservatively includes creation and
destruction of the benchmark-owned command queue, command buffer, and texture;
a real application can retain or otherwise manage those objects outside the
decode call.

## Single-image latency

Warm quality-90 4:2:0 baseline JPEG:

| Decoder path | Ready p50 | Ready p95 | Total p50 | MP/s | Process CPU p50 |
|---|---:|---:|---:|---:|---:|
| CPU JPEGli | 9.571 ms | 12.878 ms | 9.576 ms | 303.7 | 9.575 ms |
| Metal, CPU scanlines | 5.158 ms | 7.984 ms | 5.168 ms | 540.0 | 4.162 ms |
| Direct JPEGli texture | 4.476 ms | 7.212 ms | 4.496 ms | 631.2 | 3.468 ms |
| Caller command/texture, completed | 4.955 ms | 8.825 ms | 5.295 ms | 544.1 | 3.536 ms |

Relative to single-thread CPU JPEGli, pooled p50 latency fell 46.1% for
Metal-to-CPU output and 53.2% for direct texture output. The median of the 30
per-image paired speedups was 1.772x and 2.058x, respectively. The
caller-owned interval reached JPEGli's encoded-return point at 3.615 ms p50
(7.275 ms p95), including the conservative external-target setup described
above; its completed-ready figure also includes the benchmark's commit and
wait.

Warm quality-90 4:2:0 progressive JPEG:

| Decoder path | Ready p50 | Ready p95 | Total p50 | MP/s | Process CPU p50 |
|---|---:|---:|---:|---:|---:|
| CPU JPEGli | 18.982 ms | 28.606 ms | 18.989 ms | 150.7 | 18.982 ms |
| Metal, CPU scanlines | 14.779 ms | 24.213 ms | 14.790 ms | 191.5 | 13.801 ms |
| Direct JPEGli texture | 14.065 ms | 23.418 ms | 14.105 ms | 200.7 | 13.117 ms |
| Caller command/texture, completed | 14.596 ms | 24.030 ms | 15.035 ms | 194.8 | 13.179 ms |

Pooled p50 latency fell 22.1% for Metal-to-CPU and 25.9% for direct texture
output. Median paired speedups were 1.265x and 1.338x. Progressive entropy and
scan-state processing remains on the CPU, so final reconstruction is a
smaller part of total latency. Caller-command encoding completed at 13.345 ms
p50 before submission/wait.

The 4:4:4 specialization retains no float component planes and runs all three
IDCTs, color conversion, and texture output in one fused kernel:

| Coding | Path | Ready p50 | Ready p95 | MP/s | Paired CPU speedup |
|---|---|---:|---:|---:|---:|
| Baseline | CPU JPEGli | 12.393 ms | 16.359 ms | 230.9 | 1.000x |
| Baseline | Metal, CPU scanlines | 7.205 ms | 10.474 ms | 392.2 | 1.689x |
| Baseline | Direct JPEGli texture | 6.381 ms | 9.664 ms | 436.6 | 1.887x |
| Progressive | CPU JPEGli | 26.963 ms | 38.139 ms | 105.1 | 1.000x |
| Progressive | Metal, CPU scanlines | 22.061 ms | 32.751 ms | 127.9 | 1.206x |
| Progressive | Direct JPEGli texture | 21.354 ms | 31.930 ms | 132.5 | 1.255x |

Cold measurements release JPEGli's process-global Metal context before each
Metal decode. They are stricter than ordinary application reuse, but do not
flush macOS driver caches or start a new process:

| Coding | Path | Cold ready p50 | Cold ready p95 |
|---|---|---:|---:|
| Baseline 4:2:0 | CPU | 9.639 ms | 12.890 ms |
| Baseline 4:2:0 | Metal, CPU scanlines | 8.444 ms | 12.229 ms |
| Baseline 4:2:0 | Direct texture | 7.928 ms | 10.912 ms |
| Baseline 4:2:0 | Caller command/texture | 9.219 ms | 14.824 ms |
| Progressive 4:2:0 | CPU | 18.582 ms | 28.636 ms |
| Progressive 4:2:0 | Metal, CPU scanlines | 17.984 ms | 28.113 ms |
| Progressive 4:2:0 | Direct texture | 17.350 ms | 26.990 ms |
| Progressive 4:2:0 | Caller command/texture | 18.980 ms | 28.939 ms |

With the embedded precompiled library, cold Metal initialization had medians
of 0.28--0.40 ms across these paths. The largest observed initialization in
the 30-image archive pass was 4.927 ms. A build without a full Xcode Metal
toolchain retains the slower lazy runtime shader-compiler fallback.

## Where time goes

The median warm Metal-to-CPU stages for 4:2:0 were:

| Stage | Baseline | Progressive |
|---|---:|---:|
| CPU entropy/coefficient decode and exact bias accounting | 3.192 ms | 12.792 ms |
| Final bias preparation | 0.010 ms | 0.011 ms |
| Coefficient copy to Metal | 0.000 ms | 0.000 ms |
| Metal command encoding | 0.006 ms | 0.013 ms |
| Submission/wait overhead | 0.462 ms | 0.497 ms |
| Separate chroma dequantization + IDCT | 0.195 ms | 0.194 ms |
| Fused luma IDCT + fancy upsampling + color/output | 0.546 ms | 0.546 ms |
| CPU scanline copy | 0.588 ms | 0.587 ms |
| Complete Metal reconstruction | 1.248 ms | 1.308 ms |

For 4:4:4, there is no separate IDCT pass: the fully fused kernel took 0.967
ms baseline and 0.966 ms progressive. Complete Metal reconstruction was 1.740
and 1.812 ms, including submission and the requested CPU scanline copy. Final
bias preparation was 0.044 and 0.043 ms, coefficient copy remained zero, and
the float-plane allocation was zero bytes.

The parent implementation at commit `a3996fa5` used a final coefficient sweep,
a coefficient copy, and full float planes. On the same CLIC configuration its
baseline Metal-to-CPU/direct p50 was 5.351/4.670 ms and complete reconstruction
was 1.614 ms. The optimized backend measured 5.158/4.476 ms and 1.248 ms:
3.6%/4.2% lower end-to-end latency and 22.7% lower reconstruction latency.
Progressive reconstruction fell 20.8%, from 1.651 to 1.308 ms. Its later
end-to-end archive pass was 7.2% slower in absolute direct-texture time than
the earlier archive (14.065 versus 13.125 ms), while the same-run CPU decoder
also moved from 18.196 to 18.982 ms and CPU entropy from 11.535 to 12.792 ms.
That cross-run delta is therefore not presented as an isolated GPU regression;
the paired same-run result is the 25.9% latency reduction reported above.

## Crossover and memory

The deterministic crossover corpus resampled
`testdata/jxl/flower/flower.png` from 64x64 through 4096x3072. Full sweeps and a
final frozen-build boundary pass covered qualities 50, 90, and 95,
baseline/progressive coding, and 4:4:4/4:2:0 sampling. Metal-to-CPU was faster
at both p50 and p95 by 480,000 pixels in every final configuration; several
crossed at 110,592--307,200 pixels. AUTO therefore uses a conservative
480,000-pixel threshold. FORCE and both texture endpoints bypass the threshold
but retain format and working-set checks.

Median/maximum retained memory on the CLIC corpus was:

| Coding/sampling | CPU JPEGli | Metal scanline/direct | Caller-owned destination |
|---|---:|---:|---:|
| Baseline 4:2:0 | 0.62 / 0.77 MiB | 25.02 / 36.96 MiB | 14.02 / 20.96 MiB |
| Progressive 4:2:0 | 8.43 / 12.68 MiB | 25.02 / 36.96 MiB | 14.02 / 20.96 MiB |
| Baseline 4:4:4 | 0.28 / 0.34 MiB | 27.68 / 40.82 MiB | 16.68 / 24.82 MiB |
| Progressive 4:4:4 | 16.11 / 24.26 MiB | 27.68 / 40.82 MiB | 16.68 / 24.82 MiB |

The caller-owned column excludes the destination texture supplied by the
application. Median 4:2:0 scratch comprises 7.97 MiB of unified coefficients
and 5.31 MiB of chroma float planes; 4:4:4 uses 15.94 MiB of coefficients and
zero float planes. The previous Metal decoder retained 43.43 MiB median and
64.77 MiB maximum on 4:2:0, so the new shared/fused path reduced those figures
by about 42%. Only one scratch set is cached; sets above 96 MiB are released,
and a decode above the 512 MiB working-set limit falls back before choosing
full-image Metal storage.

## CPU decoder and perceptual comparison

For the same quality-90 4:2:0 files, the sum of the 30 per-image median decode
times in the original three-decoder comparison was:

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
quality 75 and above and both metrics at high-quality 4:4:4. Every accelerated
path is pixel-identical to CPU JPEGli and therefore has the same perceptual
scores.

The latency benchmark also records macOS `ri_energy_nj` around individual
decodes, but that counter updates too coarsely for many sub-20-ms samples. A
separate 44-minute campaign therefore used 1.5-second serial windows and five
trials for each image/path. Its process-attributed energy results, raw-data
method, and whole-device attribution limitations are recorded in
[apple_energy_results.md](apple_energy_results.md).
