# Apple M4 Max single-image decode energy results

This campaign is incorporated into the broader experimental systems
characterization in
[paper/apple_metal_decoder_characterization.tex](paper/apple_metal_decoder_characterization.tex).

This campaign measured five serial JPEG decode paths on an Apple M4 Max
(40-core GPU, 128 GiB) running macOS 26.5.1:

- CPU JPEGli
- TurboJPEG 3.2.0
- Apple ImageIO
- JPEGli Metal reconstruction with CPU-visible RGBA scanlines
- JPEGli Metal reconstruction with direct RGBA8 Metal output

The benchmark binary was built from commit `b4067165`. The run started at
2026-08-16 19:35:24 UTC and ended at 20:19:27 UTC. The machine remained on AC
power with live `powermode 2` (High Power Mode) for the whole run. `pmset`
reported no thermal or performance warning before or after the campaign.

## What the energy number means

The primary measurement is the delta of macOS `RUSAGE_INFO_V6.ri_energy_nj`
for the benchmark process. Each result below is therefore **process-attributed
energy**, not energy measured at the battery, AC adapter, or individual CPU/GPU
rails. Average attributed watts are the same energy delta divided by the
window's wall time.

This distinction matters most for ImageIO. Some large baseline ImageIO cases
reported only 0.8--3.6 attributed watts while the client waited several
milliseconds for a decode. Work performed by an Apple framework, hardware
block, driver, or service need not be billed completely to the calling
process. ImageIO's values are retained and reported, but they must not be
interpreted as whole-device joules or used to claim a physical battery-energy
win without privileged rail telemetry or an external power meter. The same
caution applies to every path, although the effect is most conspicuous for
ImageIO.

## Method

The source set contains a small and large representative of five consumer
image classes:

| Class | Small | Large |
|---|---:|---:|
| Screen/UI | 320x145 | 2048x928 |
| Standard photograph | 212x320 | 1360x2048 |
| High texture | 320x240 | 2048x1536 |
| Synthetic edges | 320x320 | 1024x1024 |
| Grayscale photograph | 320x320 | 500x500 |

JPEGli encoded all sources before timing. Six profiles covered quality,
sampling, and scan organization: quality-50 4:2:0 baseline; quality-90 4:2:0
and 4:2:2 baseline; quality-95 4:4:4 baseline; and quality-90 4:2:0 and 4:4:4
progressive.

For each profile, image, and path, the benchmark performed three warmups, then
five path-rotated measurement windows. Each window decoded the same
image serially for at least 1.5 seconds after a 250 ms settle interval. A new
decoder and its normal finish/destroy lifecycle were used for every decode.
The repetitions exist only to resolve the coarse process counter: there was no
pipelining, concurrency, worker pool, multi-stream scheduling, or more than one
image in flight.

Every Metal path was forced during measurement and checked before measurement
to match CPU JPEGli byte-for-byte. CPU JPEGli, TurboJPEG, and ImageIO were
checked for successful, dimensionally correct, opaque RGBA output; their pixel
values are not expected to be identical because their reconstruction differs.
All 1,500 requested windows produced a valid energy counter. No sample was
removed. Tables use the median of the five trials for each image/path. Paired
aggregate ratios use a geometric mean so each image/profile case has equal
weight.

## Main result

Negative percentages mean less process-attributed energy or latency. The
ImageIO column has the attribution limitation described above.

| Workload | Metal output | Energy vs CPU JPEGli | Energy vs TurboJPEG | Energy vs ImageIO | Latency vs CPU JPEGli |
|---|---|---:|---:|---:|---:|
| All 60 paired cases | CPU scanlines | -47.0% | +14.2% | +1.5% | -20.2% |
| All 60 paired cases | Direct texture | **-53.8%** | **-0.3%** | -11.4% | **-28.3%** |
| Baseline, small | CPU scanlines | -46.6% | +42.7% | -27.1% | -0.4% |
| Baseline, small | Direct texture | -54.7% | +21.1% | -38.2% | -10.9% |
| Baseline, large | CPU scanlines | -63.7% | -11.2% | +55.1% | -46.0% |
| Baseline, large | Direct texture | **-70.2%** | **-27.1%** | +27.3% | **-54.6%** |
| Progressive, small | CPU scanlines | -17.1% | +30.4% | -0.9% | +10.5% |
| Progressive, small | Direct texture | -21.3% | +23.7% | -5.9% | +7.3% |
| Progressive, large | CPU scanlines | -29.0% | +5.9% | -13.5% | -18.9% |
| Progressive, large | Direct texture | -31.8% | +1.8% | -16.9% | -22.2% |

Both Metal outputs used less attributed energy than CPU JPEGli in all 60
paired cases. Direct Metal beat TurboJPEG in 25 of 60 cases overall and in all
20 large baseline cases. Metal-to-CPU beat TurboJPEG in 19 of those 20 large
baseline cases. TurboJPEG won every progressive small case and most other
small cases. This is the consumer-relevant crossover: GPU reconstruction is
an energy and latency win for substantial baseline images, while submission
and lifecycle overhead dominate small inputs and CPU progressive scan work
limits the available gain.

The geometric mean of the per-case median attributed power was 7.15 W for CPU
JPEGli, 6.51 W for TurboJPEG, 4.31 W for ImageIO, 4.74 W for Metal-to-CPU, and
4.61 W for direct Metal. Energy per image, rather than watts alone, is the more
useful comparison because the paths run for different durations.

## Common quality-90 4:2:0 baseline case

Each cell is median `mJ/image / attributed W`. The direct path finishes with a
GPU-consumable Metal allocation and intentionally performs no CPU readback.

| Image | MP | CPU JPEGli | TurboJPEG | ImageIO | Metal-to-CPU | Direct Metal |
|---|---:|---:|---:|---:|---:|---:|
| Screen/UI, small | 0.046 | 1.108 / 7.51 | **0.340 / 6.81** | 0.776 / 6.84 | 0.569 / 3.14 | 0.490 / 2.84 |
| Standard photo, small | 0.068 | 1.836 / 6.85 | **0.746 / 6.11** | 1.416 / 6.42 | 1.044 / 3.71 | 0.887 / 3.65 |
| High texture, small | 0.077 | 1.935 / 7.15 | **0.777 / 6.39** | 1.362 / 6.66 | 1.029 / 4.02 | 0.891 / 3.85 |
| Synthetic edges, small | 0.102 | 2.326 / 7.89 | **0.643 / 7.11** | 1.450 / 7.25 | 0.838 / 3.78 | 0.664 / 3.39 |
| Grayscale photo, small | 0.102 | 1.837 / 7.04 | 1.067 / 6.40 | 1.513 / 6.55 | 1.200 / 4.04 | **1.003 / 4.11** |
| Grayscale photo, large | 0.250 | 4.529 / 7.11 | 2.679 / 6.42 | 3.285 / 6.52 | 2.607 / 4.66 | **2.219 / 4.47** |
| Synthetic edges, large | 1.049 | 22.398 / 7.66 | 7.047 / 6.98 | 5.351 / 1.45 | 5.561 / 4.76 | **4.250 / 4.57** |
| Screen/UI, large | 1.901 | 38.928 / 7.47 | 10.508 / 6.79 | 7.206 / 0.93 | 8.136 / 4.49 | **5.897 / 4.51** |
| Standard photo, large | 2.785 | 70.370 / 7.18 | 30.925 / 6.49 | 12.009 / 1.40 | 27.416 / 5.20 | **23.499 / 5.17** |
| High texture, large | 3.146 | 101.575 / 7.12 | 55.309 / 6.58 | 9.349 / 0.90 | 52.780 / 5.68 | **48.560 / 5.59** |

Bold identifies the lower of TurboJPEG and direct Metal, the two paths that
were closest overall in this process counter. It deliberately does not
identify ImageIO's very low large-baseline process values as a whole-device
winner.

## Quality, sampling, and progressive scans

This table shows the direct Metal path's paired geometric-mean attributed
energy delta across all ten source images in each profile.

| JPEG profile | vs CPU JPEGli | vs TurboJPEG | vs ImageIO |
|---|---:|---:|---:|
| Q50, 4:2:0 baseline | -73.1% | -12.1% | -34.9% |
| Q90, 4:2:0 baseline | -64.2% | -9.8% | -10.0% |
| Q90, 4:2:2 baseline | -58.7% | -3.9% | -24.7% |
| Q95, 4:4:4 baseline | -54.2% | +2.3% | +40.2% |
| Q90, 4:2:0 progressive | -30.0% | +10.2% | -15.7% |
| Q90, 4:4:4 progressive | -23.3% | +14.3% | -7.3% |

The trend is consistent with the decoder design. Baseline 4:2:0 leaves a
large fraction of work in the fused GPU reconstruction stage. More chroma
coefficients and higher qualities increase entropy/coefficient work. In a
progressive file, all scan-state and entropy processing remains on the CPU,
so the reconstruction offload is a smaller fraction of total energy.

## Repeatability and policy

Median within-case energy coefficient of variation was 0.81% for CPU JPEGli,
0.92% for TurboJPEG, 0.96% for ImageIO, 0.71% for Metal-to-CPU, and 0.90% for
direct Metal. ImageIO's 95th-percentile variation was 11.3%, versus 1.9--2.4%
for the other paths. One direct-Metal small-image trial had an isolated
latency interruption; it remains in the raw data and does not change the
five-trial median.

These measurements support keeping the existing 480,000-pixel AUTO threshold
for a responsiveness-first decoder. Forced direct Metal can save attributed
energy below that boundary, but small progressive images were 7.3% slower
than CPU JPEGli and small-image TurboJPEG remained substantially faster and
usually lower-energy. Direct texture output is the preferred Metal result
when the next consumer is already on the GPU; requesting CPU scanlines adds a
copy and moved the all-case energy result from -53.8% to -47.0% versus CPU
JPEGli.

## Reproduction

The campaign driver refuses to overwrite output and requires AC power plus
live High Power Mode:

```sh
tools/benchmark/run_apple_energy_campaign.sh \
  build-metal/tools/jpegli_decode_benchmark \
  /path/to/image-directory \
  /path/to/new-output-directory
```

Each raw CSV contains trial number, decode count, wall and process CPU time,
process energy, mJ/image, mJ/MP, attributed watts, latency, and MP/s. Logs
retain the timestamped beginning and end of every measurement window. The
output directory also records the exact source checksums, commit, OS, hardware,
Metal device, start/end time, and before/after power state.
