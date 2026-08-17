#!/bin/bash

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 JPEGLI_DECODE_BENCHMARK IMAGE_DIRECTORY OUTPUT_DIRECTORY [full|ioreport|paired]" >&2
  exit 1
fi

benchmark=$1
images=$2
output=$3
scope=${4:-full}
script_directory=$(cd "$(dirname "$0")" && pwd)
repository=$(git -C "$script_directory/../.." rev-parse --show-toplevel)

if [[ "$scope" != full && "$scope" != ioreport && "$scope" != paired ]]; then
  echo "campaign scope must be full, ioreport, or paired" >&2
  exit 1
fi

if [[ ! -x "$benchmark" ]]; then
  echo "benchmark is not executable: $benchmark" >&2
  exit 1
fi
if [[ ! -d "$images" ]]; then
  echo "image directory does not exist: $images" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing output: $output" >&2
  exit 1
fi

require_high_power() {
  local power_source
  local live_mode
  power_source=$(pmset -g batt | head -1)
  live_mode=$(pmset -g live | awk '$1 == "powermode" { print $2 }')
  if [[ "$power_source" != *"AC Power"* || "$live_mode" != "2" ]]; then
    echo "campaign requires AC power and High Power Mode; current state: " \
         "$power_source, powermode ${live_mode:-unknown}" >&2
    exit 1
  fi
}

require_high_power
mkdir -p "$output"
date -u +%Y-%m-%dT%H:%M:%SZ > "$output/start-utc.txt"
sw_vers > "$output/sw-vers.txt"
sysctl hw.model hw.ncpu hw.perflevel0.physicalcpu \
  hw.perflevel1.physicalcpu > "$output/hardware.txt"
system_profiler SPDisplaysDataType > "$output/display-metal.txt"
pmset -g batt > "$output/power-before.txt"
pmset -g live >> "$output/power-before.txt"
pmset -g therm >> "$output/power-before.txt"
pmset -g custom >> "$output/power-before.txt"
git -C "$repository" rev-parse HEAD > "$output/commit.txt"
git -C "$repository" status --short --branch >> "$output/commit.txt"

for image in "$images"/*; do
  if [[ -f "$image" ]]; then
    shasum -a 256 "$image"
  fi
done > "$output/source-sha256.txt"

paths=metal_to_cpu,metal_direct
window_ms=${JPEGLI_ENERGY_WINDOW_MS:-1500}
if [[ "$scope" == paired ]]; then
  trials=${JPEGLI_ENERGY_TRIALS:-1}
else
  trials=${JPEGLI_ENERGY_TRIALS:-5}
fi
settle_ms=${JPEGLI_ENERGY_SETTLE_MS:-250}
warmups=${JPEGLI_ENERGY_WARMUPS:-3}
paired_repetitions=${JPEGLI_PAIRED_REPETITIONS:-5}

{
  echo "paths=$paths"
  echo "window_ms=$window_ms"
  echo "trials=$trials"
  echo "settle_ms=$settle_ms"
  echo "warmups=$warmups"
  echo "paired_repetitions=$paired_repetitions"
  echo "scope=$scope"
  echo "metric=serial single-image latency, process-attributed energy, IOReport CPU/GPU rail energy, effective decode delivery bandwidth, and GFX DRAM traffic"
  echo "note=JPEG and RGBA delivery bandwidth are not hardware memory-controller counters; the CSV marks bucketed PMP GFX bandwidth estimates explicitly"
} > "$output/campaign-config.txt"

run_case() {
  local entropy_mode=$1
  local quality=$2
  local sampling=$3
  local progressive=$4
  local label=$5
  local case_paths=${6:-$paths}
  require_high_power
  echo "Starting $label" >&2
  caffeinate -i "$benchmark" "$images" \
    --energy_benchmark \
    --energy_paths="$case_paths" \
    --energy_window_ms="$window_ms" \
    --energy_trials="$trials" \
    --energy_settle_ms="$settle_ms" \
    --warmups="$warmups" \
    --metal_entropy="$entropy_mode" \
    --quality="$quality" \
    --chroma_subsampling="$sampling" \
    --progressive_level="$progressive" \
    --csv="$output/$label.csv" 2>&1 | tee "$output/$label.log"
}

# The full baseline matrix spans low through lossless-adjacent entropy density
# and the three common chroma layouts. The focused IOReport matrix pairs OFF
# and AUTO around the measured selector boundary and profitable regions.
if [[ "$scope" == full ]]; then
  for entropy_mode in off force auto; do
    for quality in 25 50 75 90 95 100; do
      run_case "$entropy_mode" "$quality" 420 0 \
        "q${quality}-420-baseline-entropy-${entropy_mode}"
    done
    run_case "$entropy_mode" 90 422 0 \
      "q90-422-baseline-entropy-${entropy_mode}"
    run_case "$entropy_mode" 95 444 0 \
      "q95-444-baseline-entropy-${entropy_mode}"
  done
  run_case off 90 420 2 q90-420-progressive-entropy-off
  run_case auto 90 420 2 q90-420-progressive-entropy-auto
  run_case off 90 444 2 q90-444-progressive-entropy-off
  run_case auto 90 444 2 q90-444-progressive-entropy-auto
elif [[ "$scope" == ioreport ]]; then
  for entropy_mode in off auto; do
    run_case "$entropy_mode" 90 420 0 \
      "q90-420-baseline-entropy-${entropy_mode}"
    run_case "$entropy_mode" 90 422 0 \
      "q90-422-baseline-entropy-${entropy_mode}"
    run_case "$entropy_mode" 95 420 0 \
      "q95-420-baseline-entropy-${entropy_mode}"
    run_case "$entropy_mode" 95 444 0 \
      "q95-444-baseline-entropy-${entropy_mode}"
    run_case "$entropy_mode" 100 420 0 \
      "q100-420-baseline-entropy-${entropy_mode}"
    run_case "$entropy_mode" 90 420 2 \
      "q90-420-progressive-entropy-${entropy_mode}"
  done
else
  profiles=("90 420 0" "90 422 0" "95 420 0" "95 444 0" "100 420 0")
  for profile in "${profiles[@]}"; do
    read -r quality sampling progressive <<< "$profile"
    for path in metal_to_cpu metal_direct; do
      for ((repetition = 0; repetition < paired_repetitions; ++repetition)); do
        if ((repetition % 2 == 0)); then
          modes=(off auto)
        else
          modes=(auto off)
        fi
        for entropy_mode in "${modes[@]}"; do
          run_case "$entropy_mode" "$quality" "$sampling" "$progressive" \
            "q${quality}-${sampling}-baseline-${path}-rep${repetition}-entropy-${entropy_mode}" \
            "$path"
        done
      done
    done
  done
fi

python3 "$script_directory/analyze_apple_gpu_entropy_campaign.py" "$output" \
  | tee "$output/analysis.log"

pmset -g batt > "$output/power-after.txt"
pmset -g live >> "$output/power-after.txt"
pmset -g therm >> "$output/power-after.txt"
pmset -g custom >> "$output/power-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$output/end-utc.txt"

(
  cd "$output"
  find . -type f ! -name results-sha256.txt -print | LC_ALL=C sort |
    while IFS= read -r result; do
      shasum -a 256 "$result"
    done
) > "$output/results-sha256.txt"

echo "GPU entropy bandwidth/energy campaign complete: $output" >&2
