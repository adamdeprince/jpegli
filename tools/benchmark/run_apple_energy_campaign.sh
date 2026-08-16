#!/bin/bash

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 JPEGLI_DECODE_BENCHMARK IMAGE_DIRECTORY OUTPUT_DIRECTORY" >&2
  exit 1
fi

benchmark=$1
images=$2
output=$3
script_directory=$(cd "$(dirname "$0")" && pwd)
repository=$(git -C "$script_directory/../.." rev-parse --show-toplevel)

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

power_source=$(pmset -g batt | head -1)
if [[ "$power_source" != *"AC Power"* ]]; then
  echo "energy campaign requires AC power; current state: $power_source" >&2
  exit 1
fi
power_profile=$(system_profiler SPPowerDataType)
if [[ "$power_profile" != *"High Power Mode: Yes"* ]]; then
  echo "energy campaign requires High Power Mode" >&2
  exit 1
fi

mkdir -p "$output"
date -u +%Y-%m-%dT%H:%M:%SZ > "$output/start-utc.txt"
sw_vers > "$output/sw-vers.txt"
sysctl hw.model hw.ncpu hw.perflevel0.physicalcpu \
  hw.perflevel1.physicalcpu > "$output/hardware.txt"
system_profiler SPDisplaysDataType > "$output/display-metal.txt"
pmset -g batt > "$output/power-before.txt"
pmset -g therm >> "$output/power-before.txt"
pmset -g custom >> "$output/power-before.txt"
git -C "$repository" rev-parse HEAD > "$output/commit.txt"
git -C "$repository" status --short --branch >> "$output/commit.txt"

for image in "$images"/*; do
  if [[ -f "$image" ]]; then
    shasum -a 256 "$image"
  fi
done > "$output/source-sha256.txt"

paths=jpegli,turbojpeg,apple_imageio,metal_to_cpu,metal_direct
window_ms=1500
trials=5
settle_ms=250
warmups=3

run_case() {
  local quality=$1
  local sampling=$2
  local progressive=$3
  local label=$4
  echo "Starting $label" >&2
  caffeinate -i "$benchmark" "$images" \
    --energy_benchmark \
    --energy_paths="$paths" \
    --energy_window_ms="$window_ms" \
    --energy_trials="$trials" \
    --energy_settle_ms="$settle_ms" \
    --warmups="$warmups" \
    --quality="$quality" \
    --chroma_subsampling="$sampling" \
    --progressive_level="$progressive" \
    --csv="$output/$label.csv" 2>&1 | tee "$output/$label.log"
}

# Low, common, and high quality; sequential/progressive; and all common color
# sampling layouts. JPEG encoding occurs before and outside every energy
# window.
run_case 50 420 0 q50-420-baseline
run_case 90 420 0 q90-420-baseline
run_case 90 422 0 q90-422-baseline
run_case 95 444 0 q95-444-baseline
run_case 90 420 2 q90-420-progressive
run_case 90 444 2 q90-444-progressive

pmset -g batt > "$output/power-after.txt"
pmset -g therm >> "$output/power-after.txt"
pmset -g custom >> "$output/power-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$output/end-utc.txt"

echo "Energy campaign complete: $output" >&2
