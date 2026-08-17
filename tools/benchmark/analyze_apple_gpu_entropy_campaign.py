#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Summarizes paired OFF/FORCE/AUTO Apple Metal entropy measurements."""

import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict


METRICS = (
    "latency_ms_per_decode",
    "energy_mj_per_decode",
    "compressed_input_mb_per_second",
    "rgba_output_gb_per_second",
    "process_cpu_ms_per_decode",
    "cpu_rail_energy_mj_per_decode",
    "average_cpu_rail_power_w",
    "gpu_rail_energy_mj_per_decode",
    "average_gpu_rail_power_w",
    "cpu_gpu_rail_energy_mj_per_decode",
    "average_cpu_gpu_rail_power_w",
    "gpu_dram_total_gb_per_second",
)

METRIC_AVAILABILITY = {
    "gpu_rail_energy_mj_per_decode": "gpu_rail_energy_available",
    "average_gpu_rail_power_w": "gpu_rail_energy_available",
    "cpu_rail_energy_mj_per_decode": "cpu_rail_energy_available",
    "average_cpu_rail_power_w": "cpu_rail_energy_available",
    "cpu_gpu_rail_energy_mj_per_decode": "cpu_gpu_rail_energy_available",
    "average_cpu_gpu_rail_power_w": "cpu_gpu_rail_energy_available",
    "gpu_dram_total_gb_per_second": "gpu_dram_bandwidth_available",
}


def geometric_mean(values):
    if not values or any(value <= 0 for value in values):
        return float("nan")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def format_ratio(value):
    return "n/a" if not math.isfinite(value) else f"{(value - 1) * 100:+.1f}%"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} CAMPAIGN_DIRECTORY", file=sys.stderr)
        return 1
    directory = os.path.abspath(sys.argv[1])
    csv_paths = sorted(
        path
        for path in glob.glob(os.path.join(directory, "*.csv"))
        if not path.endswith(("paired.csv", "summary.csv"))
    )
    if not csv_paths:
        print(f"no campaign CSV files in {directory}", file=sys.stderr)
        return 1

    trials = defaultdict(list)
    for path in csv_paths:
        with open(path, newline="") as source:
            for row in csv.DictReader(source):
                mode = row["requested_metal_entropy_mode"]
                key = (
                    row["image"],
                    row["path"],
                    row["quality"],
                    row["chroma_subsampling"],
                    row["progressive_level"],
                    mode,
                )
                if row["energy_available"] != "1":
                    raise ValueError(f"missing energy sample in {path}: {key}")
                trials[key].append(row)

    medians = {}
    for key, rows in trials.items():
        value = {}
        for metric in METRICS:
            availability = METRIC_AVAILABILITY.get(metric)
            samples = [
                float(row[metric])
                for row in rows
                if metric in row
                and (availability is None or row.get(availability) == "1")
            ]
            value[metric] = (
                statistics.median(samples) if samples else float("nan")
            )
        value["used_gpu_entropy"] = int(
            statistics.median(int(row["used_gpu_entropy"]) for row in rows)
        )
        value["gpu_dram_bandwidth_estimated"] = any(
            row.get("gpu_dram_bandwidth_estimated") == "1" for row in rows
        )
        value["image_type"] = rows[0]["image_type"]
        value["pixels"] = int(rows[0]["pixels"])
        value["bits_per_pixel"] = float(rows[0]["bits_per_pixel"])
        medians[key] = value

    paired_path = os.path.join(directory, "gpu-entropy-paired.csv")
    paired_fields = [
        "image",
        "image_type",
        "path",
        "quality",
        "chroma_subsampling",
        "progressive_level",
        "mode",
        "pixels",
        "bits_per_pixel",
        "used_gpu_entropy",
        "gpu_dram_bandwidth_estimated",
    ]
    for metric in METRICS:
        paired_fields.extend((metric, f"{metric}_ratio_vs_off"))
    paired_rows = []
    for key in sorted(medians):
        image, path, quality, sampling, progressive, mode = key
        value = medians[key]
        off_key = (image, path, quality, sampling, progressive, "off")
        off = medians.get(off_key)
        row = {
            "image": image,
            "image_type": value["image_type"],
            "path": path,
            "quality": quality,
            "chroma_subsampling": sampling,
            "progressive_level": progressive,
            "mode": mode,
            "pixels": value["pixels"],
            "bits_per_pixel": value["bits_per_pixel"],
            "used_gpu_entropy": value["used_gpu_entropy"],
            "gpu_dram_bandwidth_estimated": value[
                "gpu_dram_bandwidth_estimated"
            ],
        }
        for metric in METRICS:
            row[metric] = value[metric]
            row[f"{metric}_ratio_vs_off"] = (
                value[metric] / off[metric]
                if off is not None
                and math.isfinite(value[metric])
                and math.isfinite(off[metric])
                and off[metric] != 0
                else float("nan")
            )
        paired_rows.append(row)
    with open(paired_path, "w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired_rows)

    groups = defaultdict(list)
    for row in paired_rows:
        if not math.isfinite(row["latency_ms_per_decode_ratio_vs_off"]):
            continue
        group = (
            row["quality"],
            row["chroma_subsampling"],
            row["progressive_level"],
            row["path"],
            row["mode"],
        )
        groups[group].append(row)

    summary_path = os.path.join(directory, "gpu-entropy-summary.md")
    has_estimated_bandwidth = any(
        row["gpu_dram_bandwidth_estimated"] for row in paired_rows
    )
    with open(summary_path, "w") as out:
        out.write("# Apple Metal GPU entropy campaign summary\n\n")
        out.write(
            "Energy is the macOS process-attributed `ri_energy_nj` value. "
            "CPU and GPU rail energy are system-wide IOReport values when "
            "available. JPEG MB/s and RGBA GB/s are effective serial "
            "decode delivery rates. GFX DRAM GB/s comes from IOReport's "
            "hardware-controller counters. "
            + (
                "On this machine it is a residency-weighted PMP histogram "
                "estimate; the 32 GB/s top bucket can understate peak traffic. "
                if has_estimated_bandwidth
                else ""
            )
            + "Ratios are paired geometric means across image types; "
            "negative latency and energy deltas are improvements.\n\n"
        )
        out.write(
            "| JPEG | Path | Mode | Images using GPU entropy | Latency vs off "
            "| CPU-attributed energy vs off | CPU rail energy vs off "
            "| GPU rail energy vs off | CPU+GPU rail energy vs off "
            "| CPU+GPU rail power vs off | JPEG bandwidth vs off "
            "| RGBA bandwidth vs off | GFX DRAM bandwidth vs off |\n"
        )
        out.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for group in sorted(groups):
            quality, sampling, progressive, path, mode = group
            if mode == "off":
                continue
            rows = groups[group]
            ratios = {
                metric: geometric_mean(
                    [row[f"{metric}_ratio_vs_off"] for row in rows]
                )
                for metric in METRICS
            }
            coding = "progressive" if progressive != "0" else "baseline"
            used = sum(row["used_gpu_entropy"] for row in rows)
            out.write(
                f"| Q{quality} {sampling} {coding} | {path} | {mode} | "
                f"{used}/{len(rows)} | "
                f"{format_ratio(ratios['latency_ms_per_decode'])} | "
                f"{format_ratio(ratios['energy_mj_per_decode'])} | "
                f"{format_ratio(ratios['cpu_rail_energy_mj_per_decode'])} | "
                f"{format_ratio(ratios['gpu_rail_energy_mj_per_decode'])} | "
                f"{format_ratio(ratios['cpu_gpu_rail_energy_mj_per_decode'])} | "
                f"{format_ratio(ratios['average_cpu_gpu_rail_power_w'])} | "
                f"{format_ratio(ratios['compressed_input_mb_per_second'])} | "
                f"{format_ratio(ratios['rgba_output_gb_per_second'])} | "
                f"{format_ratio(ratios['gpu_dram_total_gb_per_second'])} |\n"
            )

        selected = [
            row
            for row in paired_rows
            if row["mode"] == "auto" and row["used_gpu_entropy"] == 1
        ]
        if selected:
            out.write("\n## AUTO-selected image detail\n\n")
            out.write(
                "CPU+GPU rail energy is joules spent per completed image; "
                "rail power is the average wattage during the decode window. "
                "A faster path can use fewer joules while drawing more watts.\n\n"
            )
            out.write(
                "| JPEG | Image | Path | Latency off → auto | RGBA GB/s off → auto "
                "| CPU rail W off → auto | GPU rail W off → auto "
                "| CPU+GPU mJ/image off → auto | GFX GB/s off → auto |\n"
            )
            out.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
            for row in sorted(
                selected,
                key=lambda value: (
                    int(value["quality"]),
                    value["chroma_subsampling"],
                    value["image"],
                    value["path"],
                ),
            ):
                off = medians[
                    (
                        row["image"],
                        row["path"],
                        row["quality"],
                        row["chroma_subsampling"],
                        row["progressive_level"],
                        "off",
                    )
                ]
                coding = (
                    "progressive"
                    if row["progressive_level"] != "0"
                    else "baseline"
                )
                jpeg = (
                    f"Q{row['quality']} {row['chroma_subsampling']} {coding}"
                )

                def transition(metric, precision):
                    before = off[metric]
                    after = row[metric]
                    ratio = row[f"{metric}_ratio_vs_off"]
                    if not all(math.isfinite(value) for value in (before, after, ratio)):
                        return "n/a"
                    return (
                        f"{before:.{precision}f} → {after:.{precision}f} "
                        f"({format_ratio(ratio)})"
                    )

                out.write(
                    f"| {jpeg} | {row['image']} | {row['path']} | "
                    f"{transition('latency_ms_per_decode', 2)} | "
                    f"{transition('rgba_output_gb_per_second', 2)} | "
                    f"{transition('average_cpu_rail_power_w', 2)} | "
                    f"{transition('average_gpu_rail_power_w', 2)} | "
                    f"{transition('cpu_gpu_rail_energy_mj_per_decode', 1)} | "
                    f"{transition('gpu_dram_total_gb_per_second', 2)} |\n"
                )

    print(f"Wrote {paired_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
