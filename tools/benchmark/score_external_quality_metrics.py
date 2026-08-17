#!/usr/bin/env python3
"""Add ColorVideoVDP and DISTS scores to a JPEGli quality-corpus CSV.

The C++ jpegli_decode_benchmark owns encoding and decoder selection. Run it
with --perceptual_quality, --quality_image_dir, and --csv first. This script
then evaluates the exported lossless RGB images with the authors' Python
implementations while preserving the C++ SSIMULACRA2 and Butteraugli columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_COLUMNS = ("cvvdp_jod", "dists")
ROW_KEY_COLUMNS = (
    "image",
    "width",
    "height",
    "quality",
    "chroma_subsampling",
    "progressive_level",
    "decoder",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--quality-image-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="aggregate output (default: OUTPUT.summary.csv)",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="run metadata (default: OUTPUT.metadata.json)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="PyTorch device (default: prefer MPS, then CUDA, then CPU)",
    )
    parser.add_argument(
        "--cvvdp-display",
        default="standard_4k",
        help="ColorVideoVDP display model (default: standard_4k)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="atomically checkpoint after this many new rows (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score only the first N rows in processing order; 0 means all",
    )
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.limit < 0:
        parser.error("--limit must not be negative")
    if args.summary_csv is None:
        args.summary_csv = args.output_csv.with_suffix(".summary.csv")
    if args.metadata_json is None:
        args.metadata_json = args.output_csv.with_suffix(".metadata.json")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in ROW_KEY_COLUMNS)


def load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        required = set(ROW_KEY_COLUMNS) | {
            "bits_per_pixel",
            "ssimulacra2",
            "butteraugli",
            "decoded_image",
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    keys = [row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate metric rows")
    return list(reader.fieldnames), rows


def restore_completed(output_path: Path, rows: list[dict[str, str]]) -> int:
    if not output_path.exists():
        return 0
    with output_path.open(newline="", encoding="utf-8") as source:
        completed = list(csv.DictReader(source))
    by_key = {row_key(row): row for row in completed}
    restored = 0
    for row in rows:
        previous = by_key.get(row_key(row))
        if previous is None or not all(previous.get(column) for column in METRIC_COLUMNS):
            continue
        for column in METRIC_COLUMNS:
            value = float(previous[column])
            if not math.isfinite(value):
                raise ValueError(f"non-finite checkpoint value for {row_key(row)}")
            row[column] = previous[column]
        restored += 1
    return restored


def atomic_write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        selected = requested
    elif torch.backends.mps.is_available():
        selected = "mps"
    elif torch.cuda.is_available():
        selected = "cuda"
    else:
        selected = "cpu"
    if selected == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def discover_source_files(source_dir: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in sources:
            duplicates.add(path.name)
        else:
            sources[path.name] = path
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"source filenames are not unique: {duplicate_list}")
    return sources


def decoded_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe decoded_image path: {value!r}")
    return root / relative


def load_rgb(path: Path, image_module: Any, numpy: Any) -> Any:
    with image_module.open(path) as image:
        rgb = image.convert("RGB")
        return numpy.asarray(rgb, dtype=numpy.uint8).copy()


def image_tensor(rgb: Any, torch: Any, device: str) -> Any:
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32).div_(255.0)


def dists_from_features(model: Any, reference: list[Any], test: list[Any], torch: Any) -> Any:
    distance_structure: Any = 0
    distance_texture: Any = 0
    weight_sum = model.alpha.sum() + model.beta.sum()
    alpha = torch.split(model.alpha / weight_sum, model.chns, dim=1)
    beta = torch.split(model.beta / weight_sum, model.chns, dim=1)
    for level, (ref_feature, test_feature) in enumerate(zip(reference, test)):
        ref_mean = ref_feature.mean((2, 3), keepdim=True)
        test_mean = test_feature.mean((2, 3), keepdim=True)
        structure = (2 * ref_mean * test_mean + 1e-6) / (
            ref_mean.square() + test_mean.square() + 1e-6
        )
        distance_structure = distance_structure + (alpha[level] * structure).sum(
            1, keepdim=True
        )
        ref_variance = (ref_feature - ref_mean).square().mean((2, 3), keepdim=True)
        test_variance = (test_feature - test_mean).square().mean(
            (2, 3), keepdim=True
        )
        covariance = (ref_feature * test_feature).mean(
            (2, 3), keepdim=True
        ) - ref_mean * test_mean
        texture = (2 * covariance + 1e-6) / (
            ref_variance + test_variance + 1e-6
        )
        distance_texture = distance_texture + (beta[level] * texture).sum(
            1, keepdim=True
        )
    return 1 - (distance_structure + distance_texture).squeeze()


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not all(row.get(column) for column in METRIC_COLUMNS):
            continue
        groups[(int(row["quality"]), row["chroma_subsampling"], row["decoder"])].append(
            row
        )

    output: list[dict[str, str]] = []
    metric_directions = {
        "ssimulacra2": "higher",
        "butteraugli": "lower",
        "cvvdp_jod": "higher",
        "dists": "lower",
    }
    for key in sorted(groups):
        group = groups[key]
        pixels = sum(int(row["width"]) * int(row["height"]) for row in group)
        jpeg_bytes = sum(int(row["jpeg_bytes"]) for row in group)
        aggregate: dict[str, str] = {
            "quality": str(key[0]),
            "chroma_subsampling": key[1],
            "decoder": key[2],
            "images": str(len(group)),
            "bits_per_pixel": f"{8 * jpeg_bytes / pixels:.9f}",
        }
        for metric, direction in metric_directions.items():
            values = [float(row[metric]) for row in group]
            worst = min(values) if direction == "higher" else max(values)
            aggregate[f"{metric}_mean"] = f"{statistics.fmean(values):.9f}"
            aggregate[f"{metric}_median"] = f"{statistics.median(values):.9f}"
            aggregate[f"{metric}_worst"] = f"{worst:.9f}"
        output.append(aggregate)
    return output


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    summary = aggregate_rows(rows)
    if not summary:
        return
    atomic_write_csv(path, list(summary[0]), summary)
    print(
        f"{'q':>3} {'chroma':>6} {'decoder':<14} {'SSIM2':>10} "
        f"{'BA':>10} {'CVVDP':>10} {'DISTS':>10}"
    )
    for row in summary:
        print(
            f"{row['quality']:>3} {row['chroma_subsampling']:>6} "
            f"{row['decoder']:<14} {float(row['ssimulacra2_mean']):10.4f} "
            f"{float(row['butteraugli_mean']):10.4f} "
            f"{float(row['cvvdp_jod_mean']):10.4f} "
            f"{float(row['dists_mean']):10.6f}"
        )


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def existing_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def write_metadata(
    args: argparse.Namespace,
    device: str,
    torch: Any,
    pycvvdp: Any,
    cvvdp_model: Any,
    rows: list[dict[str, str]],
) -> None:
    cvvdp_root = Path(pycvvdp.__file__).parent
    dists_spec = importlib.util.find_spec("DISTS_pytorch")
    if dists_spec is None or dists_spec.origin is None:
        raise RuntimeError("could not locate the installed DISTS implementation")
    dists_root = Path(dists_spec.origin).parent
    dists_weights = Path(sys.prefix) / "weights.pt"
    vgg_weights = Path(torch.hub.get_dir()) / "checkpoints" / "vgg16-397923af.pth"
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "scoring_script_sha256": sha256_file(Path(__file__)),
        "requirements_sha256": sha256_file(
            Path(__file__).with_name("quality_metrics_requirements.txt")
        ),
        "rows": len(rows),
        "completed_rows": sum(
            all(row.get(column) for column in METRIC_COLUMNS) for row in rows
        ),
        "input_csv": str(args.input_csv.resolve()),
        "input_csv_sha256": sha256_file(args.input_csv),
        "source_dir": str(args.source_dir.resolve()),
        "quality_image_dir": str(args.quality_image_dir.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "output_csv_sha256": sha256_file(args.output_csv),
        "summary_csv": str(args.summary_csv.resolve()),
        "summary_csv_sha256": sha256_file(args.summary_csv),
        "metrics": {
            "ssimulacra2": {"direction": "higher", "implementation": "JPEGli C++"},
            "butteraugli": {
                "direction": "lower",
                "implementation": "JPEGli C++",
                "intensity_target_nits": 80,
            },
            "cvvdp_jod": {
                "direction": "higher",
                "package": "cvvdp",
                "version": package_version("cvvdp"),
                "display": args.cvvdp_display,
                "reporting_string": cvvdp_model.get_info_string(),
                "input": "native-resolution uint8 sRGB",
                "implementation_sha256": existing_hash(
                    cvvdp_root / "cvvdp_metric.py"
                ),
                "display_models_sha256": existing_hash(
                    cvvdp_root / "vvdp_data" / "display_models.json"
                ),
                "parameters_sha256": existing_hash(
                    cvvdp_root / "vvdp_data" / "cvvdp_parameters.json"
                ),
            },
            "dists": {
                "direction": "lower",
                "package": "DISTS-pytorch",
                "version": package_version("DISTS-pytorch"),
                "input": "native-resolution float32 sRGB in [0,1]; no resize",
                "implementation_sha256": existing_hash(
                    dists_root / "DISTS_pt.py"
                ),
                "weights_sha256": existing_hash(dists_weights),
                "vgg16_weights_sha256": existing_hash(vgg_weights),
            },
        },
        "runtime": {
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "numpy": package_version("numpy"),
            "Pillow": package_version("Pillow"),
        },
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    try:
        import numpy
        import pycvvdp
        import torch
        from DISTS_pytorch import DISTS
        from PIL import Image
    except ImportError as error:
        raise SystemExit(
            "Missing metric dependencies. Install the pinned packages from "
            "tools/benchmark/quality_metrics_requirements.txt"
        ) from error

    fieldnames, rows = load_manifest(args.input_csv)
    for column in METRIC_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    restored = restore_completed(args.output_csv, rows)
    sources = discover_source_files(args.source_dir)
    device = choose_device(torch, args.device)
    print(
        f"Scoring {len(rows)} rows on {device}; restored {restored} completed rows. "
        f"CVVDP display={args.cvvdp_display}; DISTS uses native resolution."
    )

    torch_device = torch.device(device)
    dists_model = DISTS().eval().to(torch_device)
    cvvdp_model = pycvvdp.cvvdp(
        display_name=args.cvvdp_display,
        heatmap=None,
        quiet=True,
        device=torch_device,
    )

    pending = [index for index, row in enumerate(rows) if not row.get("cvvdp_jod")]
    pending.sort(key=lambda index: (rows[index]["image"], index))
    if args.limit:
        pending = pending[: args.limit]
    completed_now = 0
    current_image = ""
    reference_rgb: Any = None
    reference_features: Any = None

    with torch.inference_mode():
        for position, index in enumerate(pending, start=1):
            row = rows[index]
            if row["image"] != current_image:
                source_path = sources.get(row["image"])
                if source_path is None:
                    raise FileNotFoundError(f"source image not found: {row['image']}")
                reference_rgb = load_rgb(source_path, Image, numpy)
                expected = (int(row["height"]), int(row["width"]), 3)
                if reference_rgb.shape != expected:
                    raise ValueError(
                        f"source dimensions differ for {row['image']}: "
                        f"{reference_rgb.shape} != {expected}"
                    )
                reference_tensor = image_tensor(reference_rgb, torch, device)
                reference_features = dists_model.forward_once(reference_tensor)
                current_image = row["image"]

            distorted_file = decoded_path(
                args.quality_image_dir, row["decoded_image"]
            )
            if not distorted_file.is_file():
                raise FileNotFoundError(f"decoded image not found: {distorted_file}")
            distorted_rgb = load_rgb(distorted_file, Image, numpy)
            if distorted_rgb.shape != reference_rgb.shape:
                raise ValueError(
                    f"decoded dimensions differ for {distorted_file}: "
                    f"{distorted_rgb.shape} != {reference_rgb.shape}"
                )

            cvvdp_score, _ = cvvdp_model.predict(
                distorted_rgb, reference_rgb, dim_order="HWC"
            )
            distorted_tensor = image_tensor(distorted_rgb, torch, device)
            distorted_features = dists_model.forward_once(distorted_tensor)
            dists_score = dists_from_features(
                dists_model, reference_features, distorted_features, torch
            )
            cvvdp_value = float(cvvdp_score.item())
            dists_value = float(dists_score.item())
            if not math.isfinite(cvvdp_value) or not math.isfinite(dists_value):
                raise ValueError(f"non-finite score for {row_key(row)}")
            row["cvvdp_jod"] = f"{cvvdp_value:.9f}"
            row["dists"] = f"{dists_value:.9f}"
            completed_now += 1

            if completed_now % args.checkpoint_every == 0:
                atomic_write_csv(args.output_csv, fieldnames, rows)
            if position == len(pending) or position % max(1, len(pending) // 20) == 0:
                print(f"[{position}/{len(pending)}] external metric rows complete")

    atomic_write_csv(args.output_csv, fieldnames, rows)
    write_summary(args.summary_csv, rows)
    write_metadata(args, device, torch, pycvvdp, cvvdp_model, rows)
    print(f"Per-image four-metric CSV written to {args.output_csv}")
    print(f"Aggregate four-metric CSV written to {args.summary_csv}")
    print(f"Metric metadata written to {args.metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
