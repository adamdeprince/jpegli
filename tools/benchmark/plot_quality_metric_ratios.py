#!/usr/bin/env python3
"""Plot JPEGli corpus-mean quality-loss ratios for four perceptual metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable


METRICS: dict[str, tuple[str, Callable[[float], float]]] = {
    "ssimulacra2": ("SSIMULACRA2", lambda score: max(100.0 - score, 0.0)),
    "butteraugli": ("Butteraugli", lambda score: max(score, 0.0)),
    "cvvdp_jod": ("ColorVideoVDP", lambda score: max(10.0 - score, 0.0)),
    "dists": ("DISTS", lambda score: max(score, 0.0)),
}
METRIC_DIRECTIONS = {
    "ssimulacra2": "higher",
    "butteraugli": "lower",
    "cvvdp_jod": "higher",
    "dists": "lower",
}
COLORS = {
    "ssimulacra2": "#2563eb",
    "butteraugli": "#dc2626",
    "cvvdp_jod": "#059669",
    "dists": "#9333ea",
}
MARKERS = {
    "ssimulacra2": "o",
    "butteraugli": "s",
    "cvvdp_jod": "^",
    "dists": "D",
}
COMPARISONS = {
    "turbojpeg": "TurboJPEG",
    "apple_imageio": "Apple ImageIO",
}
SUBSAMPLINGS = ("444", "422", "420")
QUALITIES = tuple(range(25, 101, 5))
KEY_COLUMNS = (
    "image",
    "quality",
    "chroma_subsampling",
    "progressive_level",
    "decoder",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--ratio-csv",
        type=Path,
        help="aggregate ratio table (default: OUTPUT-DIR/quality-loss-ratios.csv)",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        help="merged per-image corpus (default: OUTPUT-DIR/four-metric-corpus.csv)",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="plot methodology (default: OUTPUT-DIR/quality-loss-ratios.json)",
    )
    parser.add_argument(
        "--score-summary-csv",
        type=Path,
        help="aggregate metric scores (default: OUTPUT-DIR/four-metric-summary.csv)",
    )
    parser.add_argument(
        "--win-counts-csv",
        type=Path,
        help="paired per-image wins (default: OUTPUT-DIR/quality-win-counts.csv)",
    )
    args = parser.parse_args()
    if args.ratio_csv is None:
        args.ratio_csv = args.output_dir / "quality-loss-ratios.csv"
    if args.combined_csv is None:
        args.combined_csv = args.output_dir / "four-metric-corpus.csv"
    if args.metadata_json is None:
        args.metadata_json = args.output_dir / "quality-loss-ratios.json"
    if args.score_summary_csv is None:
        args.score_summary_csv = args.output_dir / "four-metric-summary.csv"
    if args.win_counts_csv is None:
        args.win_counts_csv = args.output_dir / "quality-win-counts.csv"
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(paths: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no header")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"CSV schema differs in {path}")
            for row in reader:
                key = tuple(row[column] for column in KEY_COLUMNS)
                if key in seen:
                    raise ValueError(f"duplicate row {key}")
                seen.add(key)
                rows.append(row)
    if fieldnames is None or not rows:
        raise ValueError("no metric rows found")
    missing = set(METRICS) - set(fieldnames)
    if missing:
        raise ValueError(f"missing metric columns: {', '.join(sorted(missing))}")
    rows.sort(
        key=lambda row: (
            SUBSAMPLINGS.index(row["chroma_subsampling"]),
            int(row["quality"]),
            row["image"],
            row["decoder"],
        )
    )
    return fieldnames, rows


def validate_corpus(rows: list[dict[str, str]]) -> None:
    found_subsamplings = {row["chroma_subsampling"] for row in rows}
    if found_subsamplings != set(SUBSAMPLINGS):
        raise ValueError(
            f"expected subsampling {SUBSAMPLINGS}, found {sorted(found_subsamplings)}"
        )
    found_qualities = {int(row["quality"]) for row in rows}
    if found_qualities != set(QUALITIES):
        raise ValueError(
            f"expected qualities {QUALITIES}, found {sorted(found_qualities)}"
        )
    if {row["progressive_level"] for row in rows} != {"0"}:
        raise ValueError("the quality-ratio campaign must contain baseline rows only")
    decoders = {row["decoder"] for row in rows}
    expected_decoders = {"jpegli", *COMPARISONS}
    if decoders != expected_decoders:
        raise ValueError(f"expected decoders {sorted(expected_decoders)}, found {sorted(decoders)}")

    counts: dict[tuple[str, int, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["chroma_subsampling"], int(row["quality"]), row["decoder"])] += 1
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise ValueError(f"unbalanced corpus; per-setting counts are {sorted(unique_counts)}")
    if unique_counts != {30}:
        raise ValueError(
            f"expected 30 images per quality/sampling/decoder setting, "
            f"found {sorted(unique_counts)}"
        )


def paired_ratios(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    paired: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["image"], row["chroma_subsampling"], int(row["quality"]))
        paired[key][row["decoder"]] = row

    values: dict[tuple[str, str, int, str], list[tuple[float, float]]] = defaultdict(
        list
    )
    for (image, subsampling, quality), decoder_rows in paired.items():
        del image
        if set(decoder_rows) != {"jpegli", *COMPARISONS}:
            raise ValueError(
                f"incomplete decoder set for {subsampling} Q{quality}: "
                f"{sorted(decoder_rows)}"
            )
        jpegli = decoder_rows["jpegli"]
        for competitor in COMPARISONS:
            compared = decoder_rows[competitor]
            for metric, (_, loss) in METRICS.items():
                jpegli_loss = loss(float(jpegli[metric]))
                competitor_loss = loss(float(compared[metric]))
                values[(competitor, subsampling, quality, metric)].append(
                    (jpegli_loss, competitor_loss)
                )

    aggregates: list[dict[str, str]] = []
    for key in sorted(
        values,
        key=lambda item: (
            tuple(COMPARISONS).index(item[0]),
            SUBSAMPLINGS.index(item[1]),
            item[2],
            tuple(METRICS).index(item[3]),
        ),
    ):
        losses = values[key]
        jpegli_mean = statistics.fmean(value[0] for value in losses)
        competitor_mean = statistics.fmean(value[1] for value in losses)
        if jpegli_mean == 0.0:
            ratio = 1.0 if competitor_mean == 0.0 else math.inf
        else:
            ratio = competitor_mean / jpegli_mean
        aggregates.append(
            {
                "competitor": key[0],
                "chroma_subsampling": key[1],
                "quality": str(key[2]),
                "metric": key[3],
                "images": str(len(losses)),
                "jpegli_mean_distortion": f"{jpegli_mean:.12f}",
                "competitor_mean_distortion": f"{competitor_mean:.12f}",
                "distortion_ratio": f"{ratio:.9f}",
            }
        )
    return aggregates


def aggregate_scores(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (row["chroma_subsampling"], int(row["quality"]), row["decoder"])
        ].append(row)

    output: list[dict[str, str]] = []
    for key in sorted(
        groups,
        key=lambda item: (
            SUBSAMPLINGS.index(item[0]),
            item[1],
            ("jpegli", *COMPARISONS).index(item[2]),
        ),
    ):
        group = groups[key]
        pixels = sum(int(row["width"]) * int(row["height"]) for row in group)
        jpeg_bytes = sum(int(row["jpeg_bytes"]) for row in group)
        aggregate = {
            "quality": str(key[1]),
            "chroma_subsampling": key[0],
            "decoder": key[2],
            "images": str(len(group)),
            "bits_per_pixel": f"{8 * jpeg_bytes / pixels:.9f}",
        }
        for metric, direction in METRIC_DIRECTIONS.items():
            values = [float(row[metric]) for row in group]
            worst = min(values) if direction == "higher" else max(values)
            aggregate[f"{metric}_mean"] = f"{statistics.fmean(values):.9f}"
            aggregate[f"{metric}_median"] = f"{statistics.median(values):.9f}"
            aggregate[f"{metric}_worst"] = f"{worst:.9f}"
        output.append(aggregate)
    return output


def paired_win_counts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    paired: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["image"], row["chroma_subsampling"], int(row["quality"]))
        paired[key][row["decoder"]] = row

    counts: dict[tuple[str, str, int, str], list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )
    for (_, subsampling, quality), decoder_rows in paired.items():
        jpegli = decoder_rows["jpegli"]
        for competitor in COMPARISONS:
            compared = decoder_rows[competitor]
            outcomes: list[int] = []
            for metric, (_, loss) in METRICS.items():
                jpegli_loss = loss(float(jpegli[metric]))
                competitor_loss = loss(float(compared[metric]))
                outcome = (
                    0
                    if jpegli_loss < competitor_loss
                    else 1
                    if competitor_loss < jpegli_loss
                    else 2
                )
                counts[(competitor, subsampling, quality, metric)][outcome] += 1
                outcomes.append(outcome)
            if all(value == 0 for value in outcomes):
                combined = 0
            elif all(value == 1 for value in outcomes):
                combined = 1
            else:
                combined = 2
            counts[(competitor, subsampling, quality, "all_four")][combined] += 1

    output: list[dict[str, str]] = []
    criteria = (*METRICS, "all_four")
    for key in sorted(
        counts,
        key=lambda item: (
            tuple(COMPARISONS).index(item[0]),
            SUBSAMPLINGS.index(item[1]),
            item[2],
            criteria.index(item[3]),
        ),
    ):
        jpegli_better, competitor_better, tied_or_mixed = counts[key]
        output.append(
            {
                "competitor": key[0],
                "chroma_subsampling": key[1],
                "quality": str(key[2]),
                "criterion": key[3],
                "images": str(sum(counts[key])),
                "jpegli_better": str(jpegli_better),
                "competitor_better": str(competitor_better),
                "tied_or_mixed": str(tied_or_mixed),
            }
        )
    return output


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ratio_ticks(lower: float, upper: float) -> list[float]:
    candidates = (
        0.25,
        0.4,
        0.5,
        2 / 3,
        0.8,
        1 / 1.1,
        1.0,
        1.1,
        1.25,
        1.5,
        2.0,
        2.5,
        4.0,
    )
    ticks = [value for value in candidates if lower <= value <= upper]
    return ticks if len(ticks) >= 3 else [lower, 1.0, upper]


def plot_ratios(
    output_dir: Path, aggregates: list[dict[str, str]]
) -> tuple[list[Path], list[Path]]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, FixedLocator

    plt.rcParams["svg.hashsalt"] = "jpegli-quality-ratio-v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_svg: list[Path] = []
    generated_pdf: list[Path] = []
    for competitor, competitor_label in COMPARISONS.items():
        for subsampling in SUBSAMPLINGS:
            selected = [
                row
                for row in aggregates
                if row["competitor"] == competitor
                and row["chroma_subsampling"] == subsampling
            ]
            all_ratios = [
                float(row["distortion_ratio"])
                for row in selected
                if math.isfinite(float(row["distortion_ratio"]))
                and 0.25 <= float(row["distortion_ratio"]) <= 4.0
            ]
            log_extent = max(abs(math.log2(value)) for value in all_ratios)
            log_extent = max(log_extent * 1.12, math.log2(1.1))
            lower = 2 ** (-log_extent)
            upper = 2**log_extent

            fig, axis = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)
            fig.patch.set_facecolor("#fffefa")
            axis.set_facecolor("#ffffff")
            for metric, (label, _) in METRICS.items():
                metric_rows = sorted(
                    (row for row in selected if row["metric"] == metric),
                    key=lambda row: int(row["quality"]),
                )
                raw_values = [float(row["distortion_ratio"]) for row in metric_rows]
                display_values = [
                    upper
                    if math.isinf(value) or value > upper
                    else lower
                    if value < lower
                    else value
                    for value in raw_values
                ]
                axis.plot(
                    [int(row["quality"]) for row in metric_rows],
                    display_values,
                    label=label,
                    color=COLORS[metric],
                    marker=MARKERS[metric],
                    linewidth=2.2,
                    markersize=4.8,
                )
                for row, raw_value, display_value in zip(
                    metric_rows, raw_values, display_values
                ):
                    if not lower <= raw_value <= upper:
                        high = raw_value > upper
                        marker = "^" if high else "v"
                        boundary = display_value / 1.025 if high else display_value * 1.025
                        axis.scatter(
                            int(row["quality"]),
                            boundary,
                            color=COLORS[metric],
                            marker=marker,
                            s=70,
                            zorder=5,
                        )
                        value_label = "∞" if math.isinf(raw_value) else f"{raw_value:.3g}×"
                        if metric == "cvvdp_jod" and high:
                            value_label += " (near 10-JOD ceiling)"
                        axis.annotate(
                            value_label,
                            (int(row["quality"]), boundary),
                            xytext=(-8, -13 if high else 13),
                            textcoords="offset points",
                            ha="right",
                            va="top" if high else "bottom",
                            fontsize=8.2,
                            color=COLORS[metric],
                        )

            axis.axhline(1.0, color="#334155", linewidth=1.2, linestyle="--")
            axis.set_yscale("log", base=2)
            axis.set_ylim(lower, upper)
            ticks = ratio_ticks(lower, upper)
            axis.yaxis.set_major_locator(FixedLocator(ticks))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}×"))
            axis.set_xticks(QUALITIES)
            axis.set_xlim(24, 101)
            axis.grid(axis="both", color="#cbd5e1", linewidth=0.7, alpha=0.65)
            axis.set_axisbelow(True)
            axis.set_xlabel("JPEGli encoder quality")
            axis.set_ylabel(f"{competitor_label} distortion / JPEGli distortion")
            axis.set_title(
                f"JPEGli relative reconstruction quality vs {competitor_label} — "
                f"4:{subsampling[1]}:{subsampling[2]}"
            )
            axis.legend(ncol=2, frameon=False, loc="best")
            axis.text(
                0.01,
                0.015,
                "Mean-distortion ratio, 30 paired images · above 1× favors JPEGli · log₂ axis · arrows are off-scale",
                transform=axis.transAxes,
                fontsize=8.5,
                color="#475569",
            )
            for spine in axis.spines.values():
                spine.set_color("#94a3b8")

            path = output_dir / f"quality-ratio-{competitor}-{subsampling}.svg"
            pdf_path = path.with_suffix(".pdf")
            fig.savefig(
                path,
                format="svg",
                metadata={"Date": None, "Title": axis.get_title()},
                facecolor=fig.get_facecolor(),
            )
            fig.savefig(
                pdf_path,
                format="pdf",
                metadata={"CreationDate": None, "ModDate": None},
                facecolor=fig.get_facecolor(),
            )
            plt.close(fig)
            generated_svg.append(path)
            generated_pdf.append(pdf_path)
    return generated_svg, generated_pdf


def main() -> int:
    args = parse_args()
    fieldnames, rows = read_rows(args.input_csv)
    validate_corpus(rows)
    aggregates = paired_ratios(rows)
    score_summary = aggregate_scores(rows)
    win_counts = paired_win_counts(rows)
    write_csv(args.combined_csv, fieldnames, rows)
    write_csv(args.ratio_csv, list(aggregates[0]), aggregates)
    write_csv(args.score_summary_csv, list(score_summary[0]), score_summary)
    write_csv(args.win_counts_csv, list(win_counts[0]), win_counts)
    generated_svg, generated_pdf = plot_ratios(args.output_dir, aggregates)
    metadata = {
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.input_csv
        ],
        "rows": len(rows),
        "images_per_setting": int(aggregates[0]["images"]),
        "qualities": list(QUALITIES),
        "subsampling": list(SUBSAMPLINGS),
        "comparisons": COMPARISONS,
        "aggregation": "ratio of arithmetic-mean distortion over 30 paired images",
        "ratio_direction": "above 1 means JPEGli has less predicted distortion",
        "loss_transforms": {
            "ssimulacra2": "100 - score",
            "butteraugli": "score",
            "cvvdp_jod": "10 - score",
            "dists": "score",
        },
        "axis": "base-2 logarithmic with reciprocal bounds",
        "zero_loss_handling": (
            "an incumbent/JPEGli distortion ratio is infinite when JPEGli "
            "reaches a metric's zero-distortion ceiling and the incumbent does not; "
            "the plot marks it at the upper boundary rather than applying an "
            "arbitrary epsilon"
        ),
        "plots": [path.name for path in generated_svg],
        "paper_plot_files": [path.name for path in generated_pdf],
        "output_sha256": {
            args.combined_csv.name: sha256_file(args.combined_csv),
            args.ratio_csv.name: sha256_file(args.ratio_csv),
            args.score_summary_csv.name: sha256_file(args.score_summary_csv),
            args.win_counts_csv.name: sha256_file(args.win_counts_csv),
            **{
                path.name: sha256_file(path)
                for path in (*generated_svg, *generated_pdf)
            },
        },
        "plot_script_sha256": sha256_file(Path(__file__)),
    }
    args.metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Merged {len(rows)} metric rows into {args.combined_csv}")
    print(f"Wrote {len(aggregates)} aggregate ratios to {args.ratio_csv}")
    print(f"Wrote {len(score_summary)} aggregate scores to {args.score_summary_csv}")
    print(f"Wrote {len(win_counts)} paired win counts to {args.win_counts_csv}")
    for path in (*generated_svg, *generated_pdf):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
