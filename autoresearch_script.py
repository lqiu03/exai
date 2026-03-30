#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


EPS = 1e-6
PRIMARY_TUMOR = "01"
SOLID_NORMAL = "11"
RESEARCH_SCORE_WEIGHTS = {
    "mean_stability": 0.50,
    "paired_consistency": 0.35,
    "null_control_score": 0.15,
}


@dataclass(frozen=True)
class CandidateConfig:
    name: str = "baseline_beta_gap1000"
    transform: str = "beta"
    max_gap: int = 1000
    min_region_probes: int = 3
    min_samples_per_group: int = 3
    probe_fdr: float = 0.05
    min_abs_delta_beta: float = 0.10
    region_fdr_threshold: float = 1.0
    min_region_abs_mean_delta_beta: float = 0.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file Exai autoresearch loop for Colab.")
    parser.add_argument(
        "--methylation",
        default=os.environ.get("EXAI_METHYLATION", "/content/TCGA-COAD.methylation450.chr4.tsv"),
        help="Path to the methylation beta-value matrix. Default: /content/TCGA-COAD.methylation450.chr4.tsv",
    )
    parser.add_argument(
        "--probemap",
        default=os.environ.get("EXAI_PROBEMAP", "/content/HM450.hg38.chr4.probeMap.tsv"),
        help="Path to the probe map TSV. Default: /content/HM450.hg38.chr4.probeMap.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("EXAI_OUTPUT_DIR", "/content/exai_autoresearch_runs"),
        help="Root directory for autoresearch outputs. Default: /content/exai_autoresearch_runs",
    )
    parser.add_argument(
        "--attempt-label",
        default=os.environ.get("EXAI_ATTEMPT_LABEL", ""),
        help="Optional folder name under --output-dir (e.g. training_attempt_1). If empty, auto-increments.",
    )
    parser.add_argument("--iterations", type=int, default=4, help="Total experiments including baseline.")
    parser.add_argument(
        "--target-minutes",
        type=float,
        default=0.0,
        help="Optional wall-clock target for a run attempt. "
        "If > 0, the loop runs until this duration or until --iterations is reached (whichever comes first).",
    )
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name for proposing config edits.")
    parser.add_argument("--bootstrap-iterations", type=int, default=6, help="Bootstrap iterations.")
    parser.add_argument("--bootstrap-fraction", type=float, default=0.80, help="Bootstrap fraction per class.")
    parser.add_argument("--permutations", type=int, default=6, help="Shuffled-label control iterations.")
    parser.add_argument("--random-seed", type=int, default=20260327, help="Base random seed.")
    # In Jupyter/Colab, the kernel may inject extra argv flags (e.g. "-f ...kernel.json").
    # Ignore unknown flags so the script can run both as a CLI and from notebook contexts.
    args, _unknown = parser.parse_known_args()
    return args


def resolve_attempt_dir(root_output_dir: Path, attempt_label: str) -> Path:
    if attempt_label:
        return root_output_dir / attempt_label

    prefix = "training_attempt_"
    max_seen = 0
    if root_output_dir.exists():
        for child in root_output_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                max_seen = max(max_seen, int(suffix))
    return root_output_dir / f"{prefix}{max_seen + 1}"


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    result = np.full(pvalues.shape, np.nan, dtype=float)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return np.ones_like(pvalues, dtype=float)

    vals = np.clip(pvalues[valid], 0.0, 1.0)
    order = np.argsort(vals)
    ranked = vals[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty_like(vals)
    out[order] = adjusted
    result[valid] = out
    result[~valid] = 1.0
    return result


def beta_to_m(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, EPS, 1.0 - EPS)
    return np.log2(clipped / (1.0 - clipped))


def nanmean_no_warning(values: np.ndarray, axis: int) -> np.ndarray:
    counts = np.sum(~np.isnan(values), axis=axis)
    totals = np.nansum(values, axis=axis)
    return np.divide(
        totals,
        counts,
        out=np.full_like(totals, np.nan, dtype=float),
        where=counts > 0,
    )


def nanvar_ddof1(values: np.ndarray, axis: int) -> np.ndarray:
    counts = np.sum(~np.isnan(values), axis=axis)
    means = nanmean_no_warning(values, axis=axis)
    centered = np.where(np.isnan(values), 0.0, values - np.expand_dims(means, axis=axis))
    ss = np.sum(centered * centered, axis=axis)
    return np.divide(ss, counts - 1, out=np.full_like(ss, np.nan, dtype=float), where=counts > 1)


def signed_z_scores(pvalues: np.ndarray, effects: np.ndarray) -> np.ndarray:
    safe = np.clip(np.asarray(pvalues, dtype=float), 1e-300, 1.0)
    magnitudes = stats.norm.isf(safe / 2.0)
    signs = np.sign(np.asarray(effects, dtype=float))
    return magnitudes * np.where(signs == 0.0, 1.0, signs)


def normalize_gene_list(values: Iterable[object]) -> str:
    genes: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        for part in text.replace(";", ",").split(","):
            token = part.strip()
            if token:
                genes.add(token)
    return ",".join(sorted(genes)[:12]) if genes else "."


def parse_barcode(barcode: str) -> dict[str, str]:
    parts = barcode.split("-")
    sample = parts[3]
    return {
        "barcode": barcode,
        "participant": "-".join(parts[:3]),
        "sample_type": sample[:2],
    }


def build_sample_metadata(sample_columns: Iterable[str]) -> pd.DataFrame:
    records = [parse_barcode(col) for col in sample_columns]
    metadata = pd.DataFrame.from_records(records)
    metadata["group_key"] = metadata["participant"] + "|" + metadata["sample_type"]
    return metadata


def collapse_to_participant_level(methylation: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = methylation.drop(columns=["#id"])
    collapsed = {"#id": methylation["#id"].to_numpy()}
    collapsed_meta = []
    for group_key, group_df in metadata.groupby("group_key", sort=True):
        cols = group_df["barcode"].tolist()
        collapsed[group_key] = values[cols].mean(axis=1, skipna=True).to_numpy()
        collapsed_meta.append(
            {
                "column_name": group_key,
                "participant": group_df["participant"].iloc[0],
                "sample_type": group_df["sample_type"].iloc[0],
                "n_aliquots": len(cols),
            }
        )
    collapsed_df = pd.DataFrame(collapsed)
    collapsed_meta_df = pd.DataFrame(collapsed_meta).sort_values(["sample_type", "participant"]).reset_index(drop=True)
    return collapsed_df, collapsed_meta_df


def load_inputs(methylation_path: Path, probemap_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methylation = pd.read_csv(methylation_path, sep="\t")
    probemap = pd.read_csv(probemap_path, sep="\t")
    sample_meta = build_sample_metadata(methylation.columns[1:])
    collapsed_methylation, collapsed_meta = collapse_to_participant_level(methylation, sample_meta)
    merged = probemap.merge(collapsed_methylation, on="#id", how="inner")
    merged = merged.sort_values(["chrom", "chromStart", "chromEnd", "#id"]).reset_index(drop=True)
    return merged, sample_meta, collapsed_meta


def participant_columns(collapsed_meta: pd.DataFrame, sample_type: str) -> list[str]:
    return collapsed_meta.loc[collapsed_meta["sample_type"].eq(sample_type), "column_name"].tolist()


def matched_column_pairs(collapsed_meta: pd.DataFrame) -> tuple[list[str], list[str]]:
    tumor = collapsed_meta[collapsed_meta["sample_type"] == PRIMARY_TUMOR]
    normal = collapsed_meta[collapsed_meta["sample_type"] == SOLID_NORMAL]
    shared = sorted(set(tumor["participant"]) & set(normal["participant"]))
    tumor_cols = tumor.set_index("participant").loc[shared, "column_name"].tolist()
    normal_cols = normal.set_index("participant").loc[shared, "column_name"].tolist()
    return tumor_cols, normal_cols


def run_probe_statistics(
    merged: pd.DataFrame,
    tumor_cols: list[str],
    normal_cols: list[str],
    config: CandidateConfig,
    paired: bool,
) -> pd.DataFrame:
    tumor_beta = merged[tumor_cols].to_numpy(dtype=float)
    normal_beta = merged[normal_cols].to_numpy(dtype=float)
    tumor_beta_mean = nanmean_no_warning(tumor_beta, axis=1)
    normal_beta_mean = nanmean_no_warning(normal_beta, axis=1)
    delta_beta = tumor_beta_mean - normal_beta_mean

    tumor_test = beta_to_m(tumor_beta) if config.transform == "m" else tumor_beta
    normal_test = beta_to_m(normal_beta) if config.transform == "m" else normal_beta
    tumor_test_mean = nanmean_no_warning(tumor_test, axis=1)
    normal_test_mean = nanmean_no_warning(normal_test, axis=1)
    delta_test = tumor_test_mean - normal_test_mean

    n_tumor = np.sum(~np.isnan(tumor_beta), axis=1)
    n_normal = np.sum(~np.isnan(normal_beta), axis=1)
    valid_mask = (n_tumor >= 2) & (n_normal >= 2)
    pvalues = np.ones(merged.shape[0], dtype=float)
    statistics = np.zeros(merged.shape[0], dtype=float)
    if valid_mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if paired:
                diffs = tumor_test[valid_mask] - normal_test[valid_mask]
                paired_n = np.sum(~np.isnan(diffs), axis=1)
                diff_mean = nanmean_no_warning(diffs, axis=1)
                diff_var = nanvar_ddof1(diffs, axis=1)
                denom = np.sqrt(diff_var / paired_n)
                valid_stat = np.divide(
                    diff_mean,
                    denom,
                    out=np.zeros_like(diff_mean, dtype=float),
                    where=(paired_n > 1) & np.isfinite(denom) & (denom > 0.0),
                )
                df = paired_n - 1
                valid_p = 2.0 * stats.t.sf(np.abs(valid_stat), df=df)
            else:
                tumor_var = nanvar_ddof1(tumor_test[valid_mask], axis=1)
                normal_var = nanvar_ddof1(normal_test[valid_mask], axis=1)
                vt = tumor_var / n_tumor[valid_mask]
                vn = normal_var / n_normal[valid_mask]
                denom = np.sqrt(vt + vn)
                valid_stat = np.divide(
                    delta_test[valid_mask],
                    denom,
                    out=np.zeros_like(delta_test[valid_mask], dtype=float),
                    where=np.isfinite(denom) & (denom > 0.0),
                )
                df_num = (vt + vn) ** 2
                df_den = (vt**2) / (n_tumor[valid_mask] - 1) + (vn**2) / (n_normal[valid_mask] - 1)
                df = np.divide(df_num, df_den, out=np.full_like(df_num, np.nan, dtype=float), where=df_den > 0.0)
                valid_p = 2.0 * stats.t.sf(np.abs(valid_stat), df=df)
        pvalues[valid_mask] = np.where(np.isfinite(valid_p), valid_p, 1.0)
        statistics[valid_mask] = np.where(np.isfinite(valid_stat), valid_stat, 0.0)

    result = merged[["#id", "gene", "chrom", "chromStart", "chromEnd"]].copy()
    result["n_tumor"] = n_tumor
    result["n_normal"] = n_normal
    result["tumor_mean_beta"] = tumor_beta_mean
    result["normal_mean_beta"] = normal_beta_mean
    result["delta_beta"] = delta_beta
    result["tumor_mean_test"] = tumor_test_mean
    result["normal_mean_test"] = normal_test_mean
    result["delta_test"] = delta_test
    result["test_statistic"] = statistics
    result["pvalue"] = pvalues
    result["probe_fdr"] = bh_fdr(pvalues)
    return result


def empty_region_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "chrom",
            "chromStart",
            "chromEnd",
            "direction",
            "n_probes",
            "mean_delta_beta",
            "median_delta_beta",
            "mean_delta_test",
            "min_probe_pvalue",
            "min_probe_fdr",
            "region_pvalue",
            "region_z",
            "genes",
            "probe_ids",
            "region_fdr",
            "score",
            "name",
        ]
    )


def keep_probe(row: pd.Series, config: CandidateConfig) -> bool:
    return bool(
        (row["probe_fdr"] <= config.probe_fdr)
        and (abs(row["delta_beta"]) >= config.min_abs_delta_beta)
        and (row["n_tumor"] >= config.min_samples_per_group)
        and (row["n_normal"] >= config.min_samples_per_group)
    )


def build_regions(probes: pd.DataFrame, config: CandidateConfig) -> pd.DataFrame:
    usable = probes[probes.apply(lambda row: keep_probe(row, config), axis=1)].copy()
    if usable.empty:
        return empty_region_frame()

    usable["direction"] = np.where(usable["delta_beta"] > 0.0, "hyper", "hypo")
    usable["signed_z"] = signed_z_scores(usable["pvalue"].to_numpy(), usable["delta_test"].to_numpy())
    usable = usable.sort_values(["chrom", "chromStart", "chromEnd", "#id"]).reset_index(drop=True)

    regions: list[dict[str, object]] = []
    current: list[dict[str, object]] = []

    def flush_current() -> None:
        nonlocal current
        if len(current) < config.min_region_probes:
            current = []
            return

        frame = pd.DataFrame(current)
        region_z = frame["signed_z"].sum() / math.sqrt(len(frame))
        region_p = float(max(2.0 * stats.norm.sf(abs(region_z)), 1e-300))
        regions.append(
            {
                "chrom": frame["chrom"].iloc[0],
                "chromStart": int(frame["chromStart"].min()),
                "chromEnd": int(frame["chromEnd"].max()),
                "direction": frame["direction"].iloc[0],
                "n_probes": int(len(frame)),
                "mean_delta_beta": float(frame["delta_beta"].mean()),
                "median_delta_beta": float(frame["delta_beta"].median()),
                "mean_delta_test": float(frame["delta_test"].mean()),
                "min_probe_pvalue": float(frame["pvalue"].min()),
                "min_probe_fdr": float(frame["probe_fdr"].min()),
                "region_pvalue": region_p,
                "region_z": float(region_z),
                "genes": normalize_gene_list(frame["gene"]),
                "probe_ids": ",".join(frame["#id"].tolist()),
            }
        )
        current = []

    for probe in usable.to_dict("records"):
        if not current:
            current = [probe]
            continue
        last = current[-1]
        gap = int(probe["chromStart"]) - int(last["chromEnd"])
        same_region = (
            probe["chrom"] == last["chrom"]
            and probe["direction"] == last["direction"]
            and gap <= config.max_gap
        )
        if same_region:
            current.append(probe)
        else:
            flush_current()
            current = [probe]
    flush_current()

    if not regions:
        return empty_region_frame()

    region_df = pd.DataFrame(regions)
    region_df["region_fdr"] = bh_fdr(region_df["region_pvalue"].to_numpy())
    region_df = region_df[
        (region_df["region_fdr"] <= config.region_fdr_threshold)
        & (region_df["mean_delta_beta"].abs() >= config.min_region_abs_mean_delta_beta)
    ].copy()
    if region_df.empty:
        return empty_region_frame()

    region_df["score"] = region_df["region_fdr"].apply(
        lambda value: max(0, min(1000, int(round(-100.0 * math.log10(max(value, 1e-300))))))
    )
    region_df = region_df.sort_values(
        ["direction", "region_fdr", "n_probes", "mean_delta_beta"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    names = []
    counters = {"hyper": 0, "hypo": 0}
    for direction in region_df["direction"]:
        counters[direction] += 1
        prefix = "hyper" if direction == "hyper" else "hypo"
        names.append(f"{prefix}_DMR_{counters[direction]:04d}")
    region_df["name"] = names
    return region_df


def summarize_group_coverage(regions: pd.DataFrame, direction: str) -> set[str]:
    selected = regions[regions["direction"] == direction]
    coverage: set[str] = set()
    for probe_ids in selected["probe_ids"].tolist():
        coverage.update(probe_ids.split(","))
    return coverage


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def bootstrap_subsample(
    merged: pd.DataFrame,
    collapsed_meta: pd.DataFrame,
    config: CandidateConfig,
    bootstrap_iterations: int,
    bootstrap_fraction: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tumor_cols = participant_columns(collapsed_meta, PRIMARY_TUMOR)
    normal_cols = participant_columns(collapsed_meta, SOLID_NORMAL)
    full_probe_stats = run_probe_statistics(merged, tumor_cols, normal_cols, config=config, paired=False)
    full_regions = build_regions(full_probe_stats, config)

    full_hyper = summarize_group_coverage(full_regions, "hyper")
    full_hypo = summarize_group_coverage(full_regions, "hypo")
    tumor_meta = collapsed_meta[collapsed_meta["sample_type"] == PRIMARY_TUMOR].copy()
    normal_meta = collapsed_meta[collapsed_meta["sample_type"] == SOLID_NORMAL].copy()

    rows = []
    for iteration in range(bootstrap_iterations):
        tumor_keep = max(config.min_samples_per_group, int(round(len(tumor_meta) * bootstrap_fraction)))
        normal_keep = max(config.min_samples_per_group, int(round(len(normal_meta) * bootstrap_fraction)))
        tumor_subset = tumor_meta.sample(n=tumor_keep, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
        normal_subset = normal_meta.sample(n=normal_keep, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
        subset_probe_stats = run_probe_statistics(
            merged,
            tumor_cols=tumor_subset["column_name"].tolist(),
            normal_cols=normal_subset["column_name"].tolist(),
            config=config,
            paired=False,
        )
        subset_regions = build_regions(subset_probe_stats, config)
        rows.append(
            {
                "iteration": iteration,
                "hyper_jaccard_vs_full": jaccard(summarize_group_coverage(subset_regions, "hyper"), full_hyper),
                "hypo_jaccard_vs_full": jaccard(summarize_group_coverage(subset_regions, "hypo"), full_hypo),
                "n_regions": int(len(subset_regions)),
            }
        )
    return full_probe_stats, full_regions, pd.DataFrame(rows)


def paired_sensitivity(
    merged: pd.DataFrame,
    collapsed_meta: pd.DataFrame,
    config: CandidateConfig,
    primary_probe_stats: pd.DataFrame,
    primary_regions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    tumor_cols, normal_cols = matched_column_pairs(collapsed_meta)
    paired_probe_stats = run_probe_statistics(merged, tumor_cols, normal_cols, config=config, paired=True)
    paired_regions = build_regions(paired_probe_stats, config)

    merged_probe_stats = primary_probe_stats[["#id", "delta_beta", "probe_fdr"]].merge(
        paired_probe_stats[["#id", "delta_beta", "probe_fdr"]],
        on="#id",
        suffixes=("_primary", "_paired"),
    )
    corr = merged_probe_stats[["delta_beta_primary", "delta_beta_paired"]].corr(method="spearman").iloc[0, 1]
    sign_agreement = (
        np.sign(merged_probe_stats["delta_beta_primary"]) == np.sign(merged_probe_stats["delta_beta_paired"])
    ).mean()
    strong_primary = merged_probe_stats[
        (merged_probe_stats["probe_fdr_primary"] <= config.probe_fdr)
        & (merged_probe_stats["delta_beta_primary"].abs() >= config.min_abs_delta_beta)
    ]
    strong_sign_agreement = (
        (
            np.sign(strong_primary["delta_beta_primary"]) == np.sign(strong_primary["delta_beta_paired"])
        ).mean()
        if not strong_primary.empty
        else float("nan")
    )

    primary_hyper = summarize_group_coverage(primary_regions, "hyper")
    primary_hypo = summarize_group_coverage(primary_regions, "hypo")
    paired_hyper = summarize_group_coverage(paired_regions, "hyper")
    paired_hypo = summarize_group_coverage(paired_regions, "hypo")
    return paired_probe_stats, paired_regions, {
        "matched_pairs": float(len(tumor_cols)),
        "paired_spearman_delta_beta": float(corr),
        "paired_probe_sign_agreement": float(sign_agreement),
        "paired_strong_probe_sign_agreement": float(strong_sign_agreement),
        "paired_hyper_probe_jaccard": float(jaccard(primary_hyper, paired_hyper)),
        "paired_hypo_probe_jaccard": float(jaccard(primary_hypo, paired_hypo)),
    }


def permutation_sanity(
    merged: pd.DataFrame,
    collapsed_meta: pd.DataFrame,
    config: CandidateConfig,
    permutations: int,
    permutation_seed: int,
) -> pd.DataFrame:
    primary_meta = collapsed_meta[collapsed_meta["sample_type"].isin([PRIMARY_TUMOR, SOLID_NORMAL])].copy()
    all_cols = primary_meta["column_name"].tolist()
    n_tumor = int(primary_meta["sample_type"].eq(PRIMARY_TUMOR).sum())
    rng = np.random.default_rng(permutation_seed)
    rows = []
    for iteration in range(permutations):
        permuted = np.array(all_cols, dtype=object).copy()
        rng.shuffle(permuted)
        tumor_cols = permuted[:n_tumor].tolist()
        normal_cols = permuted[n_tumor:].tolist()
        probe_stats = run_probe_statistics(merged, tumor_cols, normal_cols, config=config, paired=False)
        regions = build_regions(probe_stats, config)
        rows.append(
            {
                "iteration": iteration,
                "n_regions": int(len(regions)),
                "n_hyper_regions": int((regions["direction"] == "hyper").sum()) if not regions.empty else 0,
                "n_hypo_regions": int((regions["direction"] == "hypo").sum()) if not regions.empty else 0,
                "min_region_fdr": float(regions["region_fdr"].min()) if not regions.empty else None,
            }
        )
    return pd.DataFrame(rows)


def compute_research_metrics(
    primary_regions: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    paired_metrics: dict[str, float],
    permutation_df: pd.DataFrame,
) -> dict[str, float]:
    n_regions = int(len(primary_regions))
    mean_hyper_jaccard = float(bootstrap_df["hyper_jaccard_vs_full"].mean()) if not bootstrap_df.empty else 0.0
    mean_hypo_jaccard = float(bootstrap_df["hypo_jaccard_vs_full"].mean()) if not bootstrap_df.empty else 0.0
    mean_stability = (mean_hyper_jaccard + mean_hypo_jaccard) / 2.0
    paired_region_consistency = (
        paired_metrics["paired_hyper_probe_jaccard"] + paired_metrics["paired_hypo_probe_jaccard"]
    ) / 2.0
    paired_consistency = float(
        np.nanmean(
            [
                paired_metrics["paired_strong_probe_sign_agreement"],
                paired_region_consistency,
                paired_metrics["paired_probe_sign_agreement"],
            ]
        )
    )
    mean_permuted_regions = float(permutation_df["n_regions"].mean()) if not permutation_df.empty else 0.0
    permuted_region_ratio = mean_permuted_regions / max(1.0, float(n_regions))
    null_control_score = max(0.0, 1.0 - min(1.0, permuted_region_ratio))
    research_score = 0.50 * mean_stability + 0.35 * paired_consistency + 0.15 * null_control_score
    if n_regions == 0:
        research_score *= 0.5

    metrics = {
        "research_score": float(research_score),
        "mean_stability": float(mean_stability),
        "mean_hyper_jaccard": float(mean_hyper_jaccard),
        "mean_hypo_jaccard": float(mean_hypo_jaccard),
        "paired_region_consistency": float(paired_region_consistency),
        "paired_consistency": float(paired_consistency),
        "null_control_score": float(null_control_score),
        "mean_permuted_regions": float(mean_permuted_regions),
        "permuted_region_ratio": float(permuted_region_ratio),
        "n_regions": n_regions,
        "n_hyper_regions": int((primary_regions["direction"] == "hyper").sum()) if not primary_regions.empty else 0,
        "n_hypo_regions": int((primary_regions["direction"] == "hypo").sum()) if not primary_regions.empty else 0,
    }
    metrics.update({k: float(v) for k, v in paired_metrics.items()})
    return metrics


def sample_summary(sample_meta: pd.DataFrame, collapsed_meta: pd.DataFrame) -> dict[str, object]:
    raw_type_counts = sample_meta["sample_type"].value_counts().sort_index().to_dict()
    collapsed_type_counts = collapsed_meta["sample_type"].value_counts().sort_index().to_dict()
    duplicate_groups = collapsed_meta[collapsed_meta["n_aliquots"] > 1].copy()
    return {
        "n_raw_samples": int(len(sample_meta)),
        "n_raw_participants": int(sample_meta["participant"].nunique()),
        "raw_sample_type_counts": to_builtin(raw_type_counts),
        "collapsed_group_counts": to_builtin(collapsed_type_counts),
        "n_collapsed_groups_with_multiple_aliquots": int(len(duplicate_groups)),
        "max_aliquots_per_group": int(collapsed_meta["n_aliquots"].max()),
    }


def write_bed(region_df: pd.DataFrame, output_path: Path) -> None:
    bed_df = region_df[
        [
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "region_fdr",
            "mean_delta_beta",
            "n_probes",
            "genes",
        ]
    ].copy()
    bed_df.to_csv(output_path, sep="\t", index=False, header=False)


def format_markdown_table(
    frame: pd.DataFrame,
    float_cols: set[str] | None = None,
    scientific_cols: set[str] | None = None,
) -> str:
    if frame.empty:
        return "_No rows_"
    float_cols = float_cols or set()
    scientific_cols = scientific_cols or set()
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            if col in scientific_cols and pd.notna(value):
                cells.append(f"{float(value):.2e}")
            elif col in float_cols and pd.notna(value):
                cells.append(f"{float(value):.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def region_filtering_diagnostics(
    primary_probe_stats: pd.DataFrame,
    config: CandidateConfig,
    final_regions: pd.DataFrame,
) -> dict[str, object]:
    unrestricted_config = CandidateConfig(
        name=f"{config.name}_unrestricted",
        transform=config.transform,
        max_gap=config.max_gap,
        min_region_probes=config.min_region_probes,
        min_samples_per_group=config.min_samples_per_group,
        probe_fdr=config.probe_fdr,
        min_abs_delta_beta=config.min_abs_delta_beta,
        region_fdr_threshold=1.0,
        min_region_abs_mean_delta_beta=0.0,
    )
    unrestricted_regions = build_regions(primary_probe_stats, unrestricted_config)
    if unrestricted_regions.empty:
        return {
            "n_regions_pre_region_filters": 0,
            "n_regions_after_region_fdr": 0,
            "n_regions_post_region_filters": int(len(final_regions)),
            "dropped_by_region_fdr_threshold": 0,
            "dropped_by_min_region_abs_mean_delta_beta": 0,
            "region_fdr_threshold_active": False,
            "min_region_abs_mean_delta_active": False,
            "retained_fraction_after_region_filters": 0.0,
        }

    pass_region_fdr = unrestricted_regions[
        unrestricted_regions["region_fdr"] <= config.region_fdr_threshold
    ].copy()
    pass_all_region_filters = pass_region_fdr[
        pass_region_fdr["mean_delta_beta"].abs() >= config.min_region_abs_mean_delta_beta
    ].copy()
    pre_count = int(len(unrestricted_regions))
    after_region_fdr = int(len(pass_region_fdr))
    post_count = int(len(pass_all_region_filters))
    return {
        "n_regions_pre_region_filters": pre_count,
        "n_regions_after_region_fdr": after_region_fdr,
        "n_regions_post_region_filters": int(len(final_regions)),
        "dropped_by_region_fdr_threshold": int(pre_count - after_region_fdr),
        "dropped_by_min_region_abs_mean_delta_beta": int(after_region_fdr - post_count),
        "region_fdr_threshold_active": bool(pre_count != after_region_fdr),
        "min_region_abs_mean_delta_active": bool(after_region_fdr != post_count),
        "retained_fraction_after_region_filters": float(post_count / max(pre_count, 1)),
    }


def metric_definitions_markdown() -> str:
    return (
        "- `mean_stability = (mean_hyper_jaccard + mean_hypo_jaccard) / 2`\n"
        "- `paired_region_consistency = (paired_hyper_probe_jaccard + paired_hypo_probe_jaccard) / 2`\n"
        "- `paired_consistency = mean(paired_strong_probe_sign_agreement, paired_region_consistency, paired_probe_sign_agreement)`\n"
        "- `permuted_region_ratio = mean_permuted_regions / max(1, n_regions)`\n"
        "- `null_control_score = max(0, 1 - min(1, permuted_region_ratio))`\n"
        "- `research_score = "
        f"{RESEARCH_SCORE_WEIGHTS['mean_stability']:.2f} * mean_stability + "
        f"{RESEARCH_SCORE_WEIGHTS['paired_consistency']:.2f} * paired_consistency + "
        f"{RESEARCH_SCORE_WEIGHTS['null_control_score']:.2f} * null_control_score`\n"
    )


def write_report(
    output_dir: Path,
    config: CandidateConfig,
    metrics: dict[str, float],
    summary: dict[str, object],
    primary_regions: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    region_filtering: dict[str, object],
) -> None:
    top_hyper = primary_regions[primary_regions["direction"] == "hyper"].head(10).copy()
    top_hypo = primary_regions[primary_regions["direction"] == "hypo"].head(10).copy()
    for frame in (top_hyper, top_hypo):
        if not frame.empty:
            frame["span_bp"] = frame["chromEnd"] - frame["chromStart"]
    bootstrap_view = bootstrap_df.copy()
    if not bootstrap_view.empty:
        bootstrap_view["mean_stability"] = (
            bootstrap_view["hyper_jaccard_vs_full"] + bootstrap_view["hypo_jaccard_vs_full"]
        ) / 2.0
    threshold_comment = "active" if region_filtering["region_fdr_threshold_active"] else "inactive"
    region_effect_comment = "active" if region_filtering["min_region_abs_mean_delta_active"] else "inactive"

    report = f"""# Exai Autoresearch Experiment

## Candidate Configuration

`{json.dumps(asdict(config), sort_keys=True)}`

## Methods (Concise)

1. Merge methylation matrix and probe map on `#id`.
2. Decode TCGA barcodes and compare `01` primary tumors vs `11` solid tissue normals.
3. Collapse repeated aliquots to participant-level means.
4. Compute probe-wise differential methylation (Welch-style unpaired t-test on `{config.transform}` scale for primary, paired t-test for matched sensitivity).
5. Keep probes using `probe_fdr <= {config.probe_fdr:.3f}`, `|delta_beta| >= {config.min_abs_delta_beta:.3f}`, and per-group sample minimum `{config.min_samples_per_group}`.
6. Build regions by merging adjacent same-direction probes within `{config.max_gap}` bp, then combine probe evidence with Stouffer z and BH region FDR.
7. Apply region-level post-filters: `region_fdr <= {config.region_fdr_threshold:.3f}` and `|mean_delta_beta| >= {config.min_region_abs_mean_delta_beta:.3f}`.

## Metric Definitions

{metric_definitions_markdown()}

## Main Metrics

- `research_score`: {metrics["research_score"]:.4f}
- `mean_stability`: {metrics["mean_stability"]:.4f}
- `paired_consistency`: {metrics["paired_consistency"]:.4f}
- `null_control_score`: {metrics["null_control_score"]:.4f}
- `n_regions`: {metrics["n_regions"]}
- `n_hyper_regions`: {metrics["n_hyper_regions"]}
- `n_hypo_regions`: {metrics["n_hypo_regions"]}

## Region-Filter Impact

- `region_fdr_threshold` status: `{threshold_comment}` (`dropped {region_filtering["dropped_by_region_fdr_threshold"]}` regions)
- `min_region_abs_mean_delta_beta` status: `{region_effect_comment}` (`dropped {region_filtering["dropped_by_min_region_abs_mean_delta_beta"]}` regions)
- `n_regions_pre_region_filters`: {region_filtering["n_regions_pre_region_filters"]}
- `n_regions_after_region_fdr`: {region_filtering["n_regions_after_region_fdr"]}
- `n_regions_post_region_filters`: {region_filtering["n_regions_post_region_filters"]}
- `retained_fraction_after_region_filters`: {region_filtering["retained_fraction_after_region_filters"]:.4f}

## Bootstrap Stability

{format_markdown_table(bootstrap_view, float_cols={"hyper_jaccard_vs_full", "hypo_jaccard_vs_full", "mean_stability"})}

## Null-Control Stress Test

{format_markdown_table(permutation_df, scientific_cols={"min_region_fdr"})}

`null_control_score` is a sanity metric, not full calibration. Increase permutations for stronger null validation.

## Top Hypermethylated DMRs

{format_markdown_table(top_hyper[["name", "chrom", "chromStart", "chromEnd", "span_bp", "n_probes", "mean_delta_beta", "region_fdr", "genes"]], float_cols={"mean_delta_beta"}, scientific_cols={"region_fdr"})}

## Top Hypomethylated DMRs

{format_markdown_table(top_hypo[["name", "chrom", "chromStart", "chromEnd", "span_bp", "n_probes", "mean_delta_beta", "region_fdr", "genes"]], float_cols={"mean_delta_beta"}, scientific_cols={"region_fdr"})}

## Sample Summary

`{json.dumps(summary, sort_keys=True)}`

Region FDR values are shown in scientific notation to avoid hiding very small non-zero values as `0.0000`.
"""
    (output_dir / "report.md").write_text(report)


def run_one_experiment(
    merged: pd.DataFrame,
    sample_meta: pd.DataFrame,
    collapsed_meta: pd.DataFrame,
    config: CandidateConfig,
    output_dir: Path,
    bootstrap_iterations: int,
    bootstrap_fraction: float,
    permutations: int,
    random_seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    primary_probe_stats, primary_regions, bootstrap_df = bootstrap_subsample(
        merged=merged,
        collapsed_meta=collapsed_meta,
        config=config,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_fraction=bootstrap_fraction,
        rng=rng,
    )
    paired_probe_stats, paired_regions, paired_metrics = paired_sensitivity(
        merged=merged,
        collapsed_meta=collapsed_meta,
        config=config,
        primary_probe_stats=primary_probe_stats,
        primary_regions=primary_regions,
    )
    permutation_df = permutation_sanity(
        merged=merged,
        collapsed_meta=collapsed_meta,
        config=config,
        permutations=permutations,
        permutation_seed=random_seed + 1,
    )
    metrics = compute_research_metrics(primary_regions, bootstrap_df, paired_metrics, permutation_df)
    region_filtering = region_filtering_diagnostics(primary_probe_stats, config, primary_regions)
    metrics.update(region_filtering)
    summary = sample_summary(sample_meta, collapsed_meta)

    hyper_regions = primary_regions[primary_regions["direction"] == "hyper"].copy()
    hypo_regions = primary_regions[primary_regions["direction"] == "hypo"].copy()
    write_bed(hyper_regions, output_dir / "hypermethylated_dmrs.bed")
    write_bed(hypo_regions, output_dir / "hypomethylated_dmrs.bed")

    primary_probe_stats.to_csv(output_dir / "probe_stats_primary.tsv.gz", sep="\t", index=False)
    paired_probe_stats.to_csv(output_dir / "probe_stats_paired.tsv.gz", sep="\t", index=False)
    primary_regions.to_csv(output_dir / "dmrs_primary.tsv.gz", sep="\t", index=False)
    paired_regions.to_csv(output_dir / "dmrs_paired.tsv.gz", sep="\t", index=False)
    bootstrap_df.to_csv(output_dir / "bootstrap.tsv", sep="\t", index=False)
    permutation_df.to_csv(output_dir / "permutation_sanity.tsv", sep="\t", index=False)

    payload = {
        "config": to_builtin(asdict(config)),
        "metrics": to_builtin(metrics),
        "sample_summary": to_builtin(summary),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_report(output_dir, config, metrics, summary, primary_regions, bootstrap_df, permutation_df, region_filtering)
    return payload


def config_signature(config: CandidateConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True)


def sanitize_candidate(base: CandidateConfig, candidate_dict: dict[str, object]) -> CandidateConfig:
    valid_transforms = {"beta", "m"}
    def pick(name, default):
        return candidate_dict.get(name, default)

    transform = str(pick("transform", base.transform))
    if transform not in valid_transforms:
        transform = base.transform

    max_gap = int(pick("max_gap", base.max_gap))
    min_region_probes = int(pick("min_region_probes", base.min_region_probes))
    min_samples_per_group = int(pick("min_samples_per_group", base.min_samples_per_group))
    probe_fdr = float(pick("probe_fdr", base.probe_fdr))
    min_abs_delta_beta = float(pick("min_abs_delta_beta", base.min_abs_delta_beta))
    region_fdr_threshold = float(pick("region_fdr_threshold", base.region_fdr_threshold))
    min_region_abs_mean_delta_beta = float(
        pick("min_region_abs_mean_delta_beta", base.min_region_abs_mean_delta_beta)
    )

    max_gap = min(max(max_gap, 100), 5000)
    min_region_probes = min(max(min_region_probes, 2), 8)
    min_samples_per_group = min(max(min_samples_per_group, 2), 10)
    probe_fdr = min(max(probe_fdr, 1e-4), 0.25)
    min_abs_delta_beta = min(max(min_abs_delta_beta, 0.02), 0.30)
    region_fdr_threshold = min(max(region_fdr_threshold, 1e-4), 1.0)
    min_region_abs_mean_delta_beta = min(max(min_region_abs_mean_delta_beta, 0.0), 0.20)

    name = (
        f"{transform}_gap{max_gap}_p{min_region_probes}_s{min_samples_per_group}"
        f"_fdr{probe_fdr:.3f}_db{min_abs_delta_beta:.3f}_rfdr{region_fdr_threshold:.3f}"
        f"_rdb{min_region_abs_mean_delta_beta:.3f}"
    )
    return CandidateConfig(
        name=name,
        transform=transform,
        max_gap=max_gap,
        min_region_probes=min_region_probes,
        min_samples_per_group=min_samples_per_group,
        probe_fdr=probe_fdr,
        min_abs_delta_beta=min_abs_delta_beta,
        region_fdr_threshold=region_fdr_threshold,
        min_region_abs_mean_delta_beta=min_region_abs_mean_delta_beta,
    )


def configure_openai():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI(api_key=api_key)


def propose_with_openai(
    client,
    model: str,
    base_config: CandidateConfig,
    history: list[dict[str, object]],
) -> tuple[CandidateConfig | None, str]:
    prompt = {
        "base_config": asdict(base_config),
        "recent_results": history[-3:],
        "allowed_fields": [
            "transform",
            "max_gap",
            "min_region_probes",
            "min_samples_per_group",
            "probe_fdr",
            "min_abs_delta_beta",
            "region_fdr_threshold",
            "min_region_abs_mean_delta_beta",
        ],
        "instructions": [
            "Propose one small config change only.",
            "Optimize for research_score, mean_stability, paired_consistency, and null_control_score.",
            "Do not optimize for raw region count alone.",
            "Return strict JSON with keys: change_reason, candidate.",
        ],
    }
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": "You are a careful autoresearch assistant. Return strict JSON only.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt, indent=2, sort_keys=True),
            },
        ],
    )
    text = response.output_text or ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None, text
    parsed = json.loads(text[start : end + 1])
    candidate = sanitize_candidate(base_config, parsed.get("candidate", {}))
    return candidate, text


def fallback_candidates(base: CandidateConfig) -> list[CandidateConfig]:
    deltas = [
        {"max_gap": 500},
        {"max_gap": 750},
        {"max_gap": 1500},
        {"transform": "m"},
        {"probe_fdr": 0.03},
        {"probe_fdr": 0.08},
        {"min_abs_delta_beta": 0.08},
        {"min_abs_delta_beta": 0.12},
        {"min_region_probes": 4},
        {"min_region_probes": 5},
        {"region_fdr_threshold": 0.25},
        {"min_region_abs_mean_delta_beta": 0.02},
    ]
    return [sanitize_candidate(base, delta) for delta in deltas]


def append_lab_notebook(path: Path, text: str) -> None:
    if not path.exists():
        path.write_text("# Lab Notebook\n\n")
    with path.open("a") as handle:
        handle.write(text)


def write_history_tsv(history: list[dict[str, object]], path: Path) -> None:
    if not history:
        return
    headers = sorted({key for row in history for key in row.keys()})
    lines = ["\t".join(headers)]
    for row in history:
        lines.append("\t".join(str(row.get(h, "")) for h in headers))
    path.write_text("\n".join(lines) + "\n")


def write_selection_artifacts(output_dir: Path, history: list[dict[str, object]], best_config: CandidateConfig) -> None:
    scored = [row for row in history if "research_score" in row]
    if not scored:
        return
    table = pd.DataFrame(scored).copy()
    table = table.sort_values("research_score", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table["margin_vs_best"] = table["research_score"].iloc[0] - table["research_score"]
    columns = [
        "rank",
        "iteration",
        "label",
        "desc",
        "status",
        "research_score",
        "mean_stability",
        "paired_consistency",
        "null_control_score",
        "n_regions",
        "config_name",
        "transform",
        "max_gap",
        "min_region_probes",
        "min_samples_per_group",
        "probe_fdr",
        "min_abs_delta_beta",
        "region_fdr_threshold",
        "min_region_abs_mean_delta_beta",
        "margin_vs_best",
    ]
    available_columns = [col for col in columns if col in table.columns]
    leaderboard = table[available_columns].copy()
    leaderboard.to_csv(output_dir / "selection_leaderboard.tsv", sep="\t", index=False)

    best_score = float(leaderboard["research_score"].iloc[0])
    if len(leaderboard) > 1:
        runner_up_score = float(leaderboard["research_score"].iloc[1])
        winner_margin = best_score - runner_up_score
        runner_up_display = f"{runner_up_score:.6f}"
        winner_margin_display = f"{winner_margin:.6f}"
    else:
        runner_up_score = None
        winner_margin = None
        runner_up_display = "N/A"
        winner_margin_display = "N/A"

    top_view = leaderboard.copy()
    markdown = f"""# Autoresearch Selection Report

## Winner

- `best_config_name`: `{best_config.name}`
- `best_research_score`: {best_score:.6f}
- `runner_up_research_score`: {runner_up_display}
- `winner_margin`: {winner_margin_display}

## Candidate Ranking

{format_markdown_table(top_view, float_cols={"research_score", "mean_stability", "paired_consistency", "null_control_score", "margin_vs_best"})}
"""
    (output_dir / "selection_report.md").write_text(markdown)
    summary_payload = {
        "best_config": asdict(best_config),
        "best_research_score": best_score,
        "runner_up_research_score": runner_up_score,
        "winner_margin": winner_margin,
        "n_scored_candidates": int(len(leaderboard)),
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(to_builtin(summary_payload), indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    methylation_path = Path(args.methylation)
    probemap_path = Path(args.probemap)
    missing_inputs = [str(path) for path in [methylation_path, probemap_path] if not path.exists()]
    if missing_inputs:
        print("Missing input file(s):")
        for path in missing_inputs:
            print(f"  - {path}")
        print("Upload the two TSV files to /content or pass --methylation/--probemap explicitly.")
        return

    root_output_dir = Path(args.output_dir).resolve()
    root_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = resolve_attempt_dir(root_output_dir, args.attempt_label)
    if output_dir.exists():
        print(f"Attempt directory already exists: {output_dir}")
        print("Use a new --attempt-label or remove the existing attempt directory.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    history_tsv_path = output_dir / "history.tsv"
    notebook_path = output_dir / "lab_notebook.md"
    best_metrics_path = output_dir / "best_metrics.json"
    best_config_path = output_dir / "best_config.json"

    merged, sample_meta, collapsed_meta = load_inputs(methylation_path, probemap_path)
    client = configure_openai()

    best_config = CandidateConfig()
    best_score = float("-inf")
    history: list[dict[str, object]] = []
    tried_signatures: set[str] = set()
    start_time = time.time()
    max_seconds = args.target_minutes * 60.0 if args.target_minutes > 0 else None
    iteration = 0

    while iteration < args.iterations:
        if max_seconds is not None and iteration > 0 and (time.time() - start_time) >= max_seconds:
            break
        label = "baseline" if iteration == 0 else f"iter_{iteration:03d}"
        iter_dir = output_dir / label

        if iteration == 0:
            candidate = best_config
            desc = "baseline"
            raw_model_response = ""
        else:
            candidate = None
            raw_model_response = ""
            if client is not None:
                try:
                    candidate, raw_model_response = propose_with_openai(client, args.model, best_config, history)
                    desc = "openai_small_config_edit"
                except Exception as exc:
                    desc = f"openai_error:{type(exc).__name__}"
            else:
                desc = "fallback_config_edit"

            if candidate is None or config_signature(candidate) in tried_signatures:
                for fallback in fallback_candidates(best_config):
                    if config_signature(fallback) not in tried_signatures:
                        candidate = fallback
                        if desc.startswith("openai_error"):
                            desc = desc + "+fallback"
                        else:
                            desc = "fallback_config_edit"
                        break

            if candidate is None:
                row = {"iteration": iteration, "label": label, "desc": "no_candidates_left", "status": "skipped"}
                history.append(row)
                with history_path.open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                continue

        tried_signatures.add(config_signature(candidate))
        if raw_model_response:
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "model_response.txt").write_text(raw_model_response)

        try:
            payload = run_one_experiment(
                merged=merged,
                sample_meta=sample_meta,
                collapsed_meta=collapsed_meta,
                config=candidate,
                output_dir=iter_dir,
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_fraction=args.bootstrap_fraction,
                permutations=args.permutations,
                random_seed=args.random_seed + iteration,
            )
        except Exception as exc:
            row = {
                "iteration": iteration,
                "label": label,
                "desc": desc,
                "status": "crash",
                "error": f"{type(exc).__name__}: {exc}",
            }
            history.append(row)
            with history_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            continue

        metrics = payload["metrics"]
        score = float(metrics["research_score"])
        keep = score > best_score
        if keep:
            best_score = score
            best_config = candidate
            best_metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            best_config_path.write_text(json.dumps(asdict(best_config), indent=2, sort_keys=True))
            shutil.copyfile(iter_dir / "report.md", output_dir / "best_report.md")
            shutil.copyfile(iter_dir / "hypermethylated_dmrs.bed", output_dir / "best_hypermethylated_dmrs.bed")
            shutil.copyfile(iter_dir / "hypomethylated_dmrs.bed", output_dir / "best_hypomethylated_dmrs.bed")

        row = {
            "iteration": iteration,
            "label": label,
            "desc": desc,
            "status": "kept" if keep else "discarded",
            "research_score": score,
            "mean_stability": metrics["mean_stability"],
            "paired_consistency": metrics["paired_consistency"],
            "null_control_score": metrics["null_control_score"],
            "n_regions": metrics["n_regions"],
            "config_name": candidate.name,
            "transform": candidate.transform,
            "max_gap": candidate.max_gap,
            "min_region_probes": candidate.min_region_probes,
            "min_samples_per_group": candidate.min_samples_per_group,
            "probe_fdr": candidate.probe_fdr,
            "min_abs_delta_beta": candidate.min_abs_delta_beta,
            "region_fdr_threshold": candidate.region_fdr_threshold,
            "min_region_abs_mean_delta_beta": candidate.min_region_abs_mean_delta_beta,
        }
        history.append(row)
        with history_path.open("a") as handle:
            handle.write(json.dumps(to_builtin(row), sort_keys=True) + "\n")
        append_lab_notebook(
            notebook_path,
            (
                f"## {label}\n\n"
                f"- Change: {desc}\n"
                f"- Config: `{json.dumps(asdict(candidate), sort_keys=True)}`\n"
                f"- Status: {'kept' if keep else 'discarded'}\n"
                f"- research_score: {score:.4f}\n"
                f"- mean_stability: {metrics['mean_stability']:.4f}\n"
                f"- paired_consistency: {metrics['paired_consistency']:.4f}\n"
                f"- null_control_score: {metrics['null_control_score']:.4f}\n"
                f"- n_regions: {metrics['n_regions']}\n\n"
            ),
        )
        write_history_tsv(history, history_tsv_path)
        write_selection_artifacts(output_dir, history, best_config)
        print(json.dumps(to_builtin(row), sort_keys=True))
        elapsed_min = (time.time() - start_time) / 60.0
        print(f"elapsed_minutes: {elapsed_min:.2f}")
        iteration += 1

    print(f"attempt_dir: {output_dir}")
    print(f"best_research_score: {best_score:.6f}")
    print(f"best_config: {json.dumps(asdict(best_config), sort_keys=True)}")
    print(f"total_elapsed_minutes: {(time.time() - start_time) / 60.0:.2f}")


if __name__ == "__main__":
    main()
