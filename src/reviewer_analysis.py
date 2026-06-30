"""
Post-hoc reviewer analyses from exported OOF artefacts.

Loads reviewer_analysis_export folders produced by dataset front scripts and
generates ablation, threshold sensitivity, calibration, statistical, and energy
summaries without retraining or feature extraction.
"""

import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from . import config, fairness, routing

DATASET_NAMES = ["HAM10000", "BCN20000", "PAD-UFES-20"]
THRESHOLD_MULTIPLIERS = [0.80, 0.90, 1.00, 1.10, 1.20]
BOOTSTRAP_N = 1000


# ---------------------------------------------------------------------------
# Export path resolution
# ---------------------------------------------------------------------------

def reviewer_export_subdir(pipeline_output_dir):
    """Return reviewer_analysis_export path under a pipeline output directory."""
    if pipeline_output_dir.rstrip("/").endswith("reviewer_analysis_export"):
        return pipeline_output_dir.rstrip("/")
    return os.path.join(pipeline_output_dir, "reviewer_analysis_export")


def resolve_export_directories(explicit_pipeline_dirs=None):
    """
    Resolve reviewer_analysis_export folders for all datasets.

    Args:
        explicit_pipeline_dirs: optional dict mapping dataset name to either
            pipeline_results/{DATASET} dir or reviewer_analysis_export dir.

    Returns:
        dict: {dataset_name: export_dir}
    """
    resolved = {}

    if explicit_pipeline_dirs:
        for dataset, path in explicit_pipeline_dirs.items():
            export_dir = reviewer_export_subdir(path)
            if os.path.isdir(export_dir):
                resolved[dataset] = export_dir

    for dataset in DATASET_NAMES:
        if dataset in resolved:
            continue
        local = os.path.join("./output/pipeline_results", dataset, "reviewer_analysis_export")
        if os.path.isdir(local):
            resolved[dataset] = local

    if os.path.isdir("/kaggle/input"):
        patterns = [
            "/kaggle/input/notebooks/**/output/pipeline_results/*/reviewer_analysis_export",
            "/kaggle/input/**/output/pipeline_results/*/reviewer_analysis_export",
        ]
        for pattern in patterns:
            for export_dir in glob.glob(pattern, recursive=True):
                dataset = os.path.basename(os.path.dirname(export_dir))
                if dataset in DATASET_NAMES and dataset not in resolved:
                    resolved[dataset] = export_dir

        # Fallback: locate pipeline output via cv_results.csv, then append reviewer export subdir
        for cv_path in glob.glob("/kaggle/input/**/pipeline_results/*/cv_results.csv", recursive=True):
            pipeline_dir = os.path.dirname(cv_path)
            dataset = os.path.basename(pipeline_dir)
            if dataset not in DATASET_NAMES or dataset in resolved:
                continue
            export_dir = reviewer_export_subdir(pipeline_dir)
            if os.path.isdir(export_dir):
                resolved[dataset] = export_dir

    return resolved


def format_expected_paths(resolved=None):
    """Human-readable hint for where exports should live."""
    lines = ["Expected reviewer export folders:"]
    if resolved:
        for ds, path in sorted(resolved.items()):
            lines.append(f"  [{ds}] {path}")
    else:
        lines.append("  ./output/pipeline_results/{DATASET}/reviewer_analysis_export")
        lines.append("  /kaggle/input/.../output/pipeline_results/{DATASET}/reviewer_analysis_export")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset_exports(export_dir, dataset_name):
    """Load manifests and tables from one reviewer_analysis_export folder."""
    if not os.path.isdir(export_dir):
        warnings.warn(f"Missing export folder: {export_dir}")
        return None
    paths = {
        "fold_metrics": os.path.join(export_dir, "fold_metrics.csv"),
        "pair_manifest": os.path.join(export_dir, "pair_manifest.csv"),
        "sample_metadata": os.path.join(export_dir, "sample_metadata.csv"),
        "energy_stats": os.path.join(export_dir, "energy_stats.csv"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        warnings.warn(f"{dataset_name}: missing files {missing} in {export_dir}")
        return None
    return {
        "dataset": dataset_name,
        "export_dir": export_dir,
        "fold_metrics": pd.read_csv(paths["fold_metrics"]),
        "pair_manifest": pd.read_csv(paths["pair_manifest"]),
        "sample_metadata": pd.read_csv(paths["sample_metadata"]),
        "energy_stats": pd.read_csv(paths["energy_stats"]),
    }


def load_all_exports(export_dirs_by_dataset):
    """Load exports for every resolved dataset path."""
    exports = []
    for dataset in DATASET_NAMES:
        export_dir = export_dirs_by_dataset.get(dataset)
        if export_dir is None:
            warnings.warn(f"No export path resolved for {dataset}")
            exports.append(None)
            continue
        exports.append(load_dataset_exports(export_dir, dataset))
    return exports


def load_pair_npz(export_dir, pair_index):
    path = os.path.join(export_dir, f"pair_{pair_index:02d}_oof.npz")
    if not os.path.exists(path):
        warnings.warn(f"Missing NPZ: {path}")
        return None
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _parse_list_field(value):
    if pd.isna(value) or value == "":
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def _manifest_row(manifest_df, pair_index):
    rows = manifest_df[manifest_df["pair_index"] == pair_index]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "pair_index": int(row["pair_index"]),
        "pair_label": row.get("pair_label", ""),
        "class_names": _parse_list_field(row.get("class_names")),
        "safe_classes": _parse_list_field(row.get("safe_classes")),
        "dangerous_classes": _parse_list_field(row.get("dangerous_classes")),
    }


def _danger_indices(class_names, dangerous_classes):
    return [class_names.index(c) for c in dangerous_classes if c in class_names]


def malignant_recall(y_true, y_pred, danger_indices):
    if not danger_indices:
        return np.nan
    return float(recall_score(y_true, y_pred, labels=danger_indices, average="macro", zero_division=0))


def classification_metrics(y_true, y_pred_probs, class_names, dangerous_classes):
    y_pred = np.argmax(y_pred_probs, axis=1)
    danger_idx = _danger_indices(class_names, dangerous_classes)
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "malignant_recall": malignant_recall(y_true, y_pred, danger_idx),
    }


def fairness_tpr_summary(y_true, y_pred_probs, meta_df, class_names, dangerous_classes):
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = fairness.generate_fairness_report(y_true, y_pred, meta_df, class_names)
    if report is None or report.empty:
        return {"worst_group_tpr": np.nan, "tpr_gap": np.nan}
    subset = report[report["Class"].isin(dangerous_classes)].copy()
    subset["_tpr"] = pd.to_numeric(subset["Equal_Opportunity_TPR"], errors="coerce")
    sg_macro = subset.groupby("Subgroup")["_tpr"].mean().dropna()
    if sg_macro.empty:
        return {"worst_group_tpr": np.nan, "tpr_gap": np.nan}
    return {
        "worst_group_tpr": float(sg_macro.min()),
        "tpr_gap": float(sg_macro.max() - sg_macro.min()),
    }


def fuse_predictions(oof_lite, oof_heavy, route_mask, heavy_weight):
    hw = np.asarray(heavy_weight, dtype=np.float32)
    final = oof_lite.copy()
    routed = route_mask.astype(bool)
    if not routed.any():
        return final
    if hw.ndim == 0 or hw.size == 1:
        w = float(hw.ravel()[0])
        final[routed] = (1 - w) * oof_lite[routed] + w * oof_heavy[routed]
    else:
        w = hw[routed][:, None]
        final[routed] = (1 - w) * oof_lite[routed] + w * oof_heavy[routed]
    return final


def energy_per_sample(route_rate, joules_lite, joules_heavy):
    if np.isnan(joules_lite) or np.isnan(joules_heavy):
        return np.nan
    return float(joules_lite + route_rate * joules_heavy)


def _pair_energy_joules(energy_df, pair_index):
    sub = energy_df[energy_df["pair_index"] == pair_index]
    lite = sub[sub["model_role"] == "Lite"]
    heavy = sub[sub["model_role"] == "Heavy"]
    j_lite = float(lite["joules_per_sample"].iloc[0]) if len(lite) and pd.notna(lite["joules_per_sample"].iloc[0]) else np.nan
    j_heavy = float(heavy["joules_per_sample"].iloc[0]) if len(heavy) and pd.notna(heavy["joules_per_sample"].iloc[0]) else np.nan
    return j_lite, j_heavy


def _pair_latency_ms(energy_df, pair_index):
    sub = energy_df[energy_df["pair_index"] == pair_index]
    lite = sub[sub["model_role"] == "Lite"]
    heavy = sub[sub["model_role"] == "Heavy"]
    lat_lite = float(lite["latency_per_sample_mean_ms"].iloc[0]) if len(lite) and pd.notna(lite["latency_per_sample_mean_ms"].iloc[0]) else np.nan
    lat_heavy = float(heavy["latency_per_sample_mean_ms"].iloc[0]) if len(heavy) and pd.notna(heavy["latency_per_sample_mean_ms"].iloc[0]) else np.nan
    return lat_lite, lat_heavy


def _embedding_kb(energy_df, pair_index, role):
    sub = energy_df[(energy_df["pair_index"] == pair_index) & (energy_df["model_role"] == role)]
    if sub.empty:
        return np.nan
    dim = sub["embedding_dim"].iloc[0]
    if pd.isna(dim):
        return np.nan
    return float(dim) * 4.0 / 1024.0


def multiclass_ece(y_true, probs, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if not mask.any():
            continue
        ece += mask.mean() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def multiclass_brier(y_true, probs, n_classes):
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def df_to_latex(df, caption=""):
    try:
        tex = df.to_latex(index=False, float_format="%.4f", escape=True)
    except Exception:
        tex = df.to_string(index=False)
    header = f"% {caption}\n" if caption else ""
    return header + tex + "\n"


def save_results_table(df, out_dir, csv_name, latex_name, caption=""):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    latex_path = os.path.join(out_dir, latex_name)
    with open(latex_path, "w") as f:
        f.write(df_to_latex(df, caption=caption))
    return csv_path, latex_path


def _evaluate_variant(name, probs, route_mask, y_true, meta_df, class_names, dangerous_classes, joules_lite, joules_heavy):
    route_rate = float(np.mean(route_mask)) if route_mask is not None else np.nan
    if name == "Lite only":
        route_rate, energy = 0.0, joules_lite
    elif name == "Heavy only":
        route_rate, energy = 1.0, joules_heavy
    else:
        energy = energy_per_sample(route_rate, joules_lite, joules_heavy)
    cls = classification_metrics(y_true, probs, class_names, dangerous_classes)
    fair = fairness_tpr_summary(y_true, probs, meta_df, class_names, dangerous_classes)
    return {"routing_variant": name, "routing_rate": route_rate, "expected_energy_per_sample": energy, **cls, **fair}


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def run_routing_ablation(exports):
    rows = []
    for exp in exports:
        if exp is None:
            continue
        dataset = exp["dataset"]
        meta_df = exp["sample_metadata"]
        energy_df = exp["energy_stats"]
        manifest = exp["pair_manifest"]
        for pair_index in sorted(manifest["pair_index"].unique()):
            info = _manifest_row(manifest, pair_index)
            if info is None:
                continue
            data = load_pair_npz(exp["export_dir"], pair_index)
            if data is None:
                continue
            class_names = info["class_names"]
            dangerous = info["dangerous_classes"]
            y_true = data["y_true"].astype(int)
            oof_lite, oof_heavy, oof_ecofair = data["oof_lite"], data["oof_heavy"], data["oof_ecofair"]
            ent_t, gap_t, risk_t, hw = data["entropy_threshold"], data["gap_threshold"], data["risk_threshold"], data["heavy_weight"]
            entropy, gap, risk = data["entropy"], data["safe_danger_gap"], data["patient_risk"]
            j_lite, j_heavy = _pair_energy_joules(energy_df, pair_index)
            valid_risk = np.isfinite(risk) & np.isfinite(risk_t)
            risk_cond = np.zeros(len(y_true), dtype=bool)
            if valid_risk.any():
                risk_cond[valid_risk] = risk[valid_risk] > risk_t[valid_risk]
            variants = [
                ("Lite only", oof_lite, np.zeros(len(y_true), dtype=bool)),
                ("Heavy only", oof_heavy, np.ones(len(y_true), dtype=bool)),
                ("Entropy-only routing", fuse_predictions(oof_lite, oof_heavy, entropy > ent_t, hw), entropy > ent_t),
                ("Safe-danger-gap-only routing", fuse_predictions(oof_lite, oof_heavy, gap < gap_t, hw), gap < gap_t),
                ("Metadata-risk-only routing", fuse_predictions(oof_lite, oof_heavy, risk_cond, hw), risk_cond),
                ("Entropy + gap (no metadata risk)", fuse_predictions(oof_lite, oof_heavy, (entropy > ent_t) | (gap < gap_t), hw), (entropy > ent_t) | (gap < gap_t)),
            ]
            full_mask = data["route_mask"].astype(bool)
            full_recon = fuse_predictions(oof_lite, oof_heavy, full_mask, hw)
            if not np.allclose(full_recon, oof_ecofair, atol=1e-5, rtol=1e-4):
                warnings.warn(
                    f"{dataset} pair {pair_index}: using stored oof_ecofair (reconstruction mismatch)"
                )
                full_probs = oof_ecofair
            else:
                full_probs = full_recon
            variants.append(("Full EcoFair routing", full_probs, full_mask))
            for vname, probs, rmask in variants:
                row = _evaluate_variant(vname, probs, rmask, y_true, meta_df, class_names, dangerous, j_lite, j_heavy)
                row.update({"dataset": dataset, "pair_index": pair_index, "pair_label": info["pair_label"]})
                rows.append(row)
    return pd.DataFrame(rows)


def _infer_routing_conditions(data):
    ent_t, gap_t, risk_t = data["entropy_threshold"], data["gap_threshold"], data["risk_threshold"]
    entropy, gap, risk = data["entropy"], data["safe_danger_gap"], data["patient_risk"]
    valid_risk = np.isfinite(risk) & np.isfinite(risk_t)
    risk_pred = np.zeros(len(entropy), dtype=bool)
    if valid_risk.any():
        risk_pred[valid_risk] = risk[valid_risk] > risk_t[valid_risk]
    return {
        "entropy_agreement": float(np.mean((entropy > ent_t) == data["route_uncertainty"])),
        "gap_agreement": float(np.mean((gap < gap_t) == data["route_ambiguity"])),
        "risk_agreement": float(np.mean(risk_pred == data["route_risk"])),
    }


def _apply_threshold_routing_from_arrays(data, mult=1.0):
    ent_t = data["entropy_threshold"] * mult
    gap_t = data["gap_threshold"] * mult
    risk_t = data["risk_threshold"] * mult
    valid_risk = np.isfinite(data["patient_risk"]) & np.isfinite(risk_t)
    route_ent = data["entropy"] > ent_t
    route_gap = data["safe_danger_gap"] < gap_t
    route_risk = np.zeros(len(route_ent), dtype=bool)
    if valid_risk.any():
        route_risk[valid_risk] = data["patient_risk"][valid_risk] > risk_t[valid_risk]
    route_mask = route_ent | route_gap | route_risk
    probs = fuse_predictions(data["oof_lite"], data["oof_heavy"], route_mask, data["heavy_weight"])
    return probs, route_mask


def run_threshold_sensitivity(exports):
    rows, condition_rows = [], []
    for exp in exports:
        if exp is None:
            continue
        dataset = exp["dataset"]
        meta_df = exp["sample_metadata"]
        energy_df = exp["energy_stats"]
        for pair_index in sorted(exp["pair_manifest"]["pair_index"].unique()):
            info = _manifest_row(exp["pair_manifest"], pair_index)
            data = load_pair_npz(exp["export_dir"], pair_index)
            if info is None or data is None:
                continue
            cond = _infer_routing_conditions(data)
            cond.update({"dataset": dataset, "pair_index": pair_index, "pair_label": info["pair_label"]})
            condition_rows.append(cond)
            y_true = data["y_true"].astype(int)
            j_lite, j_heavy = _pair_energy_joules(energy_df, pair_index)
            for mult in THRESHOLD_MULTIPLIERS:
                probs, route_mask = _apply_threshold_routing_from_arrays(data, mult=mult)
                cls = classification_metrics(y_true, probs, info["class_names"], info["dangerous_classes"])
                fair = fairness_tpr_summary(y_true, probs, meta_df, info["class_names"], info["dangerous_classes"])
                rows.append({
                    "dataset": dataset, "pair_index": pair_index, "pair_label": info["pair_label"],
                    "threshold_multiplier": mult, "routing_rate": float(route_mask.mean()),
                    "expected_energy_per_sample": energy_per_sample(float(route_mask.mean()), j_lite, j_heavy),
                    **cls, **fair,
                })
    if condition_rows:
        print("\nThreshold condition reconstruction agreement (1.0 = exact match):")
        print(pd.DataFrame(condition_rows)[["dataset", "pair_index", "entropy_agreement", "gap_agreement", "risk_agreement"]].to_string(index=False))
    return pd.DataFrame(rows)


def plot_threshold_sensitivity(thresh_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plot_paths = []
    if thresh_df is None or thresh_df.empty or "dataset" not in thresh_df.columns:
        return plot_paths
    metrics = [
        ("routing_rate", "Routing rate"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("malignant_recall", "Malignant recall"),
        ("expected_energy_per_sample", "Energy per sample (J)"),
    ]
    for dataset in thresh_df["dataset"].unique():
        sub = thresh_df[thresh_df["dataset"] == dataset]
        agg = sub.groupby("threshold_multiplier")[[m[0] for m in metrics]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        for col, label in metrics:
            ax.plot(agg["threshold_multiplier"], agg[col], marker="o", label=label)
        ax.set_xlabel("Threshold multiplier")
        ax.set_title(f"Threshold sensitivity — {dataset}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"threshold_sensitivity_{dataset.replace('-', '_')}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)
    return plot_paths


def run_uncertainty_quality(exports):
    rows = []
    for exp in exports:
        if exp is None:
            continue
        for pair_index in sorted(exp["pair_manifest"]["pair_index"].unique()):
            info = _manifest_row(exp["pair_manifest"], pair_index)
            data = load_pair_npz(exp["export_dir"], pair_index)
            if info is None or data is None:
                continue
            y_true = data["y_true"].astype(int)
            n_classes = len(info["class_names"])
            for model_name, probs in [("Lite", data["oof_lite"]), ("Heavy", data["oof_heavy"]), ("EcoFair", data["oof_ecofair"])]:
                y_pred = probs.argmax(axis=1)
                wrong = (y_pred != y_true).astype(int)
                entropy_vals = routing.calculate_entropy(probs)
                auroc = float(roc_auc_score(wrong, entropy_vals)) if len(np.unique(wrong)) > 1 else np.nan
                correct = y_pred == y_true
                rows.append({
                    "dataset": exp["dataset"], "pair_index": pair_index, "pair_label": info["pair_label"],
                    "model": model_name, "ece": multiclass_ece(y_true, probs),
                    "brier_score": multiclass_brier(y_true, probs, n_classes),
                    "entropy_auroc_lite_error": auroc if model_name == "Lite" else np.nan,
                    "mean_entropy_correct": float(entropy_vals[correct].mean()) if correct.any() else np.nan,
                    "mean_entropy_incorrect": float(entropy_vals[~correct].mean()) if (~correct).any() else np.nan,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    lite_key = out[out["model"] == "Lite"][["dataset", "pair_index", "entropy_auroc_lite_error"]].rename(
        columns={"entropy_auroc_lite_error": "lite_entropy_auroc"})
    return out.merge(lite_key, on=["dataset", "pair_index"], how="left")


def _paired_tests(values_a, values_b):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"mean_diff": np.nan, "paired_t_pvalue": np.nan, "wilcoxon_pvalue": np.nan, "cohens_dz": np.nan}
    diff = a - b
    t_p = float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        w_p = np.nan
    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan
    return {"mean_diff": float(diff.mean()), "paired_t_pvalue": t_p, "wilcoxon_pvalue": w_p, "cohens_dz": dz}


def _bootstrap_ci(y_true, y_pred, metric_fn, n_boot=BOOTSTRAP_N, seed=None):
    seed = config.RANDOM_STATE if seed is None else seed
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [metric_fn(y_true[rng.integers(0, n, n)], y_pred[rng.integers(0, n, n)]) for _ in range(n_boot)]
    scores = np.asarray(scores)
    return float(scores.mean()), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def run_statistical_tests(exports):
    fold_rows, boot_rows = [], []
    fold_metric_map = [
        ("balanced_accuracy", "balanced_acc_ecofair", "balanced_acc_lite", "balanced_acc_heavy"),
        ("macro_f1", "macro_f1_ecofair", "macro_f1_lite", "macro_f1_heavy"),
        ("malignant_recall", "malignant_recall_ecofair", "malignant_recall_lite", "malignant_recall_heavy"),
        ("expected_energy_per_sample", "energy_ecofair_per_sample", "energy_lite_per_sample", "energy_heavy_per_sample"),
    ]
    for exp in exports:
        if exp is None:
            continue
        fm, meta_df = exp["fold_metrics"], exp["sample_metadata"]
        for pair_index in sorted(exp["pair_manifest"]["pair_index"].unique()):
            info = _manifest_row(exp["pair_manifest"], pair_index)
            data = load_pair_npz(exp["export_dir"], pair_index)
            if info is None:
                continue
            sub = fm[fm["pair_index"] == pair_index]
            if sub.empty:
                continue
            danger_idx = _danger_indices(info["class_names"], info["dangerous_classes"])
            for metric_name, eco_col, lite_col, heavy_col in fold_metric_map:
                if eco_col not in sub.columns:
                    continue
                for base_col, comparison in [(lite_col, "EcoFair vs Lite"), (heavy_col, "EcoFair vs Heavy")]:
                    if base_col not in sub.columns:
                        continue
                    fold_rows.append({
                        "dataset": exp["dataset"], "pair_index": pair_index, "pair_label": info["pair_label"],
                        "comparison": comparison, "metric": metric_name, "n_folds": len(sub),
                        **_paired_tests(sub[eco_col].values, sub[base_col].values),
                    })
            if data is not None:
                y_true = data["y_true"].astype(int)
                pred_eco = data["oof_ecofair"].argmax(axis=1)
                for metric_name, fn in [
                    ("balanced_accuracy", lambda yt, yp: balanced_accuracy_score(yt, yp)),
                    ("malignant_recall", lambda yt, yp: malignant_recall(yt, yp, danger_idx)),
                ]:
                    mean, lo, hi = _bootstrap_ci(y_true, pred_eco, fn)
                    boot_rows.append({
                        "dataset": exp["dataset"], "pair_index": pair_index, "pair_label": info["pair_label"],
                        "comparison": "EcoFair OOF", "metric": metric_name,
                        "bootstrap_mean": mean, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                    })
                y_true = data["y_true"].astype(int)
                for comparison, base_probs in [("EcoFair vs Lite", data["oof_lite"]), ("EcoFair vs Heavy", data["oof_heavy"])]:
                    fair_eco = fairness_tpr_summary(y_true, data["oof_ecofair"], meta_df, info["class_names"], info["dangerous_classes"])
                    fair_base = fairness_tpr_summary(y_true, base_probs, meta_df, info["class_names"], info["dangerous_classes"])
                    for mname in ["worst_group_tpr", "tpr_gap"]:
                        fold_rows.append({
                            "dataset": exp["dataset"], "pair_index": pair_index, "pair_label": info["pair_label"],
                            "comparison": comparison, "metric": mname, "n_folds": 1,
                            "mean_diff": fair_eco[mname] - fair_base[mname] if all(np.isfinite([fair_eco[mname], fair_base[mname]])) else np.nan,
                            "paired_t_pvalue": np.nan, "wilcoxon_pvalue": np.nan, "cohens_dz": np.nan,
                        })
    fold_df = pd.DataFrame(fold_rows)
    boot_df = pd.DataFrame(boot_rows)
    return fold_df if boot_df.empty else pd.concat([fold_df, boot_df], ignore_index=True, sort=False)


def run_energy_latency_transmission(exports):
    rows, latency_missing = [], False
    for exp in exports:
        if exp is None:
            continue
        for pair_index in sorted(exp["pair_manifest"]["pair_index"].unique()):
            info = _manifest_row(exp["pair_manifest"], pair_index)
            if info is None:
                continue
            sub = exp["fold_metrics"][exp["fold_metrics"]["pair_index"] == pair_index]
            route_rate = float(sub["routing_rate"].mean()) if not sub.empty else np.nan
            j_lite, j_heavy = _pair_energy_joules(exp["energy_stats"], pair_index)
            lat_lite, lat_heavy = _pair_latency_ms(exp["energy_stats"], pair_index)
            if np.isnan(lat_lite) and np.isnan(lat_heavy):
                latency_missing = True
            eco_energy = energy_per_sample(route_rate, j_lite, j_heavy)
            eco_lat = lat_lite + route_rate * lat_heavy if np.isfinite(lat_lite) and np.isfinite(lat_heavy) else np.nan
            worst_lat = lat_lite + lat_heavy if np.isfinite(lat_lite) and np.isfinite(lat_heavy) else np.nan
            kb_lite, kb_heavy = _embedding_kb(exp["energy_stats"], pair_index, "Lite"), _embedding_kb(exp["energy_stats"], pair_index, "Heavy")
            tx_sel = tx_con = np.nan
            if np.isfinite(kb_lite) and np.isfinite(kb_heavy) and np.isfinite(route_rate):
                tx_sel = (1 - route_rate) * kb_lite + route_rate * kb_heavy
                tx_con = kb_lite + route_rate * kb_heavy
            rows.append({
                "dataset": exp["dataset"], "pair_index": pair_index, "pair_label": info["pair_label"],
                "routing_rate": route_rate, "energy_lite_per_sample": j_lite, "energy_heavy_per_sample": j_heavy,
                "energy_ecofair_expected_per_sample": eco_energy,
                "energy_saving_vs_heavy_pct": (1 - eco_energy / j_heavy) * 100 if np.isfinite(eco_energy) and np.isfinite(j_heavy) else np.nan,
                "energy_overhead_vs_lite_pct": (eco_energy / j_lite - 1) * 100 if np.isfinite(eco_energy) and np.isfinite(j_lite) else np.nan,
                "latency_lite_ms": lat_lite, "latency_heavy_ms": lat_heavy,
                "latency_ecofair_expected_ms": eco_lat, "latency_ecofair_worst_case_ms": worst_lat,
                "embedding_kb_lite": kb_lite, "embedding_kb_heavy": kb_heavy,
                "transmission_kb_selected_only": tx_sel, "transmission_kb_conservative": tx_con,
            })
    if latency_missing:
        print("\nNote: latency statistics were unavailable in the saved energy_stats artefacts for some pairs.")
    return pd.DataFrame(rows)


def run_final_summary(ablation_df, stats_df, uncertainty_df, energy_df):
    rows = []
    if ablation_df is None or ablation_df.empty:
        return pd.DataFrame(rows)

    def _sig(dataset, pair_index, comparison, metric):
        if stats_df is None or stats_df.empty:
            return np.nan
        m = stats_df[(stats_df["dataset"] == dataset) & (stats_df["pair_index"] == pair_index)
                     & (stats_df["comparison"] == comparison) & (stats_df["metric"] == metric)]
        if m.empty or not pd.notna(m["paired_t_pvalue"].iloc[0]):
            return np.nan
        return "yes" if m["paired_t_pvalue"].iloc[0] < 0.05 else "no"

    for (dataset, pair_index), ab_sub in ablation_df.groupby(["dataset", "pair_index"]):
        full = ab_sub[ab_sub["routing_variant"] == "Full EcoFair routing"]
        if full.empty:
            continue
        full = full.iloc[0]
        ab_only = ab_sub[~ab_sub["routing_variant"].isin(["Lite only", "Heavy only", "Full EcoFair routing"])]
        best_ab = ab_only.loc[ab_only["balanced_accuracy"].idxmax()] if not ab_only.empty else None
        unc = uncertainty_df[(uncertainty_df["dataset"] == dataset) & (uncertainty_df["pair_index"] == pair_index) & (uncertainty_df["model"] == "Lite")]
        en = energy_df[(energy_df["dataset"] == dataset) & (energy_df["pair_index"] == pair_index)]
        rows.append({
            "dataset": dataset, "pair_index": pair_index, "pair_label": ab_sub["pair_label"].iloc[0],
            "ecofair_routing_rate": full["routing_rate"], "ecofair_balanced_accuracy": full["balanced_accuracy"],
            "ecofair_malignant_recall": full["malignant_recall"],
            "ecofair_energy_saving_vs_heavy_pct": float(en["energy_saving_vs_heavy_pct"].iloc[0]) if not en.empty else np.nan,
            "best_ablation_variant": best_ab["routing_variant"] if best_ab is not None else "",
            "best_ablation_balanced_accuracy": best_ab["balanced_accuracy"] if best_ab is not None else np.nan,
            "lite_entropy_auroc_for_errors": float(unc["entropy_auroc_lite_error"].iloc[0]) if not unc.empty else np.nan,
            "sig_ecofair_vs_lite_balanced_acc": _sig(dataset, pair_index, "EcoFair vs Lite", "balanced_accuracy"),
            "sig_ecofair_vs_heavy_balanced_acc": _sig(dataset, pair_index, "EcoFair vs Heavy", "balanced_accuracy"),
            "sig_ecofair_vs_lite_malignant_recall": _sig(dataset, pair_index, "EcoFair vs Lite", "malignant_recall"),
            "sig_ecofair_vs_heavy_malignant_recall": _sig(dataset, pair_index, "EcoFair vs Heavy", "malignant_recall"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_additional_analyses(export_dirs_by_dataset, output_dir):
    """
    Run all reviewer-facing post-hoc analyses.

    Args:
        export_dirs_by_dataset: {dataset_name: path to reviewer_analysis_export}
        output_dir: directory for CSV/LaTeX/plot outputs

    Returns:
        dict with dataframes, saved file paths, and resolved export dirs
    """
    os.makedirs(output_dir, exist_ok=True)
    exports = load_all_exports(export_dirs_by_dataset)
    loaded = [e["dataset"] for e in exports if e is not None]
    saved = []

    if not loaded:
        return {"loaded": [], "export_dirs": export_dirs_by_dataset, "saved_files": [], "output_dir": output_dir}

    print("\n1. Routing signal ablation...")
    ablation_df = run_routing_ablation(exports)
    saved.extend(save_results_table(ablation_df, output_dir, "routing_ablation_summary.csv", "routing_ablation_summary_latex.txt", "Routing signal ablation summary"))

    print("2. Threshold sensitivity...")
    thresh_df = run_threshold_sensitivity(exports)
    saved.extend(save_results_table(thresh_df, output_dir, "threshold_sensitivity_summary.csv", "threshold_sensitivity_latex.txt", "Threshold sensitivity summary"))
    saved.extend(plot_threshold_sensitivity(thresh_df, output_dir))

    print("3. Uncertainty and calibration quality...")
    uncertainty_df = run_uncertainty_quality(exports)
    saved.extend(save_results_table(uncertainty_df, output_dir, "uncertainty_quality_summary.csv", "uncertainty_quality_latex.txt", "Uncertainty and calibration quality"))

    print("4. Statistical testing...")
    stats_df = run_statistical_tests(exports)
    saved.extend(save_results_table(stats_df, output_dir, "statistical_tests_summary.csv", "statistical_tests_latex.txt", "Statistical tests summary"))

    print("5. Energy, latency, and transmission...")
    energy_df = run_energy_latency_transmission(exports)
    saved.extend(save_results_table(energy_df, output_dir, "energy_latency_transmission_summary.csv", "energy_latency_transmission_latex.txt", "Energy, latency, transmission"))

    print("6. Final reviewer summary...")
    final_df = run_final_summary(ablation_df, stats_df, uncertainty_df, energy_df)
    saved.extend(save_results_table(final_df, output_dir, "reviewer_final_summary.csv", "reviewer_final_summary_latex.txt", "Reviewer final summary"))

    return {
        "loaded": loaded,
        "export_dirs": export_dirs_by_dataset,
        "saved_files": saved,
        "output_dir": output_dir,
        "ablation_df": ablation_df,
        "threshold_df": thresh_df,
        "uncertainty_df": uncertainty_df,
        "stats_df": stats_df,
        "energy_df": energy_df,
        "final_df": final_df,
    }


def print_run_summary(result):
    """Print paths and saved artefacts after run_additional_analyses."""
    loaded = result.get("loaded", [])
    print(f"\nLoaded exports for: {loaded if loaded else 'none'}")
    if result.get("export_dirs"):
        print("\nResolved export folders:")
        for ds in DATASET_NAMES:
            path = result["export_dirs"].get(ds)
            status = path if path and os.path.isdir(path) else "NOT FOUND"
            print(f"  {ds}: {status}")
    if not loaded:
        print("\n" + format_expected_paths(result.get("export_dirs")))
        return
    print(f"\nAll outputs saved under: {os.path.abspath(result['output_dir'])}")
    print("Saved files:")
    for path in result.get("saved_files", []):
        print(f"  - {os.path.basename(path)}")
