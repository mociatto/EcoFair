"""
Utility functions for EcoFair project.
"""

import json
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from . import config


def set_seed(seed: int = None) -> None:
    """Set random seeds for reproducibility."""
    if seed is None:
        seed = config.RANDOM_STATE
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_energy_stats(energy_dir: str):
    """
    Load energy statistics from a directory containing energy_stats.json.

    Args:
        energy_dir: Path to directory containing energy_stats.json

    Returns:
        float: Joules per sample, or None if not found
    """
    stats = load_energy_stats_full(energy_dir)
    if stats is None:
        return None
    return stats.get('joules_per_sample')


def load_energy_stats_full(energy_dir: str):
    """
    Load the full energy_stats.json payload from a model output directory.

    Args:
        energy_dir: Path to directory containing energy_stats.json

    Returns:
        dict: Parsed energy statistics, or None if not found / unreadable
    """
    energy_path = os.path.join(energy_dir, 'energy_stats.json')
    if not os.path.exists(energy_path):
        return None
    try:
        with open(energy_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _first_column(meta_df, candidates):
    """Return the first present column name from a list of candidates."""
    col_map = {c.lower(): c for c in meta_df.columns}
    for name in candidates:
        if name in meta_df.columns:
            return name
        if name.lower() in col_map:
            return col_map[name.lower()]
    return None


def _build_sample_metadata_df(meta_df, y_true, class_names):
    """Build de-identified per-sample metadata for reviewer subgroup analysis."""
    n_samples = len(meta_df)
    rows = {
        'sample_index': np.arange(n_samples),
        'true_label': y_true,
    }

    dx_col = _first_column(meta_df, ['dx', 'diagnosis', 'diagnostic', 'label'])
    if dx_col is not None:
        rows['diagnosis'] = meta_df[dx_col].astype(str).values
        rows['class_name'] = meta_df[dx_col].astype(str).str.lower().str.strip().values
    else:
        rows['diagnosis'] = np.full(n_samples, '', dtype=object)
        rows['class_name'] = np.array([class_names[i] if i < len(class_names) else '' for i in y_true])

    age_col = _first_column(meta_df, ['age', 'age_approx'])
    if age_col is not None:
        rows['age'] = pd.to_numeric(meta_df[age_col], errors='coerce').values
        age_vals = pd.Series(rows['age']).fillna(0).clip(lower=0)
        rows['age_bin'] = pd.cut(
            age_vals,
            bins=[0, 30, 60, 200],
            labels=['<30', '30-60', '60+'],
            include_lowest=True,
        ).astype(str).values
    else:
        rows['age'] = np.full(n_samples, np.nan)
        rows['age_bin'] = np.full(n_samples, '', dtype=object)

    sex_col = _first_column(meta_df, ['sex', 'gender'])
    if sex_col is not None:
        rows['sex'] = meta_df[sex_col].astype(str).values
    else:
        rows['sex'] = np.full(n_samples, '', dtype=object)

    loc_col = _first_column(meta_df, ['localization', 'region', 'lesion_location', 'anatom_site', 'anatom_site_general'])
    if loc_col is not None:
        rows['localization'] = meta_df[loc_col].astype(str).values
    else:
        rows['localization'] = np.full(n_samples, '', dtype=object)

    if 'risk_score' in meta_df.columns:
        rows['patient_risk'] = pd.to_numeric(meta_df['risk_score'], errors='coerce').values
    else:
        rows['patient_risk'] = np.full(n_samples, np.nan)

    return pd.DataFrame(rows)


def export_reviewer_analysis(
    cv_results,
    meta_df,
    y,
    dataset_name,
    output_dir,
    pair_labels,
    pair_dirs,
    class_names,
    safe_classes,
    dangerous_classes,
):
    """
    Export compressed OOF arrays and tabular manifests for post-hoc reviewer analyses.

    Creates a separate folder under output_dir without modifying existing pipeline CSVs.

    Returns:
        dict: Summary information about exported artefacts
    """
    export_dir = os.path.join(output_dir, 'reviewer_analysis_export')
    os.makedirs(export_dir, exist_ok=True)

    y_true = np.argmax(y, axis=1)
    sample_meta_df = _build_sample_metadata_df(meta_df, y_true, class_names)
    if cv_results:
        _, _, _, _, _, route_components_first = cv_results[0]
        sample_meta_df['fold_id'] = route_components_first['fold_id']
    else:
        sample_meta_df['fold_id'] = -1
    sample_meta_path = os.path.join(export_dir, 'sample_metadata.csv')
    sample_meta_df.to_csv(sample_meta_path, index=False)

    fold_rows = []
    manifest_rows = []
    energy_rows = []
    saved_files = [sample_meta_path]
    pair_summaries = []

    for pair_idx, result in enumerate(cv_results):
        fold_metrics, oof_lite, oof_heavy, oof_ecofair, route_mask, route_components = result
        lite_dir, heavy_dir = pair_dirs[pair_idx]
        lite_model = os.path.basename(lite_dir)
        heavy_model = os.path.basename(heavy_dir)
        pair_label = pair_labels[pair_idx]
        n_samples = len(y_true)

        npz_name = f'pair_{pair_idx:02d}_oof.npz'
        npz_path = os.path.join(export_dir, npz_name)
        np.savez_compressed(
            npz_path,
            y_true=y_true.astype(np.int16),
            oof_lite=oof_lite.astype(np.float32),
            oof_heavy=oof_heavy.astype(np.float32),
            oof_ecofair=oof_ecofair.astype(np.float32),
            route_mask=route_mask.astype(bool),
            route_uncertainty=route_components['uncertainty'].astype(bool),
            route_ambiguity=route_components['ambiguity'].astype(bool),
            route_risk=route_components['safety'].astype(bool),
            entropy=route_components['entropy'].astype(np.float32),
            safe_danger_gap=route_components['safe_danger_gap'].astype(np.float32),
            prob_safe=route_components['prob_safe'].astype(np.float32),
            prob_danger=route_components['prob_danger'].astype(np.float32),
            patient_risk=route_components['patient_risk'].astype(np.float32),
            fold_id=route_components['fold_id'].astype(np.int16),
            entropy_threshold=route_components['entropy_threshold'].astype(np.float32),
            gap_threshold=route_components['gap_threshold'].astype(np.float32),
            risk_threshold=route_components['risk_threshold'].astype(np.float32),
            heavy_weight=route_components['heavy_weight'].astype(np.float32),
        )
        saved_files.append(npz_path)

        route_union = (
            route_components['uncertainty']
            | route_components['ambiguity']
            | route_components['safety']
        )
        route_mask_ok = bool(np.all(route_mask == route_union))
        thresholds_captured = not np.all(np.isnan(route_components['entropy_threshold']))

        pair_summaries.append({
            'pair_index': pair_idx,
            'pair_label': pair_label,
            'n_samples': n_samples,
            'oof_shape': oof_lite.shape,
            'route_rate': float(route_mask.mean()),
            'route_mask_consistent': route_mask_ok,
            'thresholds_captured': thresholds_captured,
            'npz_file': npz_name,
        })

        manifest_rows.append({
            'dataset': dataset_name,
            'pair_index': pair_idx,
            'pair_label': pair_label,
            'lite_model': lite_model,
            'heavy_model': heavy_model,
            'npz_filename': npz_name,
            'n_samples': n_samples,
            'n_classes': len(class_names),
            'class_names': ';'.join(class_names),
            'safe_classes': ';'.join(safe_classes),
            'dangerous_classes': ';'.join(dangerous_classes),
        })

        n_folds = len(fold_metrics['acc_lite'])
        for fold_i in range(n_folds):
            fold_rows.append({
                'dataset': dataset_name,
                'pair_index': pair_idx,
                'pair_label': pair_label,
                'lite_model': lite_model,
                'heavy_model': heavy_model,
                'fold_number': fold_i + 1,
                'entropy_threshold': fold_metrics['entropy_threshold'][fold_i],
                'gap_threshold': fold_metrics['gap_threshold'][fold_i],
                'risk_threshold': fold_metrics['risk_threshold'][fold_i],
                'heavy_weight': fold_metrics['heavy_weight'][fold_i],
                'routing_rate': fold_metrics['routing_rate'][fold_i],
                'routing_rate_uncertainty': fold_metrics['routing_rate_uncertainty'][fold_i],
                'routing_rate_ambiguity': fold_metrics['routing_rate_ambiguity'][fold_i],
                'routing_rate_risk': fold_metrics['routing_rate_risk'][fold_i],
                'acc_lite': fold_metrics['acc_lite'][fold_i],
                'acc_heavy': fold_metrics['acc_heavy'][fold_i],
                'acc_ecofair': fold_metrics['acc_dynamic'][fold_i],
                'macro_f1_lite': fold_metrics['macro_f1_lite'][fold_i],
                'macro_f1_heavy': fold_metrics['macro_f1_heavy'][fold_i],
                'macro_f1_ecofair': fold_metrics['macro_f1_dynamic'][fold_i],
                'balanced_acc_lite': fold_metrics['balanced_acc_lite'][fold_i],
                'balanced_acc_heavy': fold_metrics['balanced_acc_heavy'][fold_i],
                'balanced_acc_ecofair': fold_metrics['balanced_acc_dynamic'][fold_i],
                'malignant_recall_lite': fold_metrics['malignant_recall_lite'][fold_i],
                'malignant_recall_heavy': fold_metrics['malignant_recall_heavy'][fold_i],
                'malignant_recall_ecofair': fold_metrics['malignant_recall_dynamic'][fold_i],
                'energy_ecofair_per_sample': fold_metrics['energy_cost'][fold_i],
                'energy_lite_per_sample': fold_metrics['energy_lite'][fold_i],
                'energy_heavy_per_sample': fold_metrics['energy_heavy'][fold_i],
            })

        for role, model_dir, model_name in [
            ('Lite', lite_dir, lite_model),
            ('Heavy', heavy_dir, heavy_model),
        ]:
            stats = load_energy_stats_full(model_dir) or {}
            row = {
                'dataset': dataset_name,
                'pair_index': pair_idx,
                'pair_label': pair_label,
                'model_role': role,
                'model_name': model_name,
            }
            for key in [
                'model', 'dataset', 'resolution', 'n_samples', 'embedding_dim',
                'model_parameters', 'model_size_mb', 'latency_per_sample_mean_ms',
                'latency_per_sample_std_ms', 'gpu_memory_mb', 'joules_per_sample', 'total_energy_j',
            ]:
                row[key] = stats.get(key, np.nan)
            energy_rows.append(row)

    fold_path = os.path.join(export_dir, 'fold_metrics.csv')
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    saved_files.append(fold_path)

    manifest_path = os.path.join(export_dir, 'pair_manifest.csv')
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    saved_files.append(manifest_path)

    energy_path = os.path.join(export_dir, 'energy_stats.csv')
    pd.DataFrame(energy_rows).to_csv(energy_path, index=False)
    saved_files.append(energy_path)

    return {
        'export_dir': export_dir,
        'n_pairs': len(cv_results),
        'pair_summaries': pair_summaries,
        'saved_files': saved_files,
    }


def save_results_csv(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """
    Save a DataFrame to CSV in the output directory.

    Args:
        df: DataFrame to save
        output_dir: Directory path (created if missing)
        filename: CSV filename (e.g. 'cv_results.csv')

    Returns:
        str: Full path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path


def show_table(df: pd.DataFrame):
    """
    Display a DataFrame as a table (Jupyter display or print fallback).
    """
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string())
