"""
Visualization module for EcoFair project.

This module generates professional, journal-quality figures for the EcoFair pipeline.
All functions accept data (DataFrames/Arrays) and return matplotlib figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import LinearSegmentedColormap
from . import config
from . import features
from . import routing
from . import fairness

# Use default style (seaborn styles set legend.frameon=False which overrides our fixes)
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'


def plot_metadata_distributions(meta_df, dangerous_classes, title_suffix=''):
    """
    Plot metadata distributions with malignancy rate overlays.
    
    Replicates logic from Cell 02 of EcoFair_Main.py.
    Creates two subplots: Age Distribution and Localization Distribution
    with Malignancy Rates overlay.
    
    Args:
        meta_df: DataFrame with metadata columns (age, localization/region, dx/diagnostic/diagnosis)
        dangerous_classes: List of class names considered malignant (required).
        title_suffix: Optional string appended to figure title.
    
    Returns:
        matplotlib.figure.Figure: Figure object with two subplots
    """
    if dangerous_classes is None:
        raise ValueError("dangerous_classes is required")
    meta_df_copy = meta_df.copy()
    
    # Detect diagnosis column dynamically
    dx_col = None
    for col in ['dx', 'diagnostic', 'diagnosis']:
        if col in meta_df_copy.columns:
            dx_col = col
            break
    
    if dx_col is not None:
        meta_df_copy['is_malignant'] = (
            meta_df_copy[dx_col].astype(str).str.lower().str.strip()
            .isin([c.lower() for c in dangerous_classes])
        ).astype(int)
    else:
        meta_df_copy['is_malignant'] = 0
    
    # Detect localization column dynamically
    loc_col = None
    for col in ['localization', 'region', 'lesion_location', 'anatom_site', 'anatom_site_general']:
        if col in meta_df_copy.columns:
            loc_col = col
            break
    if loc_col is None:
        meta_df_copy['_loc'] = 'unknown'
        loc_col = '_loc'
    
    # Age bins — handle missing or non-numeric age gracefully
    age_col = None
    for col in ['age', 'age_approx']:
        if col in meta_df_copy.columns:
            age_col = col
            break
    if age_col is not None:
        meta_df_copy[age_col] = pd.to_numeric(meta_df_copy[age_col], errors='coerce')
        age_bins = np.arange(0, 101, 10)
        meta_df_copy['age_bin'] = pd.cut(
            meta_df_copy[age_col],
            bins=age_bins,
            labels=[f"{int(age_bins[i])}-{int(age_bins[i+1])}" for i in range(len(age_bins)-1)],
            include_lowest=True
        )
        age_counts = meta_df_copy.groupby('age_bin', observed=True).size()
        age_malignancy_rate = meta_df_copy.groupby('age_bin', observed=True)['is_malignant'].mean() * 100
    else:
        age_counts = pd.Series({'N/A': len(meta_df_copy)})
        age_malignancy_rate = pd.Series({'N/A': meta_df_copy['is_malignant'].mean() * 100})
    
    # Localization
    loc_malignancy_rate = meta_df_copy.groupby(loc_col)['is_malignant'].mean() * 100
    loc_malignancy_rate = loc_malignancy_rate.sort_values(ascending=True)
    loc_counts = meta_df_copy.groupby(loc_col).size().reindex(loc_malignancy_rate.index)
    
    # Dynamic y-axis ceiling (round up to nearest 500 above max count)
    max_age_count = int(age_counts.max()) if len(age_counts) > 0 else 100
    max_loc_count = int(loc_counts.max()) if len(loc_counts) > 0 else 100
    age_ylim = max(500, ((max_age_count // 500) + 1) * 500)
    loc_ylim = max(500, ((max_loc_count // 500) + 1) * 500)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Metadata Distributions{title_suffix}', fontsize=15, fontweight='normal', y=1.01)
    
    # Age distribution
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    ax1.bar(range(len(age_counts)), age_counts.values, alpha=0.7, color='skyblue', label='Sample Count')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='normal')
    ax1.set_ylim([0, age_ylim])
    ax1.set_xticks(range(len(age_counts)))
    ax1.set_xticklabels(age_counts.index, rotation=90, ha='right')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.xaxis.grid(False)
    
    ax1_twin.plot(range(len(age_malignancy_rate)), age_malignancy_rate.values,
                  color='orangered', marker='o', linewidth=2, markersize=8, label='Malignancy Rate')
    ax1_twin.set_ylim([0, 100])
    ax1_twin.tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False)
    ax1_twin.grid(False)
    
    ax1.set_title('Age Distribution and Malignancy Rate', fontsize=14, fontweight='normal', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1_twin.legend(loc='upper right', fontsize=10)
    
    # Localization distribution
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    ax2.bar(range(len(loc_counts)), loc_counts.values, alpha=0.7, color='lightgreen', label='Sample Count')
    ax2.set_ylim([0, loc_ylim])
    ax2.set_xticks(range(len(loc_counts)))
    ax2.set_xticklabels([str(loc).title() for loc in loc_counts.index], rotation=90, ha='right')
    ax2.tick_params(axis='y', labelleft=False, left=False)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.xaxis.grid(False)
    
    ax2_twin.plot(range(len(loc_malignancy_rate)), loc_malignancy_rate.values,
                  color='orangered', marker='o', linewidth=2, markersize=8, label='Malignancy Rate')
    ax2_twin.set_ylabel('Malignancy Rate (%)', fontsize=12, fontweight='normal')
    ax2_twin.tick_params(axis='y')
    ax2_twin.set_ylim([0, 100])
    ax2_twin.grid(False)
    
    ax2.set_title('Localization Distribution and Malignancy Rate', fontsize=14, fontweight='normal', pad=20)
    ax2.legend(loc='upper left', fontsize=10)
    ax2_twin.legend(loc='upper right', fontsize=10)
    
    # Print simple tables for age and localization bins:
    # e.g. "10-20: (180, 25.00)" where first is sample count, second is malignancy rate (%)
    try:
        print("\n--- Age bins: sample count and malignancy rate (%) ---")
        for bin_label, count in age_counts.items():
            rate = age_malignancy_rate.get(bin_label, np.nan)
            if pd.isna(rate):
                rate_str = "nan"
            else:
                rate_str = f"{rate:.2f}"
            print(f"{bin_label}: ({int(count)}, {rate_str})")
    except Exception:
        # Fail silently if printing table is not possible
        pass
    
    try:
        print("\n--- Localization: sample count and malignancy rate (%) ---")
        for loc_label, count in loc_counts.items():
            rate = loc_malignancy_rate.get(loc_label, np.nan)
            if pd.isna(rate):
                rate_str = "nan"
            else:
                rate_str = f"{rate:.2f}"
            print(f"{str(loc_label).title()}: ({int(count)}, {rate_str})")
    except Exception:
        pass
    
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix_comparison(y_true, lite_preds, heavy_preds, dynamic_preds, class_names):
    """
    Plot side-by-side confusion matrices for Lite, Heavy, and Dynamic systems.
    
    Args:
        y_true: True labels (class indices), shape (n_samples,)
        lite_preds: Lite predictions (class indices or probabilities)
        heavy_preds: Heavy predictions (class indices or probabilities)
        dynamic_preds: Dynamic predictions (class indices or probabilities)
        class_names: List of class names (required).
    
    Returns:
        matplotlib.figure.Figure: Figure object with three subplots
    """
    if class_names is None:
        raise ValueError("class_names is required")
    # Convert to class indices if probabilities
    if len(lite_preds.shape) > 1:
        lite_preds_class = np.argmax(lite_preds, axis=1)
    else:
        lite_preds_class = lite_preds
    
    if len(heavy_preds.shape) > 1:
        heavy_preds_class = np.argmax(heavy_preds, axis=1)
    else:
        heavy_preds_class = heavy_preds
    
    if len(dynamic_preds.shape) > 1:
        dynamic_preds_class = np.argmax(dynamic_preds, axis=1)
    else:
        dynamic_preds_class = dynamic_preds
    
    # Compute confusion matrices
    cm_lite = confusion_matrix(y_true, lite_preds_class)
    cm_heavy = confusion_matrix(y_true, heavy_preds_class)
    cm_dynamic = confusion_matrix(y_true, dynamic_preds_class)
    
    # Define colors
    colors = {
        'lite': 'skyblue',
        'heavy': 'orangered',
        'dynamic': 'lightgreen'
    }
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cms = [cm_lite, cm_heavy, cm_dynamic]
    titles = ['Pure Lite System', 'Pure Heavy System', 'Dynamic Routing System']
    cmap_names = ['lite', 'heavy', 'dynamic']
    
    for idx, (ax, cm, title, cmap_name) in enumerate(zip(axes, cms, titles, cmap_names)):
        color = colors[cmap_name]
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        colors_list = ['white', color]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.grid(False)
        ax.set_title(title, fontsize=14, fontweight='normal', pad=15)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=0)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f'{cm[i, j]}',
                       ha="center", va="center",
                       color="black",
                       fontsize=10, fontweight='normal')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Count', rotation=270, labelpad=15, fontsize=10)
    
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix_comparison_multi(pairs_results, class_names, pair_labels=None):
    """
    Plot 3x3 grid of confusion matrices: 3 pairs × (Lite, Heavy, EcoFair).

    Args:
        pairs_results: List of (y_true, oof_lite, oof_heavy, oof_dynamic) per pair
        class_names: List of class names
        pair_labels: Optional list of pair names (e.g. ['Pair 1', 'Pair 2', 'Pair 3'])

    Returns:
        matplotlib.figure.Figure
    """
    if pair_labels is None:
        pair_labels = [f'Pair {i+1}' for i in range(len(pairs_results))]
    n_pairs = len(pairs_results)
    fig, axes = plt.subplots(n_pairs, 3, figsize=(18, 6 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    colors = {'lite': 'skyblue', 'heavy': 'orangered', 'dynamic': 'lightgreen'}
    titles_row = ['Pure Lite', 'Pure Heavy', 'EcoFair']
    for row, ((y_true, oof_lite, oof_heavy, oof_dynamic), plabel) in enumerate(zip(pairs_results, pair_labels)):
        preds_list = [oof_lite, oof_heavy, oof_dynamic]
        for col, (preds, t) in enumerate(zip(preds_list, titles_row)):
            ax = axes[row, col]
            pred_class = np.argmax(preds, axis=1) if len(preds.shape) > 1 else preds
            cm = confusion_matrix(y_true, pred_class)
            cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
            colors_list = ['white', colors['lite' if col == 0 else 'heavy' if col == 1 else 'dynamic']]
            cmap = LinearSegmentedColormap.from_list('c', colors_list, N=100)
            im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'{plabel}: {t}', fontsize=12)
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', fontsize=9)
    plt.tight_layout()
    return fig


def plot_fairness_summary_grouped(pairs_fairness_reports, dangerous_classes, pair_labels=None):
    """
    Plot 1x3 grouped bar charts: one per pair, each showing 3 metrics (macro_tpr_mean,
    macro_tpr_worst_group, macro_tpr_gap) for Lite, EcoFair, Heavy.

    Args:
        pairs_fairness_reports: List of {'Lite': df, 'EcoFair': df, 'Heavy': df} per pair
        dangerous_classes: List of dangerous class names
        pair_labels: Optional list of pair names

    Returns:
        matplotlib.figure.Figure
    """
    def _macro_tpr_metrics(df):
        if df is None or df.empty:
            return np.nan, np.nan, np.nan
        subset = df[df['Class'].isin(dangerous_classes)].copy()
        subset['_tpr'] = pd.to_numeric(subset['Equal_Opportunity_TPR'], errors='coerce')
        sg_macro = subset.groupby('Subgroup')['_tpr'].mean().dropna()
        if sg_macro.empty:
            return np.nan, np.nan, np.nan
        return float(sg_macro.mean()), float(sg_macro.min()), float(sg_macro.max() - sg_macro.min())

    n_pairs = len(pairs_fairness_reports)
    if pair_labels is None:
        pair_labels = [f'Pair {i+1}' for i in range(n_pairs)]
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    metrics = ['Macro TPR (mean)', 'Worst-group TPR', 'TPR Gap']
    x = np.arange(3)
    width = 0.25
    colors_m = ['skyblue', 'lightgreen', 'orangered']
    for ax, reports, plabel in zip(axes, pairs_fairness_reports, pair_labels):
        lite_vals = _macro_tpr_metrics(reports.get('Lite'))
        eco_vals = _macro_tpr_metrics(reports.get('EcoFair'))
        heavy_vals = _macro_tpr_metrics(reports.get('Heavy'))
        vals = [[lite_vals[i] if not np.isnan(lite_vals[i]) else 0 for i in range(3)],
                [eco_vals[i] if not np.isnan(eco_vals[i]) else 0 for i in range(3)],
                [heavy_vals[i] if not np.isnan(heavy_vals[i]) else 0 for i in range(3)]]
        for i, (v, c, lbl) in enumerate(zip(vals, colors_m, ['Lite', 'EcoFair', 'Heavy'])):
            ax.bar(x + (i - 1) * width, v, width, label=lbl, color=c)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Value')
        ax.set_title(plabel)
        ax.legend()
        ax.set_ylim(0, 1.0)
    plt.tight_layout()
    return fig


def plot_clinical_safety_rescue_multi(pairs_fairness, dangerous_classes, pair_labels=None):
    """
    Plot 2×3 grid: 2 demographics (Age, Sex) × 3 pairs.

    Args:
        pairs_fairness: List of (fairness_lite, fairness_heavy, fairness_ecofair) per pair
        dangerous_classes: List of dangerous class names
        pair_labels: Optional list of pair names

    Returns:
        matplotlib.figure.Figure
    """
    if pair_labels is None:
        pair_labels = [f'Pair {i+1}' for i in range(len(pairs_fairness))]
    fig, axes = plt.subplots(2, len(pairs_fairness), figsize=(6 * len(pairs_fairness), 10))
    if len(pairs_fairness) == 1:
        axes = axes.reshape(-1, 1)
    color_lite, color_ecofair, color_heavy = 'skyblue', 'lightgreen', 'orangered'
    width = 0.25

    def _macro_tpr(df, prefix):
        if df is None or df.empty:
            return {}
        subset = df[df['Subgroup'].str.startswith(prefix, na=False) & df['Class'].isin(dangerous_classes)].copy()
        subset['_tpr'] = pd.to_numeric(subset['Equal_Opportunity_TPR'], errors='coerce')
        result = {}
        for sg, grp in subset.groupby('Subgroup'):
            clean = str(sg).replace(' (n=', ' (n=')[:20]
            result[sg] = float(grp['_tpr'].mean())
        return result

    for col, (fl, fh, fe), plabel in zip(range(len(pairs_fairness)), pairs_fairness, pair_labels):
        for row, (prefix, dem_name) in enumerate([('Age ', 'Age'), ('Sex: ', 'Sex')]):
            ax = axes[row, col]
            al = _macro_tpr(fl, prefix)
            ae = _macro_tpr(fe, prefix)
            ah = _macro_tpr(fh, prefix)
            subgroups = sorted(set(al.keys()) | set(ae.keys()) | set(ah.keys()))
            if not subgroups:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{plabel}: {dem_name}')
                continue
            x = np.arange(len(subgroups))
            vl = [al.get(s, 0) for s in subgroups]
            ve = [ae.get(s, 0) for s in subgroups]
            vh = [ah.get(s, 0) for s in subgroups]
            ax.bar(x - width, vl, width, color=color_lite, label='Lite')
            ax.bar(x, ve, width, color=color_ecofair, label='EcoFair')
            ax.bar(x + width, vh, width, color=color_heavy, label='Heavy')
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace(prefix, '')[:12] for s in subgroups], rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.set_title(f'{plabel}: {dem_name}')
            ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    return fig


def plot_pareto_frontier_multi(pairs_oof, pairs_joules, y_true, meta_df, class_names, safe_classes,
                               dangerous_classes, pair_labels=None, title_suffix='', entropy_thresholds=None):
    """
    Plot 1×3 Pareto frontier subplots, one per pair. Each shows full Pareto curve (optimal points)
    and non-optimal/failed samples in grey.

    Args:
        pairs_oof: List of (oof_lite, oof_heavy) per pair
        pairs_joules: List of (joules_lite, joules_heavy) per pair
        y_true: True labels (shared)
        meta_df: Metadata
        class_names, safe_classes, dangerous_classes: Class config
        pair_labels: Optional list of pair names
        title_suffix: Optional title suffix
        entropy_thresholds: Array of entropy thresholds to sweep

    Returns:
        matplotlib.figure.Figure
    """
    if entropy_thresholds is None:
        entropy_thresholds = np.linspace(0.1, 1.0, 15)
    n_pairs = len(pairs_oof)
    if pair_labels is None:
        pair_labels = [f'Pair {i+1}' for i in range(n_pairs)]
    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]
    n_classes = len(class_names)
    for ax, (oof_lite, oof_heavy), (j_l, j_h), plabel in zip(axes, pairs_oof, pairs_joules, pair_labels):
        j_l = j_l if j_l else 1.0
        j_h = j_h if j_h else 2.5
        entropy_norm = routing.calculate_entropy(oof_lite) / np.log(n_classes)
        energies_all = []
        worst_tprs_all = []
        for ent_t in entropy_thresholds:
            route_mask = entropy_norm > ent_t
            final_preds = oof_lite.copy()
            final_preds[route_mask] = 0.3 * oof_lite[route_mask] + 0.7 * oof_heavy[route_mask]
            pred_labels = np.argmax(final_preds, axis=1)
            n_samples = len(y_true)
            total_energy = (n_samples - route_mask.sum()) * j_l + route_mask.sum() * j_h
            fairness_df = fairness.generate_fairness_report(y_true, pred_labels, meta_df, class_names)
            subset = fairness_df[fairness_df['Class'].isin(dangerous_classes)].copy()
            subset['_tpr'] = pd.to_numeric(subset['Equal_Opportunity_TPR'], errors='coerce')
            sg_macro = subset.groupby('Subgroup')['_tpr'].mean().dropna()
            worst_tpr = float(sg_macro.min()) if len(sg_macro) > 0 else 0.0
            energies_all.append(total_energy)
            worst_tprs_all.append(worst_tpr)
        energies_all = np.array(energies_all)
        worst_tprs_all = np.array(worst_tprs_all)
        pareto_mask = np.ones(len(energies_all), dtype=bool)
        for i in range(len(energies_all)):
            for j in range(len(energies_all)):
                if i == j:
                    continue
                if energies_all[j] <= energies_all[i] and worst_tprs_all[j] >= worst_tprs_all[i]:
                    pareto_mask[i] = False
                    break
        non_pareto_idx = ~pareto_mask
        if non_pareto_idx.any():
            ax.scatter(energies_all[non_pareto_idx], worst_tprs_all[non_pareto_idx],
                       c='lightgray', s=40, alpha=0.7, label='Non-optimal')
        if pareto_mask.any():
            sort_idx = np.argsort(energies_all[pareto_mask])
            ep = energies_all[pareto_mask][sort_idx]
            wp = worst_tprs_all[pareto_mask][sort_idx]
            ax.plot(ep, wp, 'o-', color='lightgreen', linewidth=2, markersize=8, label='Pareto frontier')
        ax.set_xlabel('Total Edge Energy (J)')
        ax.set_ylabel('Worst-Group TPR')
        ax.set_title(f'{plabel}{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_clinical_safety_rescue(fairness_lite, fairness_heavy, fairness_ecofair, dangerous_classes):
    """
    Plot macro-averaged malignancy TPR (Equal Opportunity) per demographic subgroup,
    showing Pure Lite, EcoFair, and Pure Heavy as grouped bars.

    For each subgroup the TPR values across all dangerous classes are averaged into a
    single 'Overall Malignancy Detection' score, eliminating per-class label clutter.
    Separate subplots are drawn for Age and Sex demographics.

    Args:
        fairness_lite: DataFrame from generate_fairness_report for lite model
        fairness_heavy: DataFrame from generate_fairness_report for heavy model
        fairness_ecofair: DataFrame from generate_fairness_report for EcoFair model
        dangerous_classes: List of dangerous class names (required).

    Returns:
        matplotlib.figure.Figure
    """
    if dangerous_classes is None:
        raise ValueError("dangerous_classes is required")

    dangerous_classes = list(dangerous_classes)
    color_lite    = 'skyblue'
    color_ecofair = 'lightgreen'
    color_heavy   = 'orangered'
    width         = 0.25

    def _macro_tpr(df, subgroup_prefix):
        if df is None or df.empty:
            return {}
        subset = df[
            df['Subgroup'].str.startswith(subgroup_prefix, na=False) &
            df['Class'].isin(dangerous_classes)
        ].copy()
        if subset.empty:
            return {}
        result = {}
        for subgroup, grp in subset.groupby('Subgroup'):
            tpr_vals = pd.to_numeric(grp['Equal_Opportunity_TPR'], errors='coerce').dropna()
            if len(tpr_vals) == 0:
                continue
            clean = pd.Series([subgroup]).str.replace(
                r'\s*\(n=\d+\)\s*$', '', regex=True).iloc[0].strip()
            result[clean] = float(tpr_vals.mean())
        return result

    def _build_subplot(ax, subgroup_prefix, title):
        avg_lite    = _macro_tpr(fairness_lite,    subgroup_prefix)
        avg_ecofair = _macro_tpr(fairness_ecofair, subgroup_prefix)
        avg_heavy   = _macro_tpr(fairness_heavy,   subgroup_prefix)

        subgroups = sorted(set(list(avg_lite.keys()) + list(avg_ecofair.keys()) + list(avg_heavy.keys())))
        if not subgroups:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='grey')
            ax.set_title(title, fontsize=13, pad=12)
            ax.set_ylim(0.0, 1.0)
            return

        x = np.arange(len(subgroups))

        lite_vals    = [avg_lite.get(sg,    0.0) for sg in subgroups]
        ecofair_vals = [avg_ecofair.get(sg, 0.0) for sg in subgroups]
        heavy_vals   = [avg_heavy.get(sg,   0.0) for sg in subgroups]

        b1 = ax.bar(x - width, lite_vals,    width, color=color_lite,    label='Pure Lite',  edgecolor='white', linewidth=1.2)
        b2 = ax.bar(x,         ecofair_vals, width, color=color_ecofair, label='EcoFair',    edgecolor='white', linewidth=1.2)
        b3 = ax.bar(x + width, heavy_vals,   width, color=color_heavy,   label='Pure Heavy', edgecolor='white', linewidth=1.2)

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., h + 0.02,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(subgroups, fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel('Aggregated Malignancy TPR', fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    fig, (ax_age, ax_sex) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Clinical Safety Rescue: Malignancy Detection Rate (TPR) Across Subgroups',
        fontsize=13, fontweight='normal'
    )

    _build_subplot(ax_age, 'Age ',  'Age Subgroups')
    _build_subplot(ax_sex, 'Sex: ', 'Sex Subgroups')

    plt.tight_layout()
    return fig


def plot_pareto_frontier(oof_lite, oof_heavy, y_true, meta_df, class_names, safe_classes, dangerous_classes,
                         joules_lite, joules_heavy, title_suffix='', entropy_thresholds=None,
                         heavy_weight=0.7, ax=None):
    """
    Plot Pareto Frontier: Energy vs. Safety trade-off by sweeping Softmax Entropy thresholds.

    For each entropy threshold, applies gating logic to OOF predictions (no retraining),
    computes Total Edge Energy (X) and Worst-Group TPR for dangerous classes (Y).
    Annotates each point with exact (x, y) coordinates for transcription.

    Args:
        oof_lite: Out-of-fold lite predictions, shape (n_samples, n_classes)
        oof_heavy: Out-of-fold heavy predictions, shape (n_samples, n_classes)
        y_true: True labels (class indices), shape (n_samples,)
        meta_df: Metadata DataFrame for subgroup definitions
        class_names: List of class names
        safe_classes: List of safe class names
        dangerous_classes: List of dangerous class names
        joules_lite: Energy per sample for lite model (J)
        joules_heavy: Energy per sample for heavy model (J)
        title_suffix: Optional string for figure title
        entropy_thresholds: Array of entropy thresholds to sweep (default: np.linspace(0.1, 1.0, 10))
        heavy_weight: Weight for heavy model in ensemble (default: 0.7)
        ax: Optional matplotlib axes. If None, creates new figure.

    Returns:
        matplotlib.figure.Figure
    """
    if entropy_thresholds is None:
        entropy_thresholds = np.linspace(0.1, 1.0, 10)

    n_classes = len(class_names)
    n_samples = len(y_true)
    entropy_norm = routing.calculate_entropy(oof_lite) / np.log(n_classes)

    energies = []
    worst_group_tprs = []

    for ent_t in entropy_thresholds:
        route_mask = entropy_norm > ent_t
        final_preds = oof_lite.copy()
        final_preds[route_mask] = (1 - heavy_weight) * oof_lite[route_mask] + heavy_weight * oof_heavy[route_mask]
        pred_labels = np.argmax(final_preds, axis=1)

        total_energy = (n_samples - route_mask.sum()) * joules_lite + route_mask.sum() * joules_heavy

        fairness_df = fairness.generate_fairness_report(
            y_true, pred_labels, meta_df, class_names
        )
        subset = fairness_df[fairness_df['Class'].isin(dangerous_classes)].copy()
        subset['_tpr'] = pd.to_numeric(subset['Equal_Opportunity_TPR'], errors='coerce')
        sg_macro = subset.groupby('Subgroup')['_tpr'].mean().dropna()
        worst_tpr = float(sg_macro.min()) if len(sg_macro) > 0 else 0.0

        energies.append(total_energy)
        worst_group_tprs.append(worst_tpr)

    energies = np.array(energies)
    worst_group_tprs = np.array(worst_group_tprs)
    sort_idx = np.argsort(energies)
    energies = energies[sort_idx]
    worst_group_tprs = worst_group_tprs[sort_idx]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    else:
        fig = ax.get_figure()

    color = 'lightgreen'
    ax.plot(energies, worst_group_tprs, color=color, linewidth=2.5, linestyle='-', marker='o',
            markersize=8, markeredgecolor='white', markeredgewidth=1.5)

    for i, (x, y) in enumerate(zip(energies, worst_group_tprs)):
        ax.annotate(f'({x:.1f}, {y:.4f})', (x, y), textcoords='offset points',
                    xytext=(8, 8), fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Dynamic axis limits with 5% padding
    x_min, x_max = energies.min(), energies.max()
    x_range = x_max - x_min
    if x_range < 1e-9:
        x_range = max(energies.max(), 1.0)
    x_margin = 0.05 * x_range
    ax.set_xlim(x_min - x_margin, x_max + x_margin)

    y_min, y_max = worst_group_tprs.min(), worst_group_tprs.max()
    y_range = y_max - y_min
    if y_range < 1e-9:
        y_range = 1.0
    y_margin = 0.05 * y_range
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.set_xlabel('Total Edge Energy (Joules)', fontsize=12, fontweight='normal')
    ax.set_ylabel('Worst-Group TPR (Dangerous Classes)', fontsize=12, fontweight='normal')
    ax.set_title(f'Pareto Frontier: Energy vs. Safety Trade-off{title_suffix}',
                 fontsize=14, fontweight='normal', pad=15)
    ax.grid(True, alpha=0.3, axis='both', linestyle='--')
    ax.set_axisbelow(True)

    # Data export for PowerPoint: comma-separated lists
    print("\n--- Pareto Frontier Data Export ---")
    print("X (Energy, J):", ", ".join(f"{x:.2f}" for x in energies))
    print("Y (Worst-Group TPR):", ", ".join(f"{y:.4f}" for y in worst_group_tprs))

    plt.tight_layout()
    return fig
