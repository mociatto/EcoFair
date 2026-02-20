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

# Use default style (seaborn styles set legend.frameon=False which overrides our fixes)
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'


def plot_metadata_distributions(meta_df, dataset_name='HAM10000', dangerous_classes=None):
    """
    Plot metadata distributions with malignancy rate overlays.
    
    Replicates logic from Cell 02 of EcoFair_Main.py.
    Creates two subplots: Age Distribution and Localization Distribution
    with Malignancy Rates overlay.
    
    Args:
        meta_df: DataFrame with metadata columns (age, localization/region, dx/diagnostic/diagnosis)
        dataset_name: Name of dataset (for title)
        dangerous_classes: List of class names considered malignant.
                           Defaults to config.DANGEROUS_CLASSES if None.
    
    Returns:
        matplotlib.figure.Figure: Figure object with two subplots
    """
    if dangerous_classes is None:
        dangerous_classes = config.DANGEROUS_CLASSES
    
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
    for col in ['localization', 'region', 'lesion_location', 'anatom_site']:
        if col in meta_df_copy.columns:
            loc_col = col
            break
    if loc_col is None:
        meta_df_copy['_loc'] = 'unknown'
        loc_col = '_loc'
    
    # Age bins — handle missing or non-numeric age gracefully
    age_col = 'age' if 'age' in meta_df_copy.columns else None
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
    fig.suptitle(f'{dataset_name} — Metadata Distributions', fontsize=15, fontweight='normal', y=1.01)
    
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
    
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix_comparison(y_true, lite_preds, heavy_preds, dynamic_preds, class_names=None):
    """
    Plot side-by-side confusion matrices for Lite, Heavy, and Dynamic systems.
    
    Replicates logic from Cell 12b of EcoFair_Main.py.
    
    Args:
        y_true: True labels (class indices), shape (n_samples,)
        lite_preds: Lite predictions (class indices or probabilities)
        heavy_preds: Heavy predictions (class indices or probabilities)
        dynamic_preds: Dynamic predictions (class indices or probabilities)
        class_names: List of class names. If None, uses config.CLASS_NAMES
    
    Returns:
        matplotlib.figure.Figure: Figure object with three subplots
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
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


def plot_value_added_bars(y_true, lite_preds, heavy_preds, dynamic_preds, class_names=None, route_mask=None, ax=None):
    """
    Plot stacked bar chart showing value-added distribution per class.
    
    Replicates the "Value-Added" logic from Cell 12 of EcoFair_Main.py.
    Categorizes predictions into: Lite Only, Consensus, Heavy Rescued, Lite Rescued.
    Normalizes to 100% per class.
    
    Args:
        y_true: True labels (class indices), shape (n_samples,)
        lite_preds: Lite predictions (class indices or probabilities)
        heavy_preds: Heavy predictions (class indices or probabilities)
        dynamic_preds: Dynamic predictions (class indices or probabilities)
        class_names: List of class names. If None, uses config.CLASS_NAMES
        route_mask: Boolean array indicating which samples were routed to heavy model.
                   If None, will be calculated from predictions.
        ax: Optional matplotlib axes. If None, a new figure is created.
    
    Returns:
        matplotlib.figure.Figure: Figure object with stacked bar chart
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
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
    
    # Calculate route_mask if not provided
    if route_mask is None:
        route_mask = (lite_preds_class != heavy_preds_class)  # Simple heuristic
    
    # Initialize value-added tracking (matching original EcoFair_Main.py logic)
    class_value_added = {class_name: {
        'lite_only_correct': 0,      # Not routed, Lite Right
        'hybrid_consensus': 0,         # Routed, Both Right
        'heavy_rescued': 0,            # Routed, Lite Wrong, Heavy Right
        'lite_rescued': 0,             # Routed, Lite Right, Heavy Wrong
        'total_correct': 0
    } for class_name in class_names}
    
    # Categorize predictions (matching original logic exactly)
    for class_idx, class_name in enumerate(class_names):
        class_mask = (y_true == class_idx)
        if class_mask.sum() == 0:
            continue
        
        for i in np.where(class_mask)[0]:
            # We only analyze samples where the FINAL SYSTEM (Dynamic) was correct
            if dynamic_preds_class[i] == class_idx:
                class_value_added[class_name]['total_correct'] += 1
                
                if not route_mask[i]:
                    # Case A: Not Routed (Lite Only)
                    class_value_added[class_name]['lite_only_correct'] += 1
                else:
                    # Case B: Routed (Hybrid)
                    lite_right = (lite_preds_class[i] == class_idx)
                    heavy_right = (heavy_preds_class[i] == class_idx)
                    
                    if lite_right and heavy_right:
                        # Sub-case B1: Both models agreed on correct label (Consensus)
                        class_value_added[class_name]['hybrid_consensus'] += 1
                    elif not lite_right and heavy_right:
                        # Sub-case B2: Lite was wrong, Heavy saved it (Value Add)
                        class_value_added[class_name]['heavy_rescued'] += 1
                    elif lite_right and not heavy_right:
                        # Sub-case B3: Lite was right, Heavy was wrong (Rare, but kept correct by ensemble weights)
                        class_value_added[class_name]['lite_rescued'] += 1
    
    # Calculate percentages for each class (normalize to 100%)
    lite_only_pct = []
    hybrid_consensus_pct = []
    heavy_rescued_pct = []
    lite_rescued_pct = []
    class_names_list = []
    
    for class_name in class_names:
        # Get raw counts
        c_lite_only = class_value_added[class_name]['lite_only_correct']
        c_cons = class_value_added[class_name]['hybrid_consensus']
        c_h_resc = class_value_added[class_name]['heavy_rescued']
        c_l_resc = class_value_added[class_name]['lite_rescued']
        
        # FIX: We sum the PARTS to ensure the stack equals exactly 100%
        # This handles the rare edge case where Ensemble is right but individual models were "wrong"
        stack_total = c_lite_only + c_cons + c_h_resc + c_l_resc
        
        if stack_total > 0:
            l_only = (c_lite_only / stack_total) * 100
            h_cons = (c_cons / stack_total) * 100
            h_resc = (c_h_resc / stack_total) * 100
            l_resc = (c_l_resc / stack_total) * 100
        else:
            l_only = h_cons = h_resc = l_resc = 0
        
        lite_only_pct.append(l_only)
        hybrid_consensus_pct.append(h_cons)
        heavy_rescued_pct.append(h_resc)
        lite_rescued_pct.append(l_resc)
        class_names_list.append(class_name)
    
    # Plot stacked bars (matching original colors exactly)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    else:
        fig = ax.get_figure()
    
    x = np.arange(len(class_names_list))
    width = 0.65
    
    # Stack 1: Lite Only
    p1 = ax.bar(x, lite_only_pct, width, label='Lite Only (Unrouted)',
                color='lightgray', edgecolor='white', linewidth=2)
    
    # Stack 2: Hybrid Consensus
    p2 = ax.bar(x, hybrid_consensus_pct, width, bottom=lite_only_pct,
                label='Hybrid Consensus (Safety)',
                color='lightgreen', edgecolor='white', linewidth=2)
    
    # Stack 3: Heavy Rescued
    bottom_3 = np.array(lite_only_pct) + np.array(hybrid_consensus_pct)
    p3 = ax.bar(x, heavy_rescued_pct, width, bottom=bottom_3,
                label='Heavy Rescued (Value Add)',
                color='orangered', edgecolor='white', linewidth=2)
    
    # Stack 4: Lite Rescued
    bottom_4 = bottom_3 + np.array(heavy_rescued_pct)
    p4 = ax.bar(x, lite_rescued_pct, width, bottom=bottom_4,
                label='Lite Rescued (Hybrid)',
                color='skyblue', edgecolor='white', linewidth=2)
    
    # Formatting
    ax.set_ylabel('Contribution to Correct Predictions (%)', fontsize=12)
    ax.set_xlabel('Disease Class', fontsize=12)
    ax.set_title('Value-Added Analysis: Which Model is Driving Accuracy', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names_list)
    ax.set_ylim([0, 100])
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return fig


def plot_risk_stratified_accuracy(y_true, lite_preds, heavy_preds, dynamic_preds, meta_df, risk_scaler=None, ax=None):
    """
    Plot grouped bar chart comparing accuracy across risk groups.
    
    Replicates logic from Cell 13 of EcoFair_Main.py.
    Extracts risk scores, bins into Low/Medium/High, and plots accuracy comparison.
    
    Args:
        y_true: True labels (class indices), shape (n_samples,)
        lite_preds: Lite predictions (class indices or probabilities)
        heavy_preds: Heavy predictions (class indices or probabilities)
        dynamic_preds: Dynamic predictions (class indices or probabilities)
        meta_df: DataFrame with metadata (should have 'risk_score' or 'localization'/'age')
        risk_scaler: Optional MinMaxScaler for recalculating risk scores
        ax: Optional matplotlib axes. If None, a new figure is created.
    
    Returns:
        matplotlib.figure.Figure: Figure object with grouped bar chart
    """
    
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
    
    # Extract risk scores
    if 'risk_score' in meta_df.columns:
        risk_scores = meta_df['risk_score'].values
    elif risk_scaler is not None:
        risk_scores = features.calculate_cumulative_risk(meta_df, risk_scaler)
    else:
        # Fallback: use localization risk scores
        if 'localization' in meta_df.columns:
            risk_scores = meta_df['localization'].apply(features.get_sun_exposure_score).values
        else:
            risk_scores = np.full(len(meta_df), 0.5)
    
    # Bin into risk groups
    low_risk_mask = risk_scores < 0.3
    medium_risk_mask = (risk_scores >= 0.3) & (risk_scores < 0.6)
    high_risk_mask = risk_scores >= 0.6
    
    risk_groups = ['Low Risk\n(< 0.3)', 'Medium Risk\n(0.3 - 0.6)', 'High Risk\n(≥ 0.6)']
    risk_masks = [low_risk_mask, medium_risk_mask, high_risk_mask]
    
    def calculate_group_accuracy(y_true, y_pred, mask):
        if mask.sum() == 0:
            return 0.0
        return accuracy_score(y_true[mask], y_pred[mask])
    
    # Calculate accuracies for each risk group
    lite_accuracies = []
    heavy_accuracies = []
    dynamic_accuracies = []
    
    for mask in risk_masks:
        lite_acc = calculate_group_accuracy(y_true, lite_preds_class, mask)
        heavy_acc = calculate_group_accuracy(y_true, heavy_preds_class, mask)
        dynamic_acc = calculate_group_accuracy(y_true, dynamic_preds_class, mask)
        lite_accuracies.append(lite_acc)
        heavy_accuracies.append(heavy_acc)
        dynamic_accuracies.append(dynamic_acc)
    
    # Plot grouped bar chart
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    x = np.arange(len(risk_groups))
    width = 0.25
    
    bars1 = ax.bar(x - width, lite_accuracies, width, label='Pure Lite',
                  color='skyblue', edgecolor='white', linewidth=2.5)
    bars2 = ax.bar(x, heavy_accuracies, width, label='Pure Heavy',
                  color='orangered', edgecolor='white', linewidth=2.5)
    bars3 = ax.bar(x + width, dynamic_accuracies, width, label='EcoFair',
                  color='lightgreen', edgecolor='white', linewidth=2.5)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='normal')
    ax.set_title('Risk-Stratified Performance: Lite vs Heavy vs Dynamic',
                fontsize=14, fontweight='normal', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(risk_groups)
    ax.set_ylim([0, 1.0])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='normal')
    
    plt.tight_layout()
    
    return fig


def plot_battery_decay(lite_joules, heavy_joules, routing_rate, capacity_joules=10000, ax=None):
    """
    Plot battery decay curves for Pure Lite, Pure Heavy, and EcoFair systems.
    
    Replicates logic from Cell 13 of EcoFair_Main.py.
    Calculates sample capacity for each system and plots decay curves.
    
    Args:
        lite_joules: Energy consumption per sample for lite model (Joules)
        heavy_joules: Energy consumption per sample for heavy model (Joules)
        routing_rate: Fraction of samples routed to heavy model (0-1)
        capacity_joules: Battery capacity in Joules (default: 10000)
        ax: Optional matplotlib axes. If None, a new figure is created.
    
    Returns:
        matplotlib.figure.Figure: Figure object with battery decay plot
    """
    # Calculate samples each model can process
    samples_pure_lite = int(capacity_joules / lite_joules)
    samples_pure_heavy = int(capacity_joules / heavy_joules)
    
    # For EcoFair, calculate based on routing ratio
    avg_joules_per_sample_ecofair = lite_joules + (routing_rate * (heavy_joules - lite_joules))
    samples_ecofair = int(capacity_joules / avg_joules_per_sample_ecofair)
    
    # Create battery decay curves
    max_samples = max(samples_pure_lite, samples_pure_heavy, samples_ecofair)
    sample_range = np.arange(0, max_samples + 1, max(1, max_samples // 200))
    
    # Pure Lite battery decay
    battery_lite = 100 * (1 - (sample_range * lite_joules / capacity_joules))
    battery_lite = np.clip(battery_lite, 0, 100)
    
    # Pure Heavy battery decay
    battery_heavy = 100 * (1 - (sample_range * heavy_joules / capacity_joules))
    battery_heavy = np.clip(battery_heavy, 0, 100)
    
    # EcoFair battery decay
    battery_ecofair = 100 * (1 - (sample_range * avg_joules_per_sample_ecofair / capacity_joules))
    battery_ecofair = np.clip(battery_ecofair, 0, 100)
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    else:
        fig = ax.get_figure()
    
    ax.plot(sample_range, battery_lite, color='skyblue', linewidth=2.5, label='Pure Lite',
           linestyle='-', marker='o', markersize=4, markevery=max(1, len(sample_range)//50))
    ax.plot(sample_range, battery_heavy, color='orangered', linewidth=2.5, label='Pure Heavy',
           linestyle='-', marker='s', markersize=4, markevery=max(1, len(sample_range)//50))
    ax.plot(sample_range, battery_ecofair, color='lightgreen', linewidth=2.5, label='EcoFair',
           linestyle='-', marker='^', markersize=4, markevery=max(1, len(sample_range)//50))
    
    # Add critical low battery level (20%) as dashed horizontal line
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Critical Low (20%)')
    
    ax.set_xlabel('Number of Samples Processed', fontsize=12, fontweight='normal')
    ax.set_ylabel('Battery Level (%)', fontsize=12, fontweight='normal')
    ax.set_title('Battery Decay Comparison: Pure Lite vs Pure Heavy vs EcoFair',
                fontsize=14, fontweight='normal', pad=15)
    ax.set_ylim([0, 100])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return fig


def plot_routing_breakdown_doughnut(entropy, safe_danger_gap, route_mask, total_samples,
                                    ax=None, route_components=None):
    """
    Plot doughnut chart showing routing breakdown reasons.
    
    Replicates logic from Cell 12 of EcoFair_Main.py exactly.
    Shows reasons for Heavy Model Intervention: Uncertainty, Ambiguity, Safety, Multiple.
    
    Args:
        entropy: Entropy values, shape (n_samples,)
        safe_danger_gap: Safe-danger gap values, shape (n_samples,)
        route_mask: Boolean array indicating which samples were routed to heavy
        total_samples: Total number of samples
        ax: Optional matplotlib axes. If None, a new figure is created.
        route_components: Optional dict with keys 'uncertainty', 'ambiguity', 'safety'
                          (boolean arrays, full-length). When provided these are used
                          directly so the safety gate is accurate.
    
    Returns:
        matplotlib.figure.Figure: Figure object with doughnut chart
    """
    total_routed = route_mask.sum()
    
    if total_routed == 0:
        uncertainty_pct = ambiguity_pct = safety_pct = multiple_pct = 0
    else:
        if route_components is not None:
            # Use pre-computed components from apply_threshold_routing (most accurate)
            route_uncertainty = route_components['uncertainty'][route_mask]
            route_ambiguity   = route_components['ambiguity'][route_mask]
            route_safety      = route_components['safety'][route_mask]
        else:
            # Fallback: recompute from entropy/gap only (no safety gate)
            entropy_threshold = config.ENTROPY_THRESHOLD
            gap_threshold     = config.SAFE_DANGER_GAP_THRESHOLD
            route_uncertainty = entropy[route_mask] > entropy_threshold
            route_ambiguity   = safe_danger_gap[route_mask] < gap_threshold
            route_safety      = np.zeros(total_routed, dtype=bool)
        
        # Mutually-exclusive breakdown (matches EcoFair_Main.py lines 887-895)
        reason_uncertainty_only = route_uncertainty & ~route_ambiguity & ~route_safety
        reason_ambiguity_only   = route_ambiguity   & ~route_uncertainty & ~route_safety
        reason_safety_only      = route_safety      & ~route_uncertainty & ~route_ambiguity
        reason_multiple         = route_mask[route_mask] & ~(reason_uncertainty_only |
                                                             reason_ambiguity_only |
                                                             reason_safety_only)
        # reason_multiple covers everything else that was routed
        reason_multiple = ~(reason_uncertainty_only | reason_ambiguity_only | reason_safety_only)
        
        uncertainty_count = reason_uncertainty_only.sum()
        ambiguity_count   = reason_ambiguity_only.sum()
        safety_count      = reason_safety_only.sum()
        multiple_count    = reason_multiple.sum()
        
        if total_routed > 0:
            uncertainty_pct = uncertainty_count / total_routed * 100
            ambiguity_pct   = ambiguity_count   / total_routed * 100
            safety_pct      = safety_count      / total_routed * 100
            multiple_pct    = multiple_count    / total_routed * 100
        else:
            uncertainty_pct = ambiguity_pct = safety_pct = multiple_pct = 0
    
    # Create doughnut chart
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    sizes = [uncertainty_pct, ambiguity_pct, safety_pct, multiple_pct]
    labels_donut = ['Uncertainty\n(Entropy)', 'Ambiguity\n(Safe-Danger Gap)', 'Safety\n(Patient Risk)', 'Multiple\nReasons']
    colors_list = ['lightgray', 'lightgray', 'orangered', 'lightgray']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels_donut, colors=colors_list, autopct='%1.1f%%',
                                     startangle=90, pctdistance=0.85,
                                     textprops={'fontsize': 11, 'fontweight': 'normal'},
                                     wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    
    # Create center circle for doughnut effect
    centre_circle = plt.Circle((0, 0), 0.70, fc='white', ec='white', linewidth=2)
    ax.add_artist(centre_circle)
    
    # Add center text
    ax.text(0, 0, f'Routed {total_routed} | Out of {total_samples}', ha='center', va='center',
           fontsize=14, fontweight='normal')
    
    ax.set_title('Routing Breakdown: Reasons for Heavy Model Intervention',
                fontsize=14, fontweight='normal', pad=20)
    
    plt.tight_layout()
    
    return fig


def plot_classwise_accuracy_bars(y_true, lite_preds, heavy_preds, dynamic_preds, class_names=None):
    """
    Plot per-class accuracy bar charts for Lite, Heavy, and Dynamic systems.
    
    Replicates logic from Cell 12b of EcoFair_Main.py.
    
    Args:
        y_true: True labels (class indices), shape (n_samples,)
        lite_preds: Lite predictions (class indices or probabilities)
        heavy_preds: Heavy predictions (class indices or probabilities)
        dynamic_preds: Dynamic predictions (class indices or probabilities)
        class_names: List of class names. If None, uses config.CLASS_NAMES
    
    Returns:
        matplotlib.figure.Figure: Figure object with three subplots
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
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
    
    # Define colors
    colors = {
        'lite': 'skyblue',
        'heavy': 'orangered',
        'dynamic': 'lightgreen'
    }
    
    # Helper function for class-wise accuracy
    def calculate_class_wise_accuracy(y_true, y_pred, class_names):
        class_accuracies = {}
        for i, class_name in enumerate(class_names):
            true_mask = (y_true == i)
            if true_mask.sum() > 0:
                correct = ((y_true == i) & (y_pred == i)).sum()
                total = true_mask.sum()
                class_accuracies[class_name] = correct / total
            else:
                class_accuracies[class_name] = 0.0
        return class_accuracies
    
    # Compute class-wise accuracies
    class_acc_lite = calculate_class_wise_accuracy(y_true, lite_preds_class, class_names)
    class_acc_heavy = calculate_class_wise_accuracy(y_true, heavy_preds_class, class_names)
    class_acc_dynamic = calculate_class_wise_accuracy(y_true, dynamic_preds_class, class_names)
    
    # Plot class-wise accuracy bar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    systems_data = [
        (class_acc_lite, 'Pure Lite System', colors['lite']),
        (class_acc_heavy, 'Pure Heavy System', colors['heavy']),
        (class_acc_dynamic, 'Dynamic Routing System', colors['dynamic'])
    ]
    
    for ax, (class_acc_dict, title, bar_color) in zip(axes, systems_data):
        class_names_list = class_names
        accuracies_list = [class_acc_dict[cn] for cn in class_names_list]
        
        bars = ax.bar(class_names_list, accuracies_list, color=bar_color, linewidth=1.5)
        
        ax.set_ylabel('Class-wise Accuracy', fontsize=12, fontweight='normal')
        ax.set_title(title, fontsize=14, fontweight='normal', pad=15)
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xticks(range(len(class_names_list)))
        ax.set_xticklabels(class_names_list, rotation=0)
        
        for bar, acc in zip(bars, accuracies_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{acc:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='normal')
    
    plt.tight_layout()
    
    return fig


def plot_gender_age_accuracy(y_true, lite_preds, heavy_preds, dynamic_preds, meta_df, class_names=None, axes=None):
    """
    Plot model accuracy by Gender and Age categories.
    
    Replicates logic from Cell 13 of EcoFair_Main.py.
    
    Args:
        y_true: True labels (class indices), shape (n_samples,)
        lite_preds: Lite predictions (class indices or probabilities)
        heavy_preds: Heavy predictions (class indices or probabilities)
        dynamic_preds: Dynamic predictions (class indices or probabilities)
        meta_df: DataFrame with metadata columns ('age', 'sex' or 'gender')
        class_names: List of class names. If None, uses config.CLASS_NAMES
        axes: Optional tuple (ax1, ax2) to draw on. If None, a new figure is created.
    
    Returns:
        matplotlib.figure.Figure: Figure object with two subplots (Gender and Age)
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
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
    
    # Define colors
    color_lite = 'skyblue'
    color_heavy = 'orangered'
    color_ecofair = 'lightgreen'
    
    # Create figure with two subplots or use provided axes
    if axes is not None:
        ax1, ax2 = axes[0], axes[1]
        fig = ax1.get_figure()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Gender-based Grouped Bar Chart
    meta_df_copy = meta_df.copy()
    
    # Normalize sex/gender column
    if 'sex' in meta_df_copy.columns:
        sex_col = 'sex'
    elif 'gender' in meta_df_copy.columns:
        sex_col = 'gender'
    else:
        sex_col = None
    
    gender_categories = ['Male', 'Female', 'Unknown']
    gender_acc_lite = []
    gender_acc_heavy = []
    gender_acc_ecofair = []
    
    for gender in gender_categories:
        if sex_col:
            if gender == 'Unknown':
                gender_mask = (meta_df_copy[sex_col].astype(str).str.lower().str.strip().isin(['unknown', 'nan', 'none', ''])) | meta_df_copy[sex_col].isna()
            else:
                gender_mask = meta_df_copy[sex_col].astype(str).str.lower().str.strip() == gender.lower()
        else:
            gender_mask = np.zeros(len(meta_df_copy), dtype=bool)
            if gender == 'Unknown':
                gender_mask[:] = True
        
        if gender_mask.sum() > 0:
            acc_lite = accuracy_score(y_true[gender_mask], lite_preds_class[gender_mask])
            acc_heavy = accuracy_score(y_true[gender_mask], heavy_preds_class[gender_mask])
            acc_ecofair = accuracy_score(y_true[gender_mask], dynamic_preds_class[gender_mask])
        else:
            acc_lite = acc_heavy = acc_ecofair = 0.0
        
        gender_acc_lite.append(acc_lite)
        gender_acc_heavy.append(acc_heavy)
        gender_acc_ecofair.append(acc_ecofair)
    
    x_gender = np.arange(len(gender_categories))
    width = 0.25
    
    bars1 = ax1.bar(x_gender - width, gender_acc_lite, width, label='Pure Lite',
                   color=color_lite, edgecolor='white', linewidth=2)
    bars2 = ax1.bar(x_gender, gender_acc_heavy, width, label='Pure Heavy',
                   color=color_heavy, edgecolor='white', linewidth=2)
    bars3 = ax1.bar(x_gender + width, gender_acc_ecofair, width, label='EcoFair',
                   color=color_ecofair, edgecolor='white', linewidth=2)
    
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='normal')
    ax1.set_title('Model Accuracy by Gender Category', fontsize=14, fontweight='normal', pad=15)
    ax1.set_xticks(x_gender)
    ax1.set_xticklabels(gender_categories)
    ax1.set_ylim([0, 1.0])
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.xaxis.grid(False)
    ax1.set_axisbelow(True)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='normal')
    
    # 2. Age-based Grouped Bar Chart
    age_categories = ['<30', '30-60', '60+']
    age_acc_lite = []
    age_acc_heavy = []
    age_acc_ecofair = []
    
    if 'age' in meta_df_copy.columns:
        age_values = meta_df_copy['age'].values
        age_mask_young = age_values < 30
        age_mask_middle = (age_values >= 30) & (age_values < 60)
        age_mask_old = age_values >= 60
        
        age_masks = [age_mask_young, age_mask_middle, age_mask_old]
        
        for age_mask in age_masks:
            if age_mask.sum() > 0:
                acc_lite = accuracy_score(y_true[age_mask], lite_preds_class[age_mask])
                acc_heavy = accuracy_score(y_true[age_mask], heavy_preds_class[age_mask])
                acc_ecofair = accuracy_score(y_true[age_mask], dynamic_preds_class[age_mask])
            else:
                acc_lite = acc_heavy = acc_ecofair = 0.0
            
            age_acc_lite.append(acc_lite)
            age_acc_heavy.append(acc_heavy)
            age_acc_ecofair.append(acc_ecofair)
    else:
        age_acc_lite = [0.0, 0.0, 0.0]
        age_acc_heavy = [0.0, 0.0, 0.0]
        age_acc_ecofair = [0.0, 0.0, 0.0]
    
    x_age = np.arange(len(age_categories))
    
    bars1_age = ax2.bar(x_age - width, age_acc_lite, width, label='Pure Lite',
                        color=color_lite, edgecolor='white', linewidth=2)
    bars2_age = ax2.bar(x_age, age_acc_heavy, width, label='Pure Heavy',
                        color=color_heavy, edgecolor='white', linewidth=2)
    bars3_age = ax2.bar(x_age + width, age_acc_ecofair, width, label='EcoFair',
                        color=color_ecofair, edgecolor='white', linewidth=2)
    
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='normal')
    ax2.set_title('Model Accuracy by Age Category', fontsize=14, fontweight='normal', pad=15)
    ax2.set_xticks(x_age)
    ax2.set_xticklabels(age_categories)
    ax2.set_ylim([0, 1.0])
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.xaxis.grid(False)
    ax2.set_axisbelow(True)
    
    for bars in [bars1_age, bars2_age, bars3_age]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=9, fontweight='normal')
    
    plt.tight_layout()
    
    return fig


def plot_fairness_disparity(fairness_lite, fairness_heavy, fairness_ecofair, dangerous_classes=None):
    """
    Plot side-by-side Gender and Age disparity gaps for dangerous classes.
    
    Left: |TPR_Male - TPR_Female|. Right: Max(TPR) - Min(TPR) across age groups.
    Lower gap indicates fairer performance. Filters out subgroups containing 'Unknown'.
    
    Args:
        fairness_lite: DataFrame from generate_fairness_report for lite model
        fairness_heavy: DataFrame from generate_fairness_report for heavy model
        fairness_ecofair: DataFrame from generate_fairness_report for EcoFair (dynamic) model
        dangerous_classes: List of dangerous class names. Defaults to config.DANGEROUS_CLASSES.
    
    Returns:
        matplotlib.figure.Figure: Figure with two subplots
    """
    if dangerous_classes is None:
        dangerous_classes = list(config.DANGEROUS_CLASSES)
    else:
        dangerous_classes = list(dangerous_classes)
    
    def _filter_unknown(df):
        """Remove rows where Subgroup contains 'Unknown'."""
        if df.empty or 'Subgroup' not in df.columns:
            return df
        return df[~df['Subgroup'].str.contains('Unknown', case=False, na=False)].copy()
    
    def _strip_count(series):
        """Strip ' (n=...)' suffix from Subgroup for matching."""
        return series.str.replace(r'\s*\(n=\d+\)\s*$', '', regex=True).str.strip()
    
    def _get_gender_gap(df):
        """|TPR_Male - TPR_Female| for each dangerous class."""
        df = _filter_unknown(df)
        gaps = {}
        for cls in dangerous_classes:
            df_cls = df[(df['Class'] == cls) & (df['Subgroup'].str.startswith('Sex: '))]
            if df_cls.empty:
                gaps[cls] = np.nan
                continue
            df_cls = df_cls.copy()
            df_cls['_sub'] = _strip_count(df_cls['Subgroup'])
            piv = df_cls.drop_duplicates('_sub').set_index('_sub')['Equal_Opportunity_TPR']
            male_tpr = piv.get('Sex: Male', np.nan)
            female_tpr = piv.get('Sex: Female', np.nan)
            if pd.notna(male_tpr) and pd.notna(female_tpr):
                gaps[cls] = abs(float(male_tpr) - float(female_tpr))
            else:
                gaps[cls] = np.nan
        return gaps
    
    def _get_age_gap(df):
        """Max(TPR) - Min(TPR) across <30, 30-60, 60+ for each dangerous class."""
        df = _filter_unknown(df)
        gaps = {}
        age_subs = ['Age <30', 'Age 30-60', 'Age 60+']
        for cls in dangerous_classes:
            df_cls = df[(df['Class'] == cls) & (df['Subgroup'].str.startswith('Age '))]
            if df_cls.empty:
                gaps[cls] = np.nan
                continue
            df_cls = df_cls.copy()
            df_cls['_sub'] = _strip_count(df_cls['Subgroup'])
            piv = df_cls.drop_duplicates('_sub').set_index('_sub')['Equal_Opportunity_TPR']
            vals = [float(piv.get(s, np.nan)) for s in age_subs if s in piv.index]
            vals = [v for v in vals if pd.notna(v)]
            gaps[cls] = (float(np.max(vals)) - float(np.min(vals))) if len(vals) >= 2 else np.nan
        return gaps
    
    def _safe(v):
        return v if pd.notna(v) else 0.0
    
    gaps_lite_g = _get_gender_gap(fairness_lite)
    gaps_heavy_g = _get_gender_gap(fairness_heavy)
    gaps_ecofair_g = _get_gender_gap(fairness_ecofair)
    
    gaps_lite_a = _get_age_gap(fairness_lite)
    gaps_heavy_a = _get_age_gap(fairness_heavy)
    gaps_ecofair_a = _get_age_gap(fairness_ecofair)
    
    x = np.arange(len(dangerous_classes))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Gender Disparity
    ax1.bar(x - width, [_safe(gaps_lite_g.get(c, np.nan)) for c in dangerous_classes],
            width, label='Pure Lite', color='skyblue')
    ax1.bar(x, [_safe(gaps_heavy_g.get(c, np.nan)) for c in dangerous_classes],
            width, label='Pure Heavy', color='orangered')
    ax1.bar(x + width, [_safe(gaps_ecofair_g.get(c, np.nan)) for c in dangerous_classes],
            width, label='EcoFair', color='lightgreen')
    ax1.set_xlabel('Dangerous Class', fontsize=12)
    ax1.set_ylabel('Absolute TPR Disparity', fontsize=12)
    ax1.set_title('Gender Disparity (|TPR_Male - TPR_Female|)', fontsize=12, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dangerous_classes)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.xaxis.grid(False)
    
    # Right: Age Disparity
    ax2.bar(x - width, [_safe(gaps_lite_a.get(c, np.nan)) for c in dangerous_classes],
            width, label='Pure Lite', color='skyblue')
    ax2.bar(x, [_safe(gaps_heavy_a.get(c, np.nan)) for c in dangerous_classes],
            width, label='Pure Heavy', color='orangered')
    ax2.bar(x + width, [_safe(gaps_ecofair_a.get(c, np.nan)) for c in dangerous_classes],
            width, label='EcoFair', color='lightgreen')
    ax2.set_xlabel('Dangerous Class', fontsize=12)
    ax2.set_ylabel('Absolute TPR Disparity', fontsize=12)
    ax2.set_title('Age Disparity (Max - Min TPR across <30, 30-60, 60+)', fontsize=12, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dangerous_classes)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.xaxis.grid(False)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.5, 0.02, 'Lower gap indicates fairer performance across demographic.',
             ha='center', fontsize=10, bbox=props)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig
