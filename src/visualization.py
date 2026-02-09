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

# Set clean style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')


def plot_metadata_distributions(meta_df, dataset_name='HAM10000'):
    """
    Plot metadata distributions with malignancy rate overlays.
    
    Replicates logic from Cell 02 of EcoFair_Main.py.
    Creates two subplots: Age Distribution and Localization Distribution
    with Malignancy Rates overlay.
    
    Args:
        meta_df: DataFrame with metadata columns ('age', 'localization', 'dx')
        dataset_name: Name of dataset (for title)
    
    Returns:
        matplotlib.figure.Figure: Figure object with two subplots
    """
    meta_df_copy = meta_df.copy()
    
    # Create malignant indicator
    if 'dx' in meta_df_copy.columns:
        meta_df_copy['is_malignant'] = meta_df_copy['dx'].isin(config.DANGEROUS_CLASSES).astype(int)
    else:
        meta_df_copy['is_malignant'] = 0
    
    # Age bins
    age_bins = np.arange(0, 101, 10)
    meta_df_copy['age_bin'] = pd.cut(
        meta_df_copy['age'],
        bins=age_bins,
        labels=[f"{int(age_bins[i])}-{int(age_bins[i+1])}" for i in range(len(age_bins)-1)],
        include_lowest=True
    )
    
    age_counts = meta_df_copy.groupby('age_bin', observed=True).size()
    age_malignancy_rate = meta_df_copy.groupby('age_bin', observed=True)['is_malignant'].mean() * 100
    
    # Localization
    loc_malignancy_rate = meta_df_copy.groupby('localization')['is_malignant'].mean() * 100
    loc_malignancy_rate = loc_malignancy_rate.sort_values(ascending=True)
    loc_counts = meta_df_copy.groupby('localization').size()
    loc_counts = loc_counts.reindex(loc_malignancy_rate.index)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Age distribution
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    bars1 = ax1.bar(range(len(age_counts)), age_counts.values, alpha=0.7, color='skyblue', label='Sample Count')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='normal')
    ax1.set_ylim([0, 3000])
    ax1.set_xticks(range(len(age_counts)))
    ax1.set_xticklabels(age_counts.index, rotation=90, ha='right')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3, axis='y')
    
    line1 = ax1_twin.plot(range(len(age_malignancy_rate)), age_malignancy_rate.values,
                          color='orangered', marker='o', linewidth=2, markersize=8, label='Malignancy Rate')
    ax1_twin.set_ylim([0, 100])
    ax1_twin.tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False)
    
    ax1.set_title('Age Distribution and Malignancy Rate', fontsize=14, fontweight='normal', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1_twin.legend(loc='upper right', fontsize=10)
    
    # Localization distribution
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    bars2 = ax2.bar(range(len(loc_counts)), loc_counts.values, alpha=0.7, color='lightgreen', label='Sample Count')
    ax2.set_ylim([0, 3000])
    ax2.set_xticks(range(len(loc_counts)))
    ax2.set_xticklabels([loc.title() for loc in loc_counts.index], rotation=90, ha='right')
    ax2.tick_params(axis='y', labelleft=False, left=False)
    ax2.grid(True, alpha=0.3, axis='y')
    
    line2 = ax2_twin.plot(range(len(loc_malignancy_rate)), loc_malignancy_rate.values,
                          color='orangered', marker='o', linewidth=2, markersize=8, label='Malignancy Rate')
    ax2_twin.set_ylabel('Malignancy Rate (%)', fontsize=12, fontweight='normal')
    ax2_twin.tick_params(axis='y')
    ax2_twin.set_ylim([0, 100])
    
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


def plot_value_added_bars(y_true, lite_preds, heavy_preds, dynamic_preds, class_names=None, route_mask=None):
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
    
    # Initialize value-added tracking
    class_value_added = {class_name: {
        'rescued': 0, 'consensus': 0, 'poisoned': 0, 'lost': 0, 'lite_only_correct': 0
    } for class_name in class_names}
    
    # Categorize predictions
    for i in range(len(y_true)):
        true_class = y_true[i]
        lite_correct = (lite_preds_class[i] == true_class)
        dynamic_correct = (dynamic_preds_class[i] == true_class)
        routed = route_mask[i]
        
        if routed:
            if not lite_correct and dynamic_correct:
                class_value_added[class_names[true_class]]['rescued'] += 1
            elif lite_correct and dynamic_correct:
                class_value_added[class_names[true_class]]['consensus'] += 1
            elif lite_correct and not dynamic_correct:
                class_value_added[class_names[true_class]]['poisoned'] += 1
            else:
                class_value_added[class_names[true_class]]['lost'] += 1
        else:
            if lite_correct:
                class_value_added[class_names[true_class]]['lite_only_correct'] += 1
    
    # Calculate percentages for each class (normalize to 100%)
    categories = ['rescued', 'consensus', 'poisoned', 'lost', 'lite_only_correct']
    category_labels = ['Rescued', 'Consensus', 'Poisoned', 'Lost', 'Lite']
    
    category_percentages = {}
    for cat in categories:
        category_percentages[cat] = []
        for class_name in class_names:
            va = class_value_added[class_name]
            total = sum(va.values())
            if total > 0:
                pct = (va[cat] / total) * 100
            else:
                pct = 0
            category_percentages[cat].append(pct)
    
    # Plot stacked bars
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    x = np.arange(len(class_names))
    width = 0.7
    bottoms = np.zeros(len(class_names))
    
    for cat_idx, (cat, label) in enumerate(zip(categories, category_labels)):
        values_pct = category_percentages[cat]
        
        if cat == 'poisoned':
            color = 'orangered'
        elif cat == 'rescued':
            color = 'lightgreen'
        elif cat == 'consensus':
            color = 'lightblue'
        elif cat == 'lost':
            color = 'gray'
        else:  # lite_only_correct
            color = 'skyblue'
        
        bars = ax.bar(x, values_pct, width, bottom=bottoms, label=label, color=color,
                     edgecolor='white', linewidth=2)
        bottoms += values_pct
    
    ax.set_title('Value-Added Distribution: Heavy Model Contribution Analysis',
                fontsize=14, fontweight='normal', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_ylim([0, 100])
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return fig


def plot_risk_stratified_accuracy(y_true, lite_preds, heavy_preds, dynamic_preds, meta_df, risk_scaler=None):
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(risk_groups))
    width = 0.25
    
    bars1 = ax.bar(x - width, lite_accuracies, width, label='Lite Accuracy',
                  color='skyblue', edgecolor='white', linewidth=2.5)
    bars2 = ax.bar(x, heavy_accuracies, width, label='Heavy Accuracy',
                  color='orangered', edgecolor='white', linewidth=2.5)
    bars3 = ax.bar(x + width, dynamic_accuracies, width, label='Dynamic Accuracy',
                  color='lightgreen', edgecolor='white', linewidth=2.5)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='normal')
    ax.set_title('Risk-Stratified Performance: Lite vs Heavy vs Dynamic',
                fontsize=14, fontweight='normal', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(risk_groups)
    ax.set_ylim([0, 1.0])
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
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


def plot_battery_decay(lite_joules, heavy_joules, routing_rate, capacity_joules=10000):
    """
    Plot battery decay curves for Pure Lite, Pure Heavy, and EcoFair systems.
    
    Replicates logic from Cell 13 of EcoFair_Main.py.
    Calculates sample capacity for each system and plots decay curves.
    
    Args:
        lite_joules: Energy consumption per sample for lite model (Joules)
        heavy_joules: Energy consumption per sample for heavy model (Joules)
        routing_rate: Fraction of samples routed to heavy model (0-1)
        capacity_joules: Battery capacity in Joules (default: 10000)
    
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
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
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return fig
