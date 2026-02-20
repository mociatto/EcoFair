# %% Imports
import sys
import os
import subprocess

# Ensure EcoFair src is on path: clone repo if not present (e.g. on Kaggle)
REPO_URL = "https://github.com/mociatto/EcoFair.git"
REPO_DIR = "EcoFair"

if os.path.exists("./src"):
    sys.path.insert(0, os.path.abspath("."))
elif os.path.exists(os.path.join(".", REPO_DIR, "src")):
    sys.path.insert(0, os.path.abspath(REPO_DIR))
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    sys.path.insert(0, os.path.abspath(REPO_DIR))

from src import config, utils, data_loader, models, training, features, routing, fairness, visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

utils.set_seed(config.RANDOM_STATE)
print(f"EcoFair v{config.VERSION} loaded successfully.")
print(f"Using models: {config.SELECTED_LITE_MODEL} (Lite) and {config.SELECTED_HEAVY_MODEL} (Heavy)")

# %% Load Data & Features
print("\nLoading HAM10000 data...")
X_heavy, X_lite, meta_ham = data_loader.load_and_align_ham()
print(f"Loaded {len(meta_ham)} samples")

print("\nPreparing tabular features...")
X_tab, scaler, sex_encoder, loc_encoder, risk_scaler = features.prepare_tabular_features(meta_ham)
y_ham, _ = features.prepare_labels(meta_ham, config.CLASS_NAMES)
meta_ham = meta_ham.copy()
meta_ham['risk_score'] = features.calculate_cumulative_risk(meta_ham, risk_scaler)

print("\nPlotting metadata distributions...")
fig_meta = visualization.plot_metadata_distributions(meta_ham, dataset_name='HAM10000')
plt.show()

# %% PART 1: 5-Fold Cross Validation Benchmark (HAM10000)
print("\n" + "="*70)
print("PART 1: 5-FOLD CROSS-VALIDATION BENCHMARK (HAM10000)")
print("="*70)

fold_metrics, oof_lite, oof_heavy, oof_dynamic, route_mask_oof, route_components_oof = training.run_cv_pipeline(
    X_heavy, X_lite, X_tab, y_ham, meta_ham, n_splits=5, risk_scaler=risk_scaler
)

print("\n--- Cross-Validation Results (5 Folds) ---")
print(f"Lite Accuracy:    {np.mean(fold_metrics['acc_lite']):.4f} ± {np.std(fold_metrics['acc_lite']):.4f}")
print(f"Heavy Accuracy:   {np.mean(fold_metrics['acc_heavy']):.4f} ± {np.std(fold_metrics['acc_heavy']):.4f}")
print(f"EcoFair Accuracy: {np.mean(fold_metrics['acc_dynamic']):.4f} ± {np.std(fold_metrics['acc_dynamic']):.4f}")
print(f"Routing Rate:     {np.mean(fold_metrics['routing_rate'])*100:.2f}% ± {np.std(fold_metrics['routing_rate'])*100:.2f}%")
print(f"Energy per Sample:{np.mean(fold_metrics['energy_cost']):.2f} J ± {np.std(fold_metrics['energy_cost']):.2f} J")

# %% PART 2: Visualizations & Fairness (Using OOF Predictions)
print("\n" + "="*70)
print("PART 2: VISUALIZATIONS & FAIRNESS (OOF)")
print("="*70)

y_true_oof = np.argmax(y_ham, axis=1)
entropy_oof = routing.calculate_entropy(oof_lite)
safe_indices = [config.CLASS_NAMES.index(c) for c in config.SAFE_CLASSES]
danger_indices = [config.CLASS_NAMES.index(c) for c in config.DANGEROUS_CLASSES]
prob_safe_oof = oof_lite[:, safe_indices].sum(axis=1)
prob_danger_oof = oof_lite[:, danger_indices].sum(axis=1)
safe_danger_gap_oof = prob_safe_oof - prob_danger_oof

joules_lite = utils.load_energy_stats(config.SELECTED_LITE_MODEL, is_heavy=False)
joules_heavy = utils.load_energy_stats(config.SELECTED_HEAVY_MODEL, is_heavy=True)
joules_lite = joules_lite if joules_lite is not None else 1.0
joules_heavy = joules_heavy if joules_heavy is not None else 2.5
routing_rate_oof = route_mask_oof.sum() / len(route_mask_oof)

# Confusion Matrices
print("\nPlotting confusion matrices...")
fig_cm = visualization.plot_confusion_matrix_comparison(
    y_true_oof, oof_lite, oof_heavy, oof_dynamic
)
plt.show()

# Per-Class Accuracy
print("\nPlotting per-class accuracy bar charts...")
fig_classwise = visualization.plot_classwise_accuracy_bars(
    y_true_oof, oof_lite, oof_heavy, oof_dynamic
)
plt.show()

# Value-Added & Routing Breakdown
print("\nPlotting value-added analysis and routing breakdown...")
fig_va_doughnut, axes_va_doughnut = plt.subplots(1, 2, figsize=(18, 7))
visualization.plot_value_added_bars(
    y_true_oof, oof_lite, oof_heavy, oof_dynamic,
    route_mask=route_mask_oof, ax=axes_va_doughnut[0]
)
visualization.plot_routing_breakdown_doughnut(
    entropy_oof, safe_danger_gap_oof, route_mask_oof, len(route_mask_oof),
    ax=axes_va_doughnut[1], route_components=route_components_oof
)
plt.tight_layout()
plt.show()

# Comprehensive Performance (Gender, Age, Risk, Battery)
print("\nPlotting comprehensive performance analysis...")
try:
    fig_comprehensive, axes_comprehensive = plt.subplots(2, 2, figsize=(16, 12))
    visualization.plot_gender_age_accuracy(
        y_true_oof, oof_lite, oof_heavy, oof_dynamic, meta_ham,
        axes=(axes_comprehensive[0, 0], axes_comprehensive[0, 1])
    )
    visualization.plot_risk_stratified_accuracy(
        y_true_oof, oof_lite, oof_heavy, oof_dynamic, meta_ham,
        risk_scaler=risk_scaler, ax=axes_comprehensive[1, 0]
    )
    visualization.plot_battery_decay(
        joules_lite, joules_heavy, routing_rate_oof, capacity_joules=10000,
        ax=axes_comprehensive[1, 1]
    )
    plt.tight_layout()
    plt.show()
except TypeError:
    try:
        plt.close(fig_comprehensive)
    except NameError:
        pass
    fig_gender_age = visualization.plot_gender_age_accuracy(
        y_true_oof, oof_lite, oof_heavy, oof_dynamic, meta_ham
    )
    plt.show()
    fig_risk_battery, axes_risk_battery = plt.subplots(1, 2, figsize=(20, 6))
    visualization.plot_risk_stratified_accuracy(
        y_true_oof, oof_lite, oof_heavy, oof_dynamic, meta_ham,
        risk_scaler=risk_scaler, ax=axes_risk_battery[0]
    )
    visualization.plot_battery_decay(
        joules_lite, joules_heavy, routing_rate_oof, capacity_joules=10000,
        ax=axes_risk_battery[1]
    )
    plt.tight_layout()
    plt.show()

# %% PART 3: Fairness Audit
print("\n" + "="*70)
print("PART 3: SYSTEMATIC FAIRNESS AUDIT (One-vs-Rest)")
print("="*70)

y_true_labels = np.argmax(y_ham, axis=1)
lite_pred_labels = np.argmax(oof_lite, axis=1)
heavy_pred_labels = np.argmax(oof_heavy, axis=1)
dynamic_pred_labels = np.argmax(oof_dynamic, axis=1)

fairness_ecofair = fairness.generate_fairness_report(
    y_true_labels, dynamic_pred_labels, meta_ham, config.CLASS_NAMES
)
fairness_lite = fairness.generate_fairness_report(
    y_true_labels, lite_pred_labels, meta_ham, config.CLASS_NAMES
)
fairness_heavy = fairness.generate_fairness_report(
    y_true_labels, heavy_pred_labels, meta_ham, config.CLASS_NAMES
)

print("\nEcoFair Fairness Metrics across Demographic Subgroups and Classes:")
fairness.print_fairness_audit(fairness_ecofair)

print("\nGenerating Fairness Disparity Visualizations...")
visualization.plot_fairness_disparity(fairness_lite, fairness_heavy, fairness_ecofair)
plt.show()
