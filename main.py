# %% Imports
import sys
import os
import subprocess

# Ensure EcoFair src is on path: clone repo if not present (e.g. on Kaggle)
REPO_URL = "https://github.com/mociatto/EcoFair.git"
REPO_DIR = "EcoFair"

if os.path.exists("./src"):
    # Already inside the repo (e.g. running locally)
    sys.path.insert(0, os.path.abspath("."))
elif os.path.exists(os.path.join(".", REPO_DIR, "src")):
    # Repo already cloned in current directory
    sys.path.insert(0, os.path.abspath(REPO_DIR))
else:
    # Clone the repo so we can import src
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    sys.path.insert(0, os.path.abspath(REPO_DIR))

from src import config, utils, data_loader, models, training, features, routing, fairness, visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Set reproducibility
utils.set_seed(config.RANDOM_STATE)

print(f"EcoFair v{config.VERSION} loaded successfully.")
print(f"Using models: {config.SELECTED_LITE_MODEL} (Lite) and {config.SELECTED_HEAVY_MODEL} (Heavy)")

# %% Load and align HAM10000 data
print("\nLoading HAM10000 data...")
X_heavy, X_lite, meta_ham = data_loader.load_and_align_ham()
print(f"Loaded {len(meta_ham)} samples")
print(f"Heavy features shape: {X_heavy.shape}")
print(f"Lite features shape: {X_lite.shape}")

# %% Plot metadata distributions (malignancy rate vs age and localization)
print("\nPlotting metadata distributions...")
fig_meta = visualization.plot_metadata_distributions(meta_ham, dataset_name='HAM10000')
plt.show()

# %% Prepare tabular features with Neurosymbolic Risk Scoring
print("\nPreparing tabular features...")
X_tab, scaler, sex_encoder, loc_encoder, risk_scaler = features.prepare_tabular_features(meta_ham)
print(f"Tabular features shape: {X_tab.shape}")

# Prepare labels
y_ham, dx_to_idx = features.prepare_labels(meta_ham)
print(f"Labels shape: {y_ham.shape}")

# %% Split data using stratified group K-fold
y_labels_ham = np.argmax(y_ham, axis=1)
splits = data_loader.get_stratified_split(meta_ham, y_labels_ham, n_splits=5)
train_idx, test_idx = list(splits)[0]

# Further split train into train/val
meta_train = meta_ham.iloc[train_idx].reset_index(drop=True)
y_train_labels = y_labels_ham[train_idx]
splits_val = data_loader.get_stratified_split(meta_train, y_train_labels, n_splits=5)
train_idx_final, val_idx = list(splits_val)[0]

# Get absolute indices
train_idx_abs = train_idx[train_idx_final]
val_idx_abs = train_idx[val_idx]

# Split features
X_lite_train = X_lite[train_idx_abs]
X_lite_val = X_lite[val_idx_abs]
X_lite_test = X_lite[test_idx]

X_heavy_train = X_heavy[train_idx_abs]
X_heavy_val = X_heavy[val_idx_abs]
X_heavy_test = X_heavy[test_idx]

X_tab_train = X_tab[train_idx_abs]
X_tab_val = X_tab[val_idx_abs]
X_tab_test = X_tab[test_idx]

y_train = y_ham[train_idx_abs]
y_val = y_ham[val_idx_abs]
y_test = y_ham[test_idx]

meta_test = meta_ham.iloc[test_idx].reset_index(drop=True)
meta_val = meta_train.iloc[val_idx].reset_index(drop=True)

print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# %% Build models
print("\nBuilding VFL models...")
lite_adapter = models.build_image_adapter(feature_dim=X_lite.shape[1], embedding_dim=128)
heavy_adapter = models.build_image_adapter(feature_dim=X_heavy.shape[1], embedding_dim=128)
tab_client = models.build_tabular_client(input_dim=X_tab.shape[1], embedding_dim=128)
server_head = models.build_server_head(input_dim=256, num_classes=len(config.CLASS_NAMES))

lite_model = models.build_vfl_model(lite_adapter, tab_client, server_head)
heavy_model = models.build_vfl_model(heavy_adapter, tab_client, server_head)

print("Models built successfully.")

# %% Get class weights
class_weight_dict = training.get_class_weights(y_train)
print("\nClass weights:")
for i, class_name in enumerate(config.CLASS_NAMES):
    print(f"  {class_name}: {class_weight_dict[i]:.4f}")

# %% Train Lite Model
print("\nTraining Lite Model...")
lite_history = training.compile_and_train(
    lite_model, X_lite_train, X_tab_train, y_train,
    X_lite_val, X_tab_val, y_val,
    class_weight=class_weight_dict
)
print("Lite model training complete.")

# %% Train Heavy Model
print("\nTraining Heavy Model...")
heavy_history = training.compile_and_train(
    heavy_model, X_heavy_train, X_tab_train, y_train,
    X_heavy_val, X_tab_val, y_val,
    class_weight=class_weight_dict
)
print("Heavy model training complete.")

# %% Generate predictions on validation set
print("\nGenerating predictions on validation set...")
lite_preds_val = lite_model.predict([X_lite_val, X_tab_val], batch_size=config.BATCH_SIZE, verbose=0)
heavy_preds_val = heavy_model.predict([X_heavy_val, X_tab_val], batch_size=config.BATCH_SIZE, verbose=0)

y_true_val = np.argmax(y_val, axis=1)

# Calculate entropy and safe-danger gap
entropy_val = routing.calculate_entropy(lite_preds_val)
safe_indices = [config.CLASS_NAMES.index(c) for c in config.SAFE_CLASSES]
danger_indices = [config.CLASS_NAMES.index(c) for c in config.DANGEROUS_CLASSES]
prob_safe_val = lite_preds_val[:, safe_indices].sum(axis=1)
prob_danger_val = lite_preds_val[:, danger_indices].sum(axis=1)
safe_danger_gap_val = prob_safe_val - prob_danger_val

# Calculate baseline accuracy
heavy_baseline_acc = accuracy_score(y_true_val, np.argmax(heavy_preds_val, axis=1))
print(f"Heavy baseline accuracy: {heavy_baseline_acc:.4f}")

# %% Optimize thresholds using SafetyFirstOptimizer
print("\nOptimizing routing thresholds...")
optimizer = routing.SafetyFirstOptimizer(
    lite_preds_val, heavy_preds_val, y_true_val,
    entropy_val, safe_danger_gap_val, heavy_baseline_acc
)

optimal_config, all_results = optimizer.optimize()

print(f"\nOptimal Configuration:")
print(f"  Entropy Threshold: {optimal_config['entropy_t']:.2f}")
print(f"  Gap Threshold: {optimal_config['gap_t']:.2f}")
print(f"  Heavy Weight: {optimal_config['heavy_weight']:.2f}")
print(f"  Accuracy: {optimal_config['accuracy']:.4f}")
print(f"  Intervention Rate: {optimal_config['intervention_rate']:.2f}%")

# %% Apply optimized routing on test set
print("\nApplying routing on test set...")
lite_preds_test = lite_model.predict([X_lite_test, X_tab_test], batch_size=config.BATCH_SIZE, verbose=0)
heavy_preds_test = heavy_model.predict([X_heavy_test, X_tab_test], batch_size=config.BATCH_SIZE, verbose=0)

final_preds_ham, route_mask_ham = routing.apply_threshold_routing(
    lite_preds_test, heavy_preds_test,
    entropy_threshold=optimal_config['entropy_t'],
    gap_threshold=optimal_config['gap_t'],
    heavy_weight=optimal_config['heavy_weight']
)

y_true_test = np.argmax(y_test, axis=1)
y_pred_ham = np.argmax(final_preds_ham, axis=1)

acc_ham = accuracy_score(y_true_test, y_pred_ham)
acc_lite = accuracy_score(y_true_test, np.argmax(lite_preds_test, axis=1))
acc_heavy = accuracy_score(y_true_test, np.argmax(heavy_preds_test, axis=1))

print(f"\nTest Set Results:")
print(f"  Lite Accuracy: {acc_lite:.4f}")
print(f"  Heavy Accuracy: {acc_heavy:.4f}")
print(f"  EcoFair Accuracy: {acc_ham:.4f}")
print(f"  Routing Rate: {route_mask_ham.sum() / len(route_mask_ham) * 100:.2f}%")

# Calculate entropy and safe-danger gap for test set (for routing breakdown)
entropy_test = routing.calculate_entropy(lite_preds_test)
prob_safe_test = lite_preds_test[:, safe_indices].sum(axis=1)
prob_danger_test = lite_preds_test[:, danger_indices].sum(axis=1)
safe_danger_gap_test = prob_safe_test - prob_danger_test

# %% Confusion Matrix Comparison
print("\nPlotting confusion matrices...")
fig_cm = visualization.plot_confusion_matrix_comparison(
    y_true_test, lite_preds_test, heavy_preds_test, final_preds_ham
)
plt.show()

# %% Per-Class Accuracy Bar Charts
print("\nPlotting per-class accuracy bar charts...")
fig_classwise = visualization.plot_classwise_accuracy_bars(
    y_true_test, lite_preds_test, heavy_preds_test, final_preds_ham
)
plt.show()

# %% Value-Added Analysis & Routing Breakdown (side by side)
print("\nPlotting value-added analysis and routing breakdown...")
fig_va_doughnut, axes_va_doughnut = plt.subplots(1, 2, figsize=(18, 7))
visualization.plot_value_added_bars(
    y_true_test, lite_preds_test, heavy_preds_test, final_preds_ham,
    route_mask=route_mask_ham, ax=axes_va_doughnut[0]
)
visualization.plot_routing_breakdown_doughnut(
    entropy_test, safe_danger_gap_test, route_mask_ham, len(route_mask_ham),
    ax=axes_va_doughnut[1]
)
plt.tight_layout()
plt.show()

# %% Comprehensive Performance Analysis: Gender, Age, Risk Groups, and Battery Decay
print("\nPlotting comprehensive performance analysis...")

# Gender and Age accuracy plots
fig_gender_age = visualization.plot_gender_age_accuracy(
    y_true_test, lite_preds_test, heavy_preds_test, final_preds_ham, meta_test
)
plt.show()

# Risk-stratified accuracy & Battery decay (side by side)
joules_per_lite = utils.load_energy_stats(config.SELECTED_LITE_MODEL, is_heavy=False)
joules_per_heavy = utils.load_energy_stats(config.SELECTED_HEAVY_MODEL, is_heavy=True)
if joules_per_lite is None:
    joules_per_lite = 1.0
if joules_per_heavy is None:
    joules_per_heavy = 2.5
routing_rate = route_mask_ham.sum() / len(route_mask_ham)

fig_risk_battery, axes_risk_battery = plt.subplots(1, 2, figsize=(20, 6))
visualization.plot_risk_stratified_accuracy(
    y_true_test, lite_preds_test, heavy_preds_test, final_preds_ham, meta_test,
    risk_scaler=risk_scaler, ax=axes_risk_battery[0]
)
visualization.plot_battery_decay(
    joules_per_lite, joules_per_heavy, routing_rate, capacity_joules=10000,
    ax=axes_risk_battery[1]
)
plt.tight_layout()
plt.show()

# %% Fairness Analysis
print("\nGenerating fairness analysis...")
ham_fairness = fairness.generate_fairness_report(
    y_true_test, y_pred_ham, meta_test
)

print("\n" + "="*80)
print("FAIRNESS REPORT - HAM10000 Dataset")
print("="*80)
print(ham_fairness.to_string(index=False))
print("="*80)
