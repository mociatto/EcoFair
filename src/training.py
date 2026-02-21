"""
Training utilities for EcoFair project.

This module handles the training loop, class weighting logic,
and cross-validation pipeline orchestration.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

from . import config


def compile_and_train(model, X_img, X_tab, y, X_val_img, X_val_tab, y_val, class_weight=None):
    """
    Compile and train a VFL model.
    
    Args:
        model: Keras model to train
        X_img: Training image features
        X_tab: Training tabular features
        y: Training labels (one-hot encoded)
        X_val_img: Validation image features
        X_val_tab: Validation tabular features
        y_val: Validation labels (one-hot encoded)
        class_weight: Optional dictionary mapping class indices to weights
    
    Returns:
        History: Training history object
    """
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
    tqdm_callback = TqdmCallback(verbose=1, leave=False)
    hist = model.fit([X_img, X_tab], y,
                     validation_data=([X_val_img, X_val_tab], y_val),
                     epochs=config.EPOCHS,
                     batch_size=config.BATCH_SIZE,
                     callbacks=[es, tqdm_callback],
                     class_weight=class_weight,
                     verbose=0)
    return hist


def get_class_weights(y_train, class_names):
    """
    Calculate balanced class weights for training.
    
    Handles classes not present in the training split by setting their weight to 10.0.
    
    Args:
        y_train: Training labels, either one-hot encoded (2D array) or integer labels (1D array)
        class_names: List of class names (required).
    
    Returns:
        dict: Dictionary mapping class indices to weights
    """
    if class_names is None:
        raise ValueError("class_names is required")
    # Convert one-hot to labels if necessary
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        y_train_labels = y_train.flatten() if len(y_train.shape) > 1 else y_train
    
    unique_classes = np.unique(y_train_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_labels)
    
    class_weight_dict = {}
    for i, class_name in enumerate(class_names):
        if i in unique_classes:
            weight_idx = np.where(unique_classes == i)[0][0]
            class_weight_dict[i] = class_weights[weight_idx]
        else:
            class_weight_dict[i] = 10.0
    
    return class_weight_dict


def run_cv_pipeline(X_heavy, X_lite, X_tab, y, meta_df, class_names, safe_classes, dangerous_classes,
                    n_splits=5, risk_scaler=None, routing_strategy='threshold', budget=0.35,
                    energy_path_lite=None, energy_path_heavy=None):
    """
    Run 5-Fold Stratified Group Cross-Validation with Out-of-Fold (OOF) tracking.
    
    For each fold: train lite/heavy models, apply routing, compute metrics, store OOF predictions.
    
    Args:
        X_heavy: Heavy model features, shape (n_samples, feature_dim)
        X_lite: Lite model features, shape (n_samples, feature_dim)
        X_tab: Tabular features, shape (n_samples, tab_dim)
        y: Labels (one-hot), shape (n_samples, n_classes)
        meta_df: Metadata DataFrame (aligned with features)
        n_splits: Number of CV folds (default: 5)
        risk_scaler: Pre-fitted MinMaxScaler for risk scores. If None, uses fallback.
        routing_strategy: 'threshold' (SafetyFirstOptimizer, for HAM) or
                          'budget' (fixed 35% budget routing, for domain-shifted datasets)
        budget: Fraction of samples routed to heavy under budget routing (default: 0.35)
        class_names: Class names for routing (required).
        safe_classes: Safe class names (required).
        dangerous_classes: Dangerous class names (required).
        energy_path_lite: Path to directory containing energy_stats.json for lite model.
        energy_path_heavy: Path to directory containing energy_stats.json for heavy model.
    
    Returns:
        tuple: (fold_metrics, oof_lite, oof_heavy, oof_dynamic, route_mask_oof, route_components_oof)
            - fold_metrics: dict with lists for acc_lite, acc_heavy, acc_dynamic, routing_rate, energy_cost
            - oof_lite, oof_heavy, oof_dynamic: Out-of-fold probability predictions
            - route_mask_oof: Boolean array of samples routed to heavy (for visualizations)
            - route_components_oof: dict with uncertainty, ambiguity, safety masks
    """
    from . import data_loader, models, routing, features, utils
    
    if class_names is None or safe_classes is None or dangerous_classes is None:
        raise ValueError("class_names, safe_classes, and dangerous_classes are required")
    n_samples, n_classes = y.shape
    oof_lite = np.zeros_like(y, dtype=np.float32)
    oof_heavy = np.zeros_like(y, dtype=np.float32)
    oof_dynamic = np.zeros_like(y, dtype=np.float32)
    route_mask_oof = np.zeros(n_samples, dtype=bool)
    route_components_oof = {
        'uncertainty': np.zeros(n_samples, dtype=bool),
        'ambiguity': np.zeros(n_samples, dtype=bool),
        'safety': np.zeros(n_samples, dtype=bool),
    }
    
    fold_metrics = {
        'acc_lite': [], 'acc_heavy': [], 'acc_dynamic': [],
        'routing_rate': [], 'energy_cost': []
    }
    
    joules_lite = utils.load_energy_stats(energy_path_lite) if energy_path_lite else None
    joules_heavy = utils.load_energy_stats(energy_path_heavy) if energy_path_heavy else None
    joules_lite = joules_lite if joules_lite is not None else 1.0
    joules_heavy = joules_heavy if joules_heavy is not None else 2.5
    
    y_labels = np.argmax(y, axis=1)
    splits = data_loader.get_stratified_split(meta_df, y_labels, n_splits=n_splits)
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        meta_train = meta_df.iloc[train_idx].reset_index(drop=True)
        y_train_labels = y_labels[train_idx]
        splits_val = data_loader.get_stratified_split(meta_train, y_train_labels, n_splits=5)
        train_idx_inner, val_idx_inner = list(splits_val)[0]
        train_idx_abs = train_idx[train_idx_inner]
        val_idx_abs = train_idx[val_idx_inner]
        
        X_lite_train = X_lite[train_idx_abs]
        X_lite_val = X_lite[val_idx_abs]
        X_lite_test = X_lite[test_idx]
        X_heavy_train = X_heavy[train_idx_abs]
        X_heavy_val = X_heavy[val_idx_abs]
        X_heavy_test = X_heavy[test_idx]
        X_tab_train = X_tab[train_idx_abs]
        X_tab_val = X_tab[val_idx_abs]
        X_tab_test = X_tab[test_idx]
        y_train = y[train_idx_abs]
        y_val = y[val_idx_abs]
        meta_test = meta_df.iloc[test_idx].reset_index(drop=True)
        
        class_weight_dict = get_class_weights(y_train, class_names=class_names)
        
        lite_adapter = models.build_image_adapter(feature_dim=X_lite.shape[1], embedding_dim=128)
        heavy_adapter = models.build_image_adapter(feature_dim=X_heavy.shape[1], embedding_dim=128)
        tab_client = models.build_tabular_client(input_dim=X_tab.shape[1], embedding_dim=128)
        server_head = models.build_server_head(input_dim=256, num_classes=n_classes)
        
        lite_model = models.build_vfl_model(lite_adapter, tab_client, server_head)
        heavy_model = models.build_vfl_model(heavy_adapter, tab_client, server_head)
        
        compile_and_train(lite_model, X_lite_train, X_tab_train, y_train,
                         X_lite_val, X_tab_val, y_val, class_weight=class_weight_dict)
        compile_and_train(heavy_model, X_heavy_train, X_tab_train, y_train,
                         X_heavy_val, X_tab_val, y_val, class_weight=class_weight_dict)
        
        lite_preds_test = lite_model.predict([X_lite_test, X_tab_test],
                                             batch_size=config.BATCH_SIZE, verbose=0)
        heavy_preds_test = heavy_model.predict([X_heavy_test, X_tab_test],
                                               batch_size=config.BATCH_SIZE, verbose=0)
        
        y_true_test = y_labels[test_idx]
        
        if routing_strategy == 'budget':
            # Budget routing: force fixed 35% of most-uncertain samples to heavy model.
            # Used for domain-shifted datasets where threshold optimisation is unreliable.
            final_preds, route_mask, _ = routing.apply_budget_routing(
                lite_preds_test, heavy_preds_test,
                class_names, safe_classes, dangerous_classes,
                budget=budget,
            )
            # Budget routing has no per-reason decomposition; mark all routed as uncertainty
            route_components = {
                'uncertainty': route_mask.copy(),
                'ambiguity':   np.zeros(len(route_mask), dtype=bool),
                'safety':      np.zeros(len(route_mask), dtype=bool),
            }
        else:
            # Threshold routing with SafetyFirstOptimizer (HAM / source domain)
            entropy_test = routing.calculate_entropy(lite_preds_test)
            safe_indices   = [class_names.index(c) for c in safe_classes]
            danger_indices = [class_names.index(c) for c in dangerous_classes]
            prob_safe   = lite_preds_test[:, safe_indices].sum(axis=1)
            prob_danger = lite_preds_test[:, danger_indices].sum(axis=1)
            safe_danger_gap_test = prob_safe - prob_danger
            
            heavy_baseline_acc = accuracy_score(y_true_test, np.argmax(heavy_preds_test, axis=1))
            optimizer = routing.SafetyFirstOptimizer(
                lite_preds_test, heavy_preds_test, y_true_test,
                entropy_test, safe_danger_gap_test, heavy_baseline_acc
            )
            optimal_config, _ = optimizer.optimize()
            
            # Prefer the pre-computed risk_score column (always uses dataset-specific
            # localization scores). Recomputing here without localization_risk_scores
            # would silently fall back to HAM10000 scores (max 0.74), making the
            # scaled risk cap at 0.74 — just below the 0.75 safety threshold.
            if 'risk_score' in meta_test.columns:
                patient_risk = meta_test['risk_score'].values
            elif risk_scaler is not None:
                patient_risk = features.calculate_cumulative_risk(meta_test, risk_scaler)
            else:
                patient_risk = None
            
            final_preds, route_mask, route_components = routing.apply_threshold_routing(
                lite_preds_test, heavy_preds_test,
                class_names, safe_classes, dangerous_classes,
                entropy_threshold=optimal_config['entropy_t'],
                gap_threshold=optimal_config['gap_t'],
                heavy_weight=optimal_config['heavy_weight'],
                patient_risk=patient_risk,
                safety_threshold=0.75,
            )
        
        acc_lite = accuracy_score(y_true_test, np.argmax(lite_preds_test, axis=1))
        acc_heavy = accuracy_score(y_true_test, np.argmax(heavy_preds_test, axis=1))
        acc_dynamic = accuracy_score(y_true_test, np.argmax(final_preds, axis=1))
        routing_rate = route_mask.sum() / len(route_mask)
        n_routed = int(route_mask.sum())
        n_total = len(test_idx)
        total_energy = (n_total - n_routed) * joules_lite + n_routed * joules_heavy
        energy_per_sample = total_energy / n_total
        
        fold_metrics['acc_lite'].append(acc_lite)
        fold_metrics['acc_heavy'].append(acc_heavy)
        fold_metrics['acc_dynamic'].append(acc_dynamic)
        fold_metrics['routing_rate'].append(routing_rate)
        fold_metrics['energy_cost'].append(energy_per_sample)
        
        oof_lite[test_idx] = lite_preds_test
        oof_heavy[test_idx] = heavy_preds_test
        oof_dynamic[test_idx] = final_preds
        route_mask_oof[test_idx] = route_mask
        route_components_oof['uncertainty'][test_idx] = route_components['uncertainty']
        route_components_oof['ambiguity'][test_idx] = route_components['ambiguity']
        route_components_oof['safety'][test_idx] = route_components['safety']
        
        print(f"  Lite: {acc_lite:.4f} | Heavy: {acc_heavy:.4f} | EcoFair: {acc_dynamic:.4f} | "
              f"Route: {routing_rate*100:.1f}% | Energy: {energy_per_sample:.2f} J/sample")
        
        tf.keras.backend.clear_session()
    
    return fold_metrics, oof_lite, oof_heavy, oof_dynamic, route_mask_oof, route_components_oof
