"""
Fairness evaluation module for EcoFair project.

This module provides systematic fairness evaluation using a subgroup-specific,
One-vs-Rest evaluation strategy for every class across demographic subgroups.
"""

import numpy as np
import pandas as pd

from . import config


def _ensure_1d_class_indices(y):
    """Convert one-hot or probability arrays to 1D class indices."""
    if len(np.asarray(y).shape) > 1:
        return np.argmax(y, axis=1)
    return np.asarray(y).flatten()


def calculate_equal_opportunity(tp, fn):
    """
    Equal Opportunity (True Positive Rate): TP / (TP + FN).
    
    Returns 0 if denominator is 0.
    """
    denom = tp + fn
    if denom == 0:
        return 0.0
    return tp / denom


def calculate_demographic_parity(tp, fp, tn, fn):
    """
    Demographic Parity (Positive Prediction Rate): (TP + FP) / (TP + TN + FP + FN).
    
    Returns 0 if denominator is 0.
    """
    denom = tp + tn + fp + fn
    if denom == 0:
        return 0.0
    return (tp + fp) / denom


def calculate_subgroup_accuracy(tp, fp, tn, fn):
    """
    Subgroup Accuracy: (TP + TN) / (TP + TN + FP + FN).
    
    Returns 0 if denominator is 0.
    """
    denom = tp + tn + fp + fn
    if denom == 0:
        return 0.0
    return (tp + tn) / denom


def _compute_ovr_confusion(y_true, y_pred, mask, class_index):
    """
    One-vs-Rest confusion components for a given class index and subgroup mask.
    
    TP: Actual == class_index & Predicted == class_index
    FP: Actual != class_index & Predicted == class_index
    TN: Actual != class_index & Predicted != class_index
    FN: Actual == class_index & Predicted != class_index
    """
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    
    actual_pos = (y_true_m == class_index)
    actual_neg = ~actual_pos
    pred_pos  = (y_pred_m == class_index)
    pred_neg  = ~pred_pos
    
    tp = np.sum(actual_pos & pred_pos)
    fp = np.sum(actual_neg & pred_pos)
    tn = np.sum(actual_neg & pred_neg)
    fn = np.sum(actual_pos & pred_neg)
    
    return tp, fp, tn, fn


def generate_fairness_report(y_true, y_pred, meta_df, class_names=None):
    """
    Generate a comprehensive fairness report using One-vs-Rest per class.
    
    For each subgroup (Age, Sex) and each class, computes:
    - Accuracy (subgroup-specific)
    - Demographic Parity Rate (positive prediction rate for that class)
    - Equal Opportunity TPR (recall for that class)
    
    Args:
        y_true: True labels (class indices or one-hot), shape (n_samples,) or (n_samples, n_classes)
        y_pred: Predictions (class indices or one-hot), shape (n_samples,) or (n_samples, n_classes)
        meta_df: DataFrame with metadata columns ('age', 'sex' or 'gender')
        class_names: List of class names. If None, uses config.CLASS_NAMES
    
    Returns:
        pandas.DataFrame: Fairness report with columns:
            ['Subgroup', 'Class', 'Count', 'Accuracy', 'Demographic_Parity_Rate', 'Equal_Opportunity_TPR']
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    y_true = _ensure_1d_class_indices(y_true)
    y_pred = _ensure_1d_class_indices(y_pred)
    
    meta_df_copy = meta_df.copy()
    col_map = {c.lower(): c for c in meta_df_copy.columns}
    
    # Age subgroups: <30, 30-60, 60+
    age_col = col_map.get('age')
    if age_col is not None:
        age_vals = pd.to_numeric(meta_df_copy[age_col], errors='coerce').fillna(0).clip(lower=0)
        meta_df_copy['_age_bin'] = pd.cut(
            age_vals,
            bins=[0, 30, 60, 200],
            labels=['<30', '30-60', '60+'],
            include_lowest=True
        ).astype(str)
    else:
        meta_df_copy['_age_bin'] = 'Unknown'
    
    # Sex subgroups: Male, Female
    sex_col = col_map.get('sex') or col_map.get('gender')
    if sex_col is not None:
        s = meta_df_copy[sex_col].astype(str).str.lower().str.strip()
        meta_df_copy['_sex_norm'] = s.map({
            'male': 'Male', 'm': 'Male', 'man': 'Male',
            'female': 'Female', 'f': 'Female', 'woman': 'Female'
        }).fillna('Other')
    else:
        meta_df_copy['_sex_norm'] = 'Unknown'
    
    results = []
    
    # Build subgroup list
    subgroups = []
    for v in ['<30', '30-60', '60+']:
        mask = meta_df_copy['_age_bin'].astype(str) == v
        if mask.any():
            subgroups.append((f"Age {v}", mask))
    for v in ['Male', 'Female']:
        mask = meta_df_copy['_sex_norm'].astype(str) == v
        if mask.any():
            subgroups.append((f"Sex: {v}", mask))
    
    for subgroup_name, mask in subgroups:
        mask = mask.values if hasattr(mask, 'values') else mask
        if mask.sum() == 0:
            continue
        
        for class_index, class_name in enumerate(class_names):
            tp, fp, tn, fn = _compute_ovr_confusion(y_true, y_pred, mask, class_index)
            
            acc = calculate_subgroup_accuracy(tp, fp, tn, fn)
            dp  = calculate_demographic_parity(tp, fp, tn, fn)
            tpr = calculate_equal_opportunity(tp, fn)
            
            results.append({
                'Subgroup': subgroup_name,
                'Class': class_name,
                'Count': int(mask.sum()),
                'Accuracy': acc,
                'Demographic_Parity_Rate': dp,
                'Equal_Opportunity_TPR': tpr
            })
    
    columns = ['Subgroup', 'Class', 'Count', 'Accuracy', 'Demographic_Parity_Rate', 'Equal_Opportunity_TPR']
    if not results:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(results)
