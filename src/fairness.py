"""
Fairness evaluation module for EcoFair project.

This module provides systematic fairness evaluation using a subgroup-specific,
One-vs-Rest evaluation strategy for every class across demographic subgroups.
"""

import numpy as np
import pandas as pd

def _ensure_1d_class_indices(y):
    """Convert one-hot or probability arrays to 1D class indices."""
    if len(np.asarray(y).shape) > 1:
        return np.argmax(y, axis=1)
    return np.asarray(y).flatten()


def calculate_equal_opportunity(tp, fn):
    """
    Equal Opportunity (True Positive Rate): TP / (TP + FN).
    
    Returns np.nan if denominator is 0 (no actual cases of this class in subgroup).
    """
    denom = tp + fn
    if denom == 0:
        return np.nan
    return tp / denom


def calculate_demographic_parity(tp, fp, tn, fn):
    """
    Demographic Parity (Positive Prediction Rate): (TP + FP) / (TP + TN + FP + FN).
    
    Returns np.nan if denominator is 0 (no data).
    """
    denom = tp + tn + fp + fn
    if denom == 0:
        return np.nan
    return (tp + fp) / denom


def calculate_subgroup_accuracy(tp, fp, tn, fn):
    """
    Subgroup Accuracy: (TP + TN) / (TP + TN + FP + FN).
    
    Returns np.nan if denominator is 0 (no data).
    """
    denom = tp + tn + fp + fn
    if denom == 0:
        return np.nan
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


def generate_fairness_report(y_true, y_pred, meta_df, class_names):
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
        class_names: List of class names (required).
    
    Returns:
        pandas.DataFrame: Fairness report with columns:
            ['Subgroup', 'Class', 'Count', 'Accuracy', 'Demographic_Parity_Rate', 'Equal_Opportunity_TPR']
    """
    if class_names is None:
        raise ValueError("class_names is required")
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


def print_fairness_audit(fairness_df):
    """
    Print fairness metrics as formatted pivot tables.
    
    Appends (n=Count) to Subgroup names and prints three pivot tables:
    Accuracy, Equal Opportunity (TPR), Demographic Parity.
    Uses '-' for missing data.
    
    Args:
        fairness_df: DataFrame from generate_fairness_report
    """
    required_cols = ['Subgroup', 'Class', 'Count', 'Accuracy', 'Equal_Opportunity_TPR', 'Demographic_Parity_Rate']
    if fairness_df.empty or not all(c in fairness_df.columns for c in required_cols):
        print("No fairness data (empty or missing columns). Raw report:")
        print(fairness_df)
        return
    
    df = fairness_df.copy()
    df['Subgroup'] = df.apply(lambda r: f"{r['Subgroup']} (n={int(r['Count'])})", axis=1)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print("\n--- Subgroup Accuracy ---")
    print(df.pivot(index='Subgroup', columns='Class', values='Accuracy').fillna('-'))
    
    print("\n--- Equal Opportunity (True Positive Rate) ---")
    print("Goal: TPR should be roughly equal across subgroups for the same class.")
    print("'-' indicates no actual cases of that class in the subgroup.")
    print(df.pivot(index='Subgroup', columns='Class', values='Equal_Opportunity_TPR').fillna('-'))
    
    print("\n--- Demographic Parity (Positive Prediction Rate) ---")
    print("Measures the raw rate at which a class is predicted for a specific subgroup.")
    print(df.pivot(index='Subgroup', columns='Class', values='Demographic_Parity_Rate').fillna('-'))
