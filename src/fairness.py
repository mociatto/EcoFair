"""
Fairness evaluation module for EcoFair project.

This module provides systematic fairness evaluation for the EcoFair pipeline,
calculating metrics like Demographic Parity and Equal Opportunity across
sensitive subgroups (Age and Sex).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import config


def calculate_demographic_parity(y_pred, sensitive_group_mask, positive_class_indices=None):
    """
    Calculate Demographic Parity (Positive Prediction Rate) for a subgroup.
    
    Demographic Parity measures if the Positive Prediction Rate (PPR) is similar
    across groups. PPR is the proportion of predictions that are positive.
    
    Args:
        y_pred: Model predictions (class indices), shape (n_samples,)
        sensitive_group_mask: Boolean array (True for the specific subgroup)
        positive_class_indices: List of class indices considered "positive" (malignant).
                                If None, uses config.DANGEROUS_CLASSES
    
    Returns:
        float: Positive Prediction Rate (PPR) for the subgroup
    """
    if positive_class_indices is None:
        positive_class_indices = [config.CLASS_NAMES.index(c) for c in config.DANGEROUS_CLASSES]
    
    # Convert to binary: is prediction positive (malignant)?
    y_pred_binary = np.isin(y_pred, positive_class_indices)
    
    # Calculate PPR for the subgroup
    if sensitive_group_mask.sum() == 0:
        return 0.0
    
    subgroup_ppr = y_pred_binary[sensitive_group_mask].mean()
    
    return subgroup_ppr


def calculate_equal_opportunity(y_true, y_pred, sensitive_group_mask, positive_class_indices=None):
    """
    Calculate Equal Opportunity (True Positive Rate/Recall) for a subgroup.
    
    Equal Opportunity measures if True Positive Rate (Recall/Sensitivity) is
    similar across groups. This is the proportion of actual positives that
    are correctly predicted as positive.
    
    Args:
        y_true: True labels (class indices), shape (n_samples,)
        y_pred: Model predictions (class indices), shape (n_samples,)
        sensitive_group_mask: Boolean array (True for the specific subgroup)
        positive_class_indices: List of class indices considered "positive" (malignant).
                                If None, uses config.DANGEROUS_CLASSES
    
    Returns:
        float: True Positive Rate (Recall) for the subgroup
    """
    if positive_class_indices is None:
        positive_class_indices = [config.CLASS_NAMES.index(c) for c in config.DANGEROUS_CLASSES]
    
    # Convert to binary: is label/prediction positive (malignant)?
    y_true_binary = np.isin(y_true, positive_class_indices)
    y_pred_binary = np.isin(y_pred, positive_class_indices)
    
    # Filter for Ground Truth Positives within the subgroup
    pos_mask = sensitive_group_mask & y_true_binary
    
    if pos_mask.sum() == 0:
        return 0.0  # No positive cases in subgroup
    
    # Calculate Recall: TP / (TP + FN) = TP / (all positives in subgroup)
    subgroup_recall = y_pred_binary[pos_mask].mean()
    
    return subgroup_recall


def generate_fairness_report(y_true, y_pred, meta_df, class_names=None):
    """
    Generate a comprehensive fairness report across demographic subgroups.
    
    Calculates accuracy, F1 score, demographic parity (positive rate), and
    equal opportunity (recall) for each subgroup defined by age and sex.
    
    Args:
        y_true: True labels (class indices), shape (n_samples,)
        y_pred: Model predictions (class indices), shape (n_samples,)
        meta_df: DataFrame with metadata columns ('age', 'sex' or 'gender')
        class_names: List of class names. If None, uses config.CLASS_NAMES
    
    Returns:
        pandas.DataFrame: Fairness report with columns:
            ['Subgroup', 'Count', 'Accuracy', 'F1_Score', 'Positive_Rate', 'Recall']
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    positive_class_indices = [class_names.index(c) for c in config.DANGEROUS_CLASSES]
    
    # Preprocessing: Extract and normalize age
    meta_df_copy = meta_df.copy()
    
    # Create age bins: <30, 30-60, 60+
    if 'age' in meta_df_copy.columns:
        meta_df_copy['age_bin'] = pd.cut(
            meta_df_copy['age'],
            bins=[0, 30, 60, 200],
            labels=['<30', '30-60', '60+'],
            include_lowest=True
        )
    else:
        meta_df_copy['age_bin'] = 'Unknown'
    
    # Extract and normalize sex/gender
    if 'sex' in meta_df_copy.columns:
        sex_col = 'sex'
    elif 'gender' in meta_df_copy.columns:
        sex_col = 'gender'
    else:
        sex_col = None
    
    if sex_col:
        # Normalize to 'Male', 'Female'
        meta_df_copy['sex_normalized'] = meta_df_copy[sex_col].astype(str).str.lower().str.strip()
        meta_df_copy['sex_normalized'] = meta_df_copy['sex_normalized'].map({
            'male': 'Male',
            'm': 'Male',
            'female': 'Female',
            'f': 'Female',
            'woman': 'Female',
            'man': 'Male'
        }).fillna('Other')
    else:
        meta_df_copy['sex_normalized'] = 'Unknown'
    
    # Initialize results list
    results = []
    
    # Define subgroups
    subgroups = []
    
    # Age subgroups
    if 'age_bin' in meta_df_copy.columns:
        for age_bin in ['<30', '30-60', '60+']:
            if age_bin in meta_df_copy['age_bin'].values:
                subgroups.append(('Age', age_bin))
    
    # Sex subgroups
    if 'sex_normalized' in meta_df_copy.columns:
        for sex in ['Male', 'Female']:
            if sex in meta_df_copy['sex_normalized'].values:
                subgroups.append(('Sex', sex))
    
    # Calculate metrics for each subgroup
    for subgroup_type, subgroup_value in subgroups:
        if subgroup_type == 'Age':
            mask = meta_df_copy['age_bin'] == subgroup_value
            subgroup_name = f"Age {subgroup_value}"
        else:  # Sex
            mask = meta_df_copy['sex_normalized'] == subgroup_value
            subgroup_name = f"Sex: {subgroup_value}"
        
        if mask.sum() == 0:
            continue
        
        # Count
        count = mask.sum()
        
        # Accuracy
        accuracy = accuracy_score(y_true[mask], y_pred[mask])
        
        # Macro F1 Score
        f1 = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
        
        # Demographic Parity (Positive Prediction Rate)
        # Calculate directly on filtered data
        y_pred_subgroup = y_pred[mask]
        y_pred_binary = np.isin(y_pred_subgroup, positive_class_indices)
        positive_rate = y_pred_binary.mean() if len(y_pred_binary) > 0 else 0.0
        
        # Equal Opportunity (Recall for Malignant classes)
        # Calculate directly on filtered data
        y_true_subgroup = y_true[mask]
        y_pred_subgroup = y_pred[mask]
        y_true_binary = np.isin(y_true_subgroup, positive_class_indices)
        y_pred_binary = np.isin(y_pred_subgroup, positive_class_indices)
        pos_mask = y_true_binary
        recall = y_pred_binary[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
        
        results.append({
            'Subgroup': subgroup_name,
            'Count': count,
            'Accuracy': accuracy,
            'F1_Score': f1,
            'Positive_Rate': positive_rate,
            'Recall': recall
        })
    
    # Create DataFrame
    fairness_df = pd.DataFrame(results)
    
    return fairness_df
