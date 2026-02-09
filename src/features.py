"""
Feature engineering and Neurosymbolic Risk Scoring for EcoFair project.

This module handles tabular feature engineering and label preparation,
with robust column name handling for different datasets (HAM vs PAD).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from tensorflow import keras

from . import config


def get_sun_exposure_score(localization: str) -> float:
    """
    Calculate sun exposure risk score based on body part localization.
    
    Args:
        localization: String representing body part/location
    
    Returns:
        float: Risk score from config.LOCALIZATION_RISK_SCORES, or 0.04 (default) if not found
    """
    if localization is None or not isinstance(localization, str):
        return 0.04
    loc_normalized = localization.strip().lower()
    return config.LOCALIZATION_RISK_SCORES.get(loc_normalized, 0.04)


def prepare_tabular_features(meta_df: pd.DataFrame):
    """
    Process raw metadata into numeric tabular features for the model.
    
    Handles column name variations between datasets:
    - age: age column (fills with 0 if missing)
    - sex/gender: Uses whichever exists
    - localization/region/lesion_location/anatom_site: Uses whichever exists
    
    Steps:
    1. Column standardization (handle missing/variant columns)
    2. Neurosymbolic Risk Scoring (sun_exposure × age, normalized)
    3. Encoding (StandardScaler for age, OneHotEncoder for sex and localization)
    4. Assembly (concatenate all features with risk_score as last column)
    
    Args:
        meta_df: DataFrame with metadata columns
    
    Returns:
        tuple: (tabular_features, scaler, sex_encoder, loc_encoder, risk_scaler)
            - tabular_features: numpy array of shape (n_samples, n_features)
            - scaler: StandardScaler fitted on age
            - sex_encoder: OneHotEncoder fitted on sex/gender
            - loc_encoder: OneHotEncoder fitted on localization
            - risk_scaler: MinMaxScaler fitted on raw risk scores
    """
    meta_df_copy = meta_df.copy()
    
    # Step A: Column Standardization
    # Handle age column
    if 'age' not in meta_df_copy.columns:
        meta_df_copy['age'] = 0
        print("Warning: 'age' column not found. Filling with 0.")
    age_values = meta_df_copy['age'].fillna(0).values
    
    # Handle sex/gender column
    gender_col = None
    if 'sex' in meta_df_copy.columns:
        gender_col = 'sex'
    elif 'gender' in meta_df_copy.columns:
        gender_col = 'gender'
    else:
        print("Warning: Neither 'sex' nor 'gender' column found. Using default values.")
        meta_df_copy['sex'] = 'unknown'
        gender_col = 'sex'
    
    # Handle localization column (check multiple possible names)
    loc_col_for_risk = None
    for col_name in ['localization', 'region', 'lesion_location', 'anatom_site']:
        if col_name in meta_df_copy.columns:
            loc_col_for_risk = col_name
            break
    
    # Try case-insensitive search if not found
    if loc_col_for_risk is None:
        for col in meta_df_copy.columns:
            col_lower = col.lower()
            if col_lower in ['region', 'lesion_location', 'localization', 'anatom_site']:
                loc_col_for_risk = col
                break
    
    if loc_col_for_risk is None:
        print("Warning: Could not find localization column. Using default risk score 0.04.")
        meta_df_copy['localization'] = 'unknown'
        loc_col_for_risk = 'localization'
    
    # Step B: Neurosymbolic Risk Scoring
    sun_exposure_rate = meta_df_copy[loc_col_for_risk].apply(get_sun_exposure_score).values
    raw_risk = sun_exposure_rate * age_values
    
    risk_scaler = MinMaxScaler()
    cumulative_risk = risk_scaler.fit_transform(raw_risk.reshape(-1, 1)).flatten()
    meta_df_copy['risk_score'] = cumulative_risk
    
    print(f"\nCumulative Exposure Risk Score Calculated:")
    print(f"- Raw risk range: [{raw_risk.min():.2f}, {raw_risk.max():.2f}]")
    print(f"- Normalized risk range: [{cumulative_risk.min():.4f}, {cumulative_risk.max():.4f}]")
    print(f"- Formula: risk_score = MinMax(sun_exposure_rate × age)")
    
    # Step C: Encoding
    # StandardScaler for age
    scaler = StandardScaler()
    age_scaled = scaler.fit_transform(meta_df_copy[['age']])
    
    # OneHotEncoder for sex/gender
    sex_encoder = OneHotEncoder(sparse_output=False, drop='first')
    sex_onehot = sex_encoder.fit_transform(meta_df_copy[[gender_col]])
    
    # OneHotEncoder for localization
    loc_encoder = OneHotEncoder(sparse_output=False, drop='first')
    loc_onehot = loc_encoder.fit_transform(meta_df_copy[[loc_col_for_risk]])
    
    # Step D: Assembly
    # Concatenate: [age_scaled, sex_onehot, loc_onehot, risk_score]
    tabular_features = np.concatenate([
        age_scaled,
        sex_onehot,
        loc_onehot,
        meta_df_copy[['risk_score']].values
    ], axis=1)
    
    return tabular_features, scaler, sex_encoder, loc_encoder, risk_scaler


def prepare_labels(meta_df: pd.DataFrame, class_names=None):
    """
    Convert diagnostic strings to one-hot encoded vectors.
    
    Handles column name variations: checks for 'dx', 'diagnosis', or 'diagnostic'.
    Normalizes diagnosis strings (lowercase, strip) before mapping.
    
    Args:
        meta_df: DataFrame with diagnosis column
        class_names: List of class names. If None, uses config.CLASS_NAMES (HAM10000)
    
    Returns:
        tuple: (y, dx_to_idx)
            - y: One-hot encoded labels, shape (n_samples, n_classes)
            - dx_to_idx: Dictionary mapping diagnosis strings to class indices
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    # Find diagnosis column
    diagnosis_col = None
    for col in ['dx', 'diagnosis', 'diagnostic', 'label']:
        if col in meta_df.columns:
            diagnosis_col = col
            break
    
    if diagnosis_col is None:
        raise ValueError("Could not find diagnosis column. Expected one of: 'dx', 'diagnosis', 'diagnostic', 'label'")
    
    # Normalize diagnosis strings
    meta_df_copy = meta_df.copy()
    meta_df_copy['diagnosis_normalized'] = meta_df_copy[diagnosis_col].astype(str).str.lower().str.strip()
    
    # Create mapping
    dx_to_idx = {dx: idx for idx, dx in enumerate(class_names)}
    
    # Map to indices
    labels = meta_df_copy['diagnosis_normalized'].map(dx_to_idx)
    
    # Handle unmapped diagnoses
    unmapped = labels.isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} samples with unmapped diagnoses. Mapping to first class (index 0)")
        labels = labels.fillna(0)
    
    labels = labels.astype(int).values
    y = keras.utils.to_categorical(labels, num_classes=len(class_names))
    
    return y, dx_to_idx


def calculate_cumulative_risk(meta_df: pd.DataFrame, risk_scaler: MinMaxScaler):
    """
    Re-calculate risk scores for test sets using a pre-fitted scaler.
    
    Uses the same formula as prepare_tabular_features but applies transform()
    instead of fit_transform().
    
    Args:
        meta_df: DataFrame with metadata columns (age and localization)
        risk_scaler: Pre-fitted MinMaxScaler from prepare_tabular_features
    
    Returns:
        numpy.ndarray: Normalized risk scores
    """
    # Find localization column (same logic as prepare_tabular_features)
    loc_col_for_risk = None
    for col_name in ['localization', 'region', 'lesion_location', 'anatom_site']:
        if col_name in meta_df.columns:
            loc_col_for_risk = col_name
            break
    
    if loc_col_for_risk is None:
        # Try case-insensitive search
        for col in meta_df.columns:
            col_lower = col.lower()
            if col_lower in ['region', 'lesion_location', 'localization', 'anatom_site']:
                loc_col_for_risk = col
                break
    
    if loc_col_for_risk is None:
        print("Warning: Could not find localization column. Using default risk score 0.04.")
        sun_exposure_rate = np.full(len(meta_df), 0.04)
    else:
        sun_exposure_rate = meta_df[loc_col_for_risk].apply(get_sun_exposure_score).values
    
    # Get age values (fill with 0 if missing)
    age_values = meta_df['age'].values if 'age' in meta_df.columns else np.zeros(len(meta_df))
    
    # Calculate raw risk and transform
    raw_risk = sun_exposure_rate * age_values
    cumulative_risk = risk_scaler.transform(raw_risk.reshape(-1, 1)).flatten()
    
    return cumulative_risk
