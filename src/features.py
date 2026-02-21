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


def compute_localization_risk_scores(meta_df: pd.DataFrame, dangerous_classes=None) -> dict:
    """
    Empirically derive localization risk scores from a dataset's malignancy rates.
    
    For each localization site, computes the raw fraction of samples that belong to a
    dangerous (malignant) class.  Values are already in [0, 1] as proportions, e.g.
    scalp = 0.06 means 6 % of scalp lesions in that dataset are malignant.
    No min-max rescaling is applied so scores are directly comparable across datasets.
    
    Args:
        meta_df: DataFrame with diagnosis and localization columns
        dangerous_classes: List of class names considered malignant.
                           Defaults to config.DANGEROUS_CLASSES if None.
    
    Returns:
        dict: {localization_site (lowercase): raw_malignancy_rate}
              Returns {} if the required columns cannot be found.
    """
    if dangerous_classes is None:
        dangerous_classes = config.DANGEROUS_CLASSES
    
    df = meta_df.copy()
    
    # Detect diagnosis column
    dx_col = None
    for col in ['dx', 'diagnostic', 'diagnosis', 'label']:
        if col in df.columns:
            dx_col = col
            break
    
    # Detect localization column
    loc_col = None
    for col in ['localization', 'region', 'lesion_location', 'anatom_site']:
        if col in df.columns:
            loc_col = col
            break
    if loc_col is None:
        for col in df.columns:
            if col.lower() in ['localization', 'region', 'lesion_location', 'anatom_site']:
                loc_col = col
                break
    
    if dx_col is None or loc_col is None:
        print("Warning: Cannot compute localization risk scores — diagnosis or localization column missing.")
        return {}
    
    dangerous_set = {c.lower().strip() for c in dangerous_classes}
    df['_is_malignant'] = df[dx_col].astype(str).str.lower().str.strip().isin(dangerous_set).astype(float)
    df['_loc'] = df[loc_col].astype(str).str.lower().str.strip()
    
    # Raw malignancy rate per site: fraction of samples at that site that are dangerous.
    # e.g. if 6% of all samples at 'scalp' carry a dangerous diagnosis → score = 0.06
    rates = df.groupby('_loc')['_is_malignant'].mean().round(4)
    
    return rates.to_dict()


def get_sun_exposure_score(localization: str, risk_scores: dict = None) -> float:
    """
    Look up the raw malignancy rate for a body-part localization site.
    
    Args:
        localization: String representing body part/location
        risk_scores: Dict {site: raw_malignancy_rate} from compute_localization_risk_scores.
                     When None or empty, returns 0.05 as a neutral fallback.
    
    Returns:
        float: Raw malignancy rate for the site, or the median of the provided dict
               if the site is not found, or 0.05 if no dict is provided.
    """
    if not risk_scores:
        return 0.05
    if localization is None or not isinstance(localization, str):
        return float(np.median(list(risk_scores.values())))
    loc_normalized = localization.strip().lower()
    default = float(np.median(list(risk_scores.values())))
    return risk_scores.get(loc_normalized, default)


def prepare_tabular_features(meta_df: pd.DataFrame, localization_risk_scores: dict = None):
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
        localization_risk_scores: Optional dict {site: raw_malignancy_rate} from
                                   compute_localization_risk_scores. Falls back to median=0.05 if None.
    
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
    sun_exposure_rate = meta_df_copy[loc_col_for_risk].apply(
        lambda loc: get_sun_exposure_score(loc, localization_risk_scores)
    ).values
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


def calculate_cumulative_risk(meta_df: pd.DataFrame, risk_scaler: MinMaxScaler,
                              localization_risk_scores: dict = None):
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
        sun_exposure_rate = meta_df[loc_col_for_risk].apply(
            lambda loc: get_sun_exposure_score(loc, localization_risk_scores)
        ).values
    
    # Get age values (fill with 0 if missing)
    age_values = meta_df['age'].values if 'age' in meta_df.columns else np.zeros(len(meta_df))
    
    # Calculate raw risk and transform
    raw_risk = sun_exposure_rate * age_values
    cumulative_risk = risk_scaler.transform(raw_risk.reshape(-1, 1)).flatten()
    
    return cumulative_risk
