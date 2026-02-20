"""
Data loading and alignment module for EcoFair project.

This module handles loading raw feature arrays from Kaggle outputs,
aligning them with metadata CSVs (zipper logic), and managing data splits.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config


def load_client_features(model_name: str, is_heavy: bool = True, dataset_name: str = 'HAM10000'):
    """
    Load client features and IDs from Kaggle output directories.
    
    Args:
        model_name: Name of the model (e.g., 'ResNet50', 'MobileNetV3Small')
        is_heavy: Whether to load heavy model features (True) or lite model features (False)
        dataset_name: Dataset name ('HAM10000' or 'PAD-UFES-20')
    
    Returns:
        tuple: (features, ids) - numpy arrays
    
    Raises:
        FileNotFoundError: If features.npy or ids.npy are not found
    """
    # Determine base directory based on dataset name
    if dataset_name == 'HAM10000':
        base_dir = config.HEAVY_FEATURE_ROOT if is_heavy else config.LITE_FEATURE_ROOT
    elif dataset_name == 'PAD-UFES-20':
        base_dir = config.PAD_HEAVY_FEATURE_ROOT if is_heavy else config.PAD_LITE_FEATURE_ROOT
    else:
        # Generic: derive from HAM root by substituting dataset name
        ham_base = config.HEAVY_FEATURE_ROOT if is_heavy else config.LITE_FEATURE_ROOT
        base_dir = ham_base.replace('HAM10000', dataset_name)
    
    model_dir = os.path.join(base_dir, model_name)
    features_path = os.path.join(model_dir, 'features.npy')
    ids_path = os.path.join(model_dir, 'ids.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(ids_path):
        raise FileNotFoundError(f"Features or IDs not found for {model_name} in {base_dir}")
    
    features = np.load(features_path)
    ids = np.load(ids_path)
    
    return features, ids


def load_and_align_ham(metadata_path: str = None):
    """
    Align HAM10000 metadata with extracted features using zipper logic.
    
    Finds the intersection of IDs between metadata, heavy features, and lite features,
    then aligns all three to match exactly.
    
    Args:
        metadata_path: Path to HAM10000 metadata CSV. If None, uses config.DATA_PATH
    
    Returns:
        tuple: (X_heavy, X_lite, aligned_meta)
            - X_heavy: Heavy model features, shape (n_samples, feature_dim)
            - X_lite: Lite model features, shape (n_samples, feature_dim)
            - aligned_meta: DataFrame with aligned metadata
    """
    if metadata_path is None:
        metadata_path = config.DATA_PATH
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.dropna(subset=['image_id', 'dx', 'age', 'sex', 'localization'])
    
    # Clean image IDs (remove extensions, strip whitespace)
    metadata['image_id_clean'] = metadata['image_id'].astype(str).str.replace('.jpg', '', regex=False).str.strip()
    metadata_ids_set = set(metadata['image_id_clean'].values)
    
    # Load features
    heavy_features, heavy_ids = load_client_features(config.SELECTED_HEAVY_MODEL, is_heavy=True, dataset_name='HAM10000')
    lite_features, lite_ids = load_client_features(config.SELECTED_LITE_MODEL, is_heavy=False, dataset_name='HAM10000')
    
    # Clean feature IDs
    heavy_ids_clean = [str(id).replace('.jpg', '').strip() for id in heavy_ids]
    lite_ids_clean = [str(id).replace('.jpg', '').strip() for id in lite_ids]
    
    heavy_ids_set = set(heavy_ids_clean)
    lite_ids_set = set(lite_ids_clean)
    
    # Find intersection
    intersection = heavy_ids_set & lite_ids_set & metadata_ids_set
    
    if len(intersection) == 0:
        raise ValueError("No common IDs between metadata, heavy features, and lite features")
    
    # Create ID to index mappings
    heavy_id_to_idx = {img_id: idx for idx, img_id in enumerate(heavy_ids_clean)}
    lite_id_to_idx = {img_id: idx for idx, img_id in enumerate(lite_ids_clean)}
    
    # Sort common IDs for consistent ordering
    common_ids = sorted(list(intersection))
    
    # Align all three artifacts
    aligned_heavy = []
    aligned_lite = []
    aligned_meta_rows = []
    
    for img_id in common_ids:
        aligned_heavy.append(heavy_features[heavy_id_to_idx[img_id]])
        aligned_lite.append(lite_features[lite_id_to_idx[img_id]])
        meta_row = metadata[metadata['image_id_clean'] == img_id]
        if len(meta_row) > 0:
            aligned_meta_rows.append(meta_row.iloc[0])
    
    aligned_meta = pd.DataFrame(aligned_meta_rows).reset_index(drop=True)
    aligned_meta = aligned_meta.drop(columns=['image_id_clean'], errors='ignore')
    
    X_heavy = np.vstack(aligned_heavy)
    X_lite = np.vstack(aligned_lite)
    
    return X_heavy, X_lite, aligned_meta


def load_and_align_pad(metadata_path: str):
    """
    Align PAD-UFES-20 data with extracted features and fix localization mismatch.
    
    Maps PAD region names to HAM localization names so the risk scoring works correctly.
    Performs the same ID intersection/alignment logic as HAM.
    
    Args:
        metadata_path: Path to PAD-UFES-20 metadata CSV
    
    Returns:
        tuple: (X_heavy, X_lite, aligned_meta)
            - X_heavy: Heavy model features, shape (n_samples, feature_dim)
            - X_lite: Lite model features, shape (n_samples, feature_dim)
            - aligned_meta: DataFrame with aligned metadata (includes normalized 'localization' column)
    """
    # Load PAD metadata
    pad_metadata = pd.read_csv(metadata_path)
    
    # Handle img_id vs image_id column naming
    if 'img_id' in pad_metadata.columns:
        pad_metadata = pad_metadata.rename(columns={'img_id': 'image_id'})
    
    # Normalize image_id: remove extensions, strip whitespace
    pad_metadata['image_id_clean'] = pad_metadata['image_id'].astype(str).str.replace('.png', '', regex=False).str.replace('.jpg', '', regex=False).str.strip()
    pad_metadata_ids_set = set(pad_metadata['image_id_clean'].values)
    
    # CRITICAL MAP: Map PAD regions to HAM localizations for risk scoring
    pad_to_ham_localization_map = {
        'FOREARM': 'upper extremity',
        'HAND': 'upper extremity',
        'LOWER_LIMB': 'lower extremity',
        'THIGH': 'lower extremity',
        'FOOT': 'lower extremity'
    }
    
    # Apply mapping to create normalized localization column
    if 'region' in pad_metadata.columns:
        pad_metadata['localization'] = pad_metadata['region'].map(pad_to_ham_localization_map).fillna(pad_metadata['region'])
    elif 'lesion_location' in pad_metadata.columns:
        pad_metadata['localization'] = pad_metadata['lesion_location'].map(pad_to_ham_localization_map).fillna(pad_metadata['lesion_location'])
    else:
        print("Warning: Could not find 'region' or 'lesion_location' column. Risk scoring may not work correctly.")
        pad_metadata['localization'] = 'unknown'
    
    # Load features
    heavy_features, heavy_ids = load_client_features(config.SELECTED_HEAVY_MODEL, is_heavy=True, dataset_name='PAD-UFES-20')
    lite_features, lite_ids = load_client_features(config.SELECTED_LITE_MODEL, is_heavy=False, dataset_name='PAD-UFES-20')
    
    # Clean feature IDs
    heavy_ids_clean = [str(id).replace('.png', '').replace('.jpg', '').strip() for id in heavy_ids]
    lite_ids_clean = [str(id).replace('.png', '').replace('.jpg', '').strip() for id in lite_ids]
    
    heavy_ids_set = set(heavy_ids_clean)
    lite_ids_set = set(lite_ids_clean)
    
    # Find intersection
    intersection = heavy_ids_set & lite_ids_set & pad_metadata_ids_set
    
    if len(intersection) == 0:
        raise ValueError("No common IDs between PAD metadata, heavy features, and lite features")
    
    # Create ID to index mappings
    heavy_id_to_idx = {img_id: idx for idx, img_id in enumerate(heavy_ids_clean)}
    lite_id_to_idx = {img_id: idx for idx, img_id in enumerate(lite_ids_clean)}
    
    # Sort common IDs for consistent ordering
    common_ids = sorted(list(intersection))
    
    # Align all three artifacts
    aligned_heavy = []
    aligned_lite = []
    aligned_meta_rows = []
    
    for img_id in common_ids:
        aligned_heavy.append(heavy_features[heavy_id_to_idx[img_id]])
        aligned_lite.append(lite_features[lite_id_to_idx[img_id]])
        meta_row = pad_metadata[pad_metadata['image_id_clean'] == img_id]
        if len(meta_row) > 0:
            aligned_meta_rows.append(meta_row.iloc[0])
    
    aligned_meta = pd.DataFrame(aligned_meta_rows).reset_index(drop=True)
    aligned_meta = aligned_meta.drop(columns=['image_id_clean'], errors='ignore')
    
    X_heavy = np.vstack(aligned_heavy)
    X_lite = np.vstack(aligned_lite)
    
    return X_heavy, X_lite, aligned_meta


def get_stratified_split(meta_df: pd.DataFrame, y, n_splits: int = 5):
    """
    Create a reusable StratifiedGroupKFold splitter for data splitting.
    
    Determines the appropriate grouping column (lesion_id for HAM, patient_id for PAD,
    or image_id as fallback) to prevent data leakage.
    
    Args:
        meta_df: DataFrame with metadata (must contain grouping column)
        y: Target labels (can be one-hot encoded or integer labels)
        n_splits: Number of splits for cross-validation (default: 5)
    
    Returns:
        generator: Generator object from splitter.split(meta_df, y, groups)
    
    Example:
        >>> splits = get_stratified_split(meta_df, y_labels)
        >>> for train_idx, test_idx in splits:
        ...     # Process split
    """
    # Determine grouping column
    group_col = None
    for col in ['lesion_id', 'patient_id', 'img_id', 'image_id']:
        if col in meta_df.columns:
            group_col = col
            break
    
    if group_col is None:
        raise ValueError("Could not find grouping column. Expected one of: 'lesion_id', 'patient_id', 'img_id', 'image_id'")
    
    # Convert y to integer labels if one-hot encoded
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y.flatten() if len(y.shape) > 1 else y
    
    # Create splitter
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.RANDOM_STATE
    )
    
    # Return the generator from split()
    return splitter.split(meta_df, y_labels, groups=meta_df[group_col])
