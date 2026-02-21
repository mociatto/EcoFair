"""
Data loading and alignment module for EcoFair project.

This module handles loading raw feature arrays and aligning them with metadata
using flexible zipper logic. Dataset-agnostic: all paths and column handling
are driven by arguments from the front-end.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config


def load_dataset_features(heavy_dir: str, lite_dir: str, meta_path: str,
                          meta_preprocessor=None):
    """
    Load and align features with metadata using flexible zipper logic.

    Finds the intersection of IDs between metadata, heavy features, and lite
    features, then aligns all three to match exactly. Handles both image_id
    and img_id columns, and strips .jpg/.png extensions for matching.

    Args:
        heavy_dir: Path to directory containing features.npy and ids.npy
        lite_dir: Path to directory containing features.npy and ids.npy
        meta_path: Path to metadata CSV
        meta_preprocessor: Optional callable(DataFrame) -> DataFrame to apply
                           dataset-specific preprocessing before alignment.

    Returns:
        tuple: (X_heavy, X_lite, aligned_meta)
            - X_heavy: Heavy model features, shape (n_samples, feature_dim)
            - X_lite: Lite model features, shape (n_samples, feature_dim)
            - aligned_meta: DataFrame with aligned metadata
    """
    metadata = pd.read_csv(meta_path)

    # Normalize ID column
    if 'img_id' in metadata.columns and 'image_id' not in metadata.columns:
        metadata = metadata.rename(columns={'img_id': 'image_id'})
    if 'image_id' not in metadata.columns:
        raise ValueError("Metadata must contain 'image_id' or 'img_id' column")

    # Optional preprocessing (e.g. region->localization mapping)
    if meta_preprocessor is not None:
        metadata = meta_preprocessor(metadata)

    if 'sex' not in metadata.columns and 'gender' not in metadata.columns:
        metadata['sex'] = 'unknown'
    drop_cols = ['image_id']
    for c in ['dx', 'diagnostic', 'diagnosis', 'label']:
        if c in metadata.columns:
            drop_cols.append(c)
            break
    metadata = metadata.dropna(subset=drop_cols)

    # Clean image IDs (strip extensions)
    def clean_id(s):
        s = str(s).strip()
        for ext in ['.jpg', '.png', '.jpeg']:
            if s.lower().endswith(ext):
                s = s[:-len(ext)]
        return s.strip()

    metadata['image_id_clean'] = metadata['image_id'].astype(str).apply(clean_id)
    metadata_ids_set = set(metadata['image_id_clean'].values)

    # Load heavy features
    heavy_features_path = os.path.join(heavy_dir, 'features.npy')
    heavy_ids_path = os.path.join(heavy_dir, 'ids.npy')
    if not os.path.exists(heavy_features_path) or not os.path.exists(heavy_ids_path):
        raise FileNotFoundError(f"Features or IDs not found in {heavy_dir}")
    heavy_features = np.load(heavy_features_path)
    heavy_ids = np.load(heavy_ids_path)
    heavy_ids_clean = [clean_id(x) for x in heavy_ids]
    heavy_ids_set = set(heavy_ids_clean)

    # Load lite features
    lite_features_path = os.path.join(lite_dir, 'features.npy')
    lite_ids_path = os.path.join(lite_dir, 'ids.npy')
    if not os.path.exists(lite_features_path) or not os.path.exists(lite_ids_path):
        raise FileNotFoundError(f"Features or IDs not found in {lite_dir}")
    lite_features = np.load(lite_features_path)
    lite_ids = np.load(lite_ids_path)
    lite_ids_clean = [clean_id(x) for x in lite_ids]
    lite_ids_set = set(lite_ids_clean)

    # Intersection
    intersection = heavy_ids_set & lite_ids_set & metadata_ids_set
    if len(intersection) == 0:
        raise ValueError("No common IDs between metadata, heavy features, and lite features")

    # Align
    heavy_id_to_idx = {img_id: idx for idx, img_id in enumerate(heavy_ids_clean)}
    lite_id_to_idx = {img_id: idx for idx, img_id in enumerate(lite_ids_clean)}
    common_ids = sorted(intersection)

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


def get_stratified_split(meta_df: pd.DataFrame, y, n_splits: int = 5):
    """
    Create StratifiedGroupKFold splits for cross-validation.

    Uses lesion_id, patient_id, img_id, or image_id as grouping column
    to prevent data leakage.

    Args:
        meta_df: DataFrame with metadata (must contain grouping column)
        y: Target labels (one-hot or integer)
        n_splits: Number of folds (default: 5)

    Returns:
        generator: split(meta_df, y_labels, groups)
    """
    group_col = None
    for col in ['lesion_id', 'patient_id', 'img_id', 'image_id']:
        if col in meta_df.columns:
            group_col = col
            break
    if group_col is None:
        raise ValueError("Could not find grouping column. Expected one of: lesion_id, patient_id, img_id, image_id")

    if len(np.asarray(y).shape) > 1 and np.asarray(y).shape[1] > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = np.asarray(y).flatten()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.RANDOM_STATE
    )
    return splitter.split(meta_df, y_labels, groups=meta_df[group_col])
