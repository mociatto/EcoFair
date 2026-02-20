"""
Utility functions for EcoFair project.

This module provides generic helper functions for environment setup
and loading generic statistics.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf

from . import config


def set_seed(seed: int = None) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python's random, NumPy, and TensorFlow.
    
    Args:
        seed: Random seed value. If None, uses config.RANDOM_STATE.
    """
    if seed is None:
        seed = config.RANDOM_STATE
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_energy_stats(model_name: str, is_heavy: bool = True, dataset_name: str = 'HAM10000'):
    """
    Load energy statistics for a given model and dataset.
    
    Args:
        model_name: Name of the model (e.g., 'ResNet50', 'MobileNetV3Small')
        is_heavy: Whether the model is heavy (True) or lite (False)
        dataset_name: Dataset to load stats for ('HAM10000', 'PAD-UFES-20', etc.)
    
    Returns:
        joules_per_sample: Energy consumption in Joules per sample, or None if not found
    """
    if dataset_name == 'PAD-UFES-20':
        base_dir = config.PAD_HEAVY_FEATURE_ROOT if is_heavy else config.PAD_LITE_FEATURE_ROOT
    else:
        base_dir = config.HEAVY_FEATURE_ROOT if is_heavy else config.LITE_FEATURE_ROOT
        # For datasets other than HAM/PAD, substitute dataset name in the HAM root path
        if dataset_name != 'HAM10000':
            base_dir = base_dir.replace('HAM10000', dataset_name)
    energy_path = os.path.join(base_dir, model_name, 'energy_stats.json')
    
    if not os.path.exists(energy_path):
        print(f"  Warning: Energy stats not found at {energy_path}")
        return None
    
    try:
        with open(energy_path, 'r') as f:
            energy_stats = json.load(f)
        joules_per_sample = energy_stats.get('joules_per_sample')
        if joules_per_sample is None:
            print(f"  Warning: joules_per_sample not found in {energy_path}")
            return None
        return joules_per_sample
    except Exception as e:
        print(f"  Warning: Failed to load energy stats from {energy_path}: {e}")
        return None
