"""
Utility functions for EcoFair project.
"""

import json
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from . import config


def set_seed(seed: int = None) -> None:
    """Set random seeds for reproducibility."""
    if seed is None:
        seed = config.RANDOM_STATE
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_energy_stats(energy_dir: str):
    """
    Load energy statistics from a directory containing energy_stats.json.

    Args:
        energy_dir: Path to directory containing energy_stats.json

    Returns:
        float: Joules per sample, or None if not found
    """
    energy_path = os.path.join(energy_dir, 'energy_stats.json')
    if not os.path.exists(energy_path):
        return None
    try:
        with open(energy_path, 'r') as f:
            energy_stats = json.load(f)
        return energy_stats.get('joules_per_sample')
    except Exception:
        return None


def save_results_csv(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """
    Save a DataFrame to CSV in the output directory.

    Args:
        df: DataFrame to save
        output_dir: Directory path (created if missing)
        filename: CSV filename (e.g. 'cv_results.csv')

    Returns:
        str: Full path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path


def show_table(df: pd.DataFrame):
    """
    Display a DataFrame as a table (Jupyter display or print fallback).
    """
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string())
