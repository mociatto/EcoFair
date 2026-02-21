"""
Utility functions for EcoFair project.
"""

import json
import random
import numpy as np
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
    import os
    energy_path = os.path.join(energy_dir, 'energy_stats.json')
    if not os.path.exists(energy_path):
        return None
    try:
        with open(energy_path, 'r') as f:
            energy_stats = json.load(f)
        return energy_stats.get('joules_per_sample')
    except Exception:
        return None
