"""
Routing decision logic for EcoFair project.

This module contains the core decision logic for the EcoFair pipeline, including
entropy calculation, threshold optimization, and budget-constrained routing.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from itertools import product

from . import config


def calculate_entropy(preds: np.ndarray):
    """
    Calculate entropy for probability predictions.
    
    Uses standard entropy formula: -sum(p * log(p))
    Clips probabilities to 1e-10 to avoid log(0).
    
    Args:
        preds: Numpy array of probability predictions, shape (n_samples, n_classes)
    
    Returns:
        numpy.ndarray: Entropy values for each sample, shape (n_samples,)
    """
    eps = 1e-10
    p = np.clip(preds, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


class SafetyFirstOptimizer:
    """
    Optimizer that finds thresholds meeting a minimum accuracy constraint.
    
    Used for HAM10000 dataset to find optimal routing thresholds that maintain
    accuracy while minimizing intervention rate.
    """
    
    def __init__(self, lite_preds, heavy_preds, y_true, entropy, safe_danger_gap, heavy_baseline_acc):
        """
        Initialize the optimizer.
        
        Args:
            lite_preds: Lite model predictions, shape (n_samples, n_classes)
            heavy_preds: Heavy model predictions, shape (n_samples, n_classes)
            y_true: True labels (integer array)
            entropy: Entropy values for each sample
            safe_danger_gap: Safe-danger gap values for each sample
            heavy_baseline_acc: Baseline accuracy of heavy model
        """
        self.lite_preds = lite_preds
        self.heavy_preds = heavy_preds
        self.y_true = y_true
        self.entropy = entropy
        self.safe_danger_gap = safe_danger_gap
        self.heavy_baseline_acc = heavy_baseline_acc
        self.min_acceptable_acc = heavy_baseline_acc - 0.015
        
    def evaluate_config(self, entropy_t, gap_t, heavy_weight):
        """
        Evaluate a specific threshold configuration.
        
        Args:
            entropy_t: Entropy threshold
            gap_t: Safe-danger gap threshold
            heavy_weight: Weight for heavy model in ensemble (0-1)
        
        Returns:
            dict: Configuration results with accuracy, intervention rate, and validity
        """
        route_to_heavy = (self.entropy > entropy_t) | (self.safe_danger_gap < gap_t)
        final_preds = self.lite_preds.copy()
        final_preds[route_to_heavy] = (1 - heavy_weight) * self.lite_preds[route_to_heavy] + heavy_weight * self.heavy_preds[route_to_heavy]
        
        y_pred = np.argmax(final_preds, axis=1)
        acc = accuracy_score(self.y_true, y_pred)
        intervention_rate = route_to_heavy.sum() / len(route_to_heavy) * 100
        
        return {
            'entropy_t': entropy_t,
            'gap_t': gap_t,
            'heavy_weight': heavy_weight,
            'accuracy': acc,
            'intervention_rate': intervention_rate,
            'valid': acc >= self.min_acceptable_acc
        }
    
    def optimize(self):
        """
        Optimize thresholds by searching parameter space.
        
        Searches over entropy thresholds, gap thresholds, and heavy weights.
        Returns the configuration with lowest intervention rate among valid configurations,
        or the best accuracy configuration if no valid configurations exist.
        
        Returns:
            tuple: (optimal_config, all_results)
                - optimal_config: Best configuration dict
                - all_results: List of all evaluated configurations
        """
        entropy_thresholds = [0.6, 0.7, 0.8, 0.9]
        gap_thresholds = [0.05, 0.10, 0.15]
        heavy_weights = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        results = []
        for ent_t, gap_t, hw in product(entropy_thresholds, gap_thresholds, heavy_weights):
            result = self.evaluate_config(ent_t, gap_t, hw)
            results.append(result)
        
        valid_results = [r for r in results if r['valid']]
        
        if valid_results:
            optimal = min(valid_results, key=lambda x: x['intervention_rate'])
            return optimal, results
        else:
            best = max(results, key=lambda x: x['accuracy'])
            return best, results


def apply_threshold_routing(lite_preds, heavy_preds, entropy_threshold=None, gap_threshold=None, heavy_weight=0.7, class_names=None, safe_classes=None, danger_classes=None):
    """
    Apply standard threshold-based routing (for HAM10000).
    
    Routes samples to heavy model if:
    - Entropy > entropy_threshold, OR
    - Safe-danger gap < gap_threshold
    
    Args:
        lite_preds: Lite model predictions, shape (n_samples, n_classes)
        heavy_preds: Heavy model predictions, shape (n_samples, n_classes)
        entropy_threshold: Entropy threshold (default: config.ENTROPY_THRESHOLD)
        gap_threshold: Safe-danger gap threshold (default: config.SAFE_DANGER_GAP_THRESHOLD)
        heavy_weight: Weight for heavy model in ensemble (default: 0.7)
        class_names: List of class names (default: config.CLASS_NAMES)
        safe_classes: List of safe class names (default: config.SAFE_CLASSES)
        danger_classes: List of dangerous class names (default: config.DANGEROUS_CLASSES)
    
    Returns:
        tuple: (final_preds, route_mask)
            - final_preds: Final predictions after routing, shape (n_samples, n_classes)
            - route_mask: Boolean array indicating which samples were routed to heavy model
    """
    if entropy_threshold is None:
        entropy_threshold = config.ENTROPY_THRESHOLD
    if gap_threshold is None:
        gap_threshold = config.SAFE_DANGER_GAP_THRESHOLD
    if class_names is None:
        class_names = config.CLASS_NAMES
    if safe_classes is None:
        safe_classes = config.SAFE_CLASSES
    if danger_classes is None:
        danger_classes = config.DANGEROUS_CLASSES
    
    # Calculate entropy
    entropy = calculate_entropy(lite_preds)
    
    # Calculate safe-danger gap
    safe_indices = [class_names.index(c) for c in safe_classes]
    danger_indices = [class_names.index(c) for c in danger_classes]
    prob_safe = lite_preds[:, safe_indices].sum(axis=1)
    prob_danger = lite_preds[:, danger_indices].sum(axis=1)
    safe_danger_gap = prob_safe - prob_danger
    
    # Create routing mask
    route_mask = (entropy > entropy_threshold) | (safe_danger_gap < gap_threshold)
    
    # Apply ensemble routing
    final_preds = lite_preds.copy()
    final_preds[route_mask] = (1 - heavy_weight) * lite_preds[route_mask] + heavy_weight * heavy_preds[route_mask]
    
    return final_preds, route_mask


def apply_budget_routing(lite_preds, heavy_preds, budget=0.35, heavy_weight=0.5, class_names=None, safe_classes=None, danger_classes=None):
    """
    Apply budget-constrained routing (for PAD-UFES-20).
    
    Routes the top K most uncertain samples to heavy model, where K = n_samples * budget.
    Uncertainty is measured by confusion_score = entropy + (1 - safe_danger_gap).
    
    Args:
        lite_preds: Lite model predictions, shape (n_samples, n_classes)
        heavy_preds: Heavy model predictions, shape (n_samples, n_classes)
        budget: Fraction of samples to route to heavy model (default: 0.35)
        heavy_weight: Weight for heavy model in ensemble (default: 0.5)
        class_names: List of class names (default: config.CLASS_NAMES)
        safe_classes: List of safe class names (default: config.SAFE_CLASSES)
        danger_classes: List of dangerous class names (default: config.DANGEROUS_CLASSES)
    
    Returns:
        tuple: (final_preds, route_mask, confusion_score)
            - final_preds: Final predictions after routing, shape (n_samples, n_classes)
            - route_mask: Boolean array indicating which samples were routed to heavy model
            - confusion_score: Confusion score for each sample, shape (n_samples,)
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    if safe_classes is None:
        safe_classes = config.SAFE_CLASSES
    if danger_classes is None:
        danger_classes = config.DANGEROUS_CLASSES
    
    # Calculate entropy
    entropy = calculate_entropy(lite_preds)
    
    # Calculate safe-danger gap
    safe_indices = [class_names.index(c) for c in safe_classes]
    danger_indices = [class_names.index(c) for c in danger_classes]
    prob_safe = lite_preds[:, safe_indices].sum(axis=1)
    prob_danger = lite_preds[:, danger_indices].sum(axis=1)
    safe_danger_gap = prob_safe - prob_danger
    
    # Calculate confusion_score = entropy + (1 - gap)
    # Note: gap can be negative, so (1 - gap) ensures higher confusion_score for more uncertain samples
    confusion_score = entropy + (1 - safe_danger_gap)
    
    # Sort samples by confusion_score (highest first = most uncertain)
    sorted_indices = np.argsort(confusion_score)[::-1]  # Descending order
    
    # Select Top-K indices where K = int(n_samples * budget)
    n_samples = len(confusion_score)
    top_n = int(n_samples * budget)
    top_indices = sorted_indices[:top_n]
    
    # Create route_mask: True for Top-K, False for others
    route_mask = np.zeros(n_samples, dtype=bool)
    route_mask[top_indices] = True
    
    # Apply ensemble routing
    final_preds = lite_preds.copy()
    final_preds[route_mask] = (1 - heavy_weight) * lite_preds[route_mask] + heavy_weight * heavy_preds[route_mask]
    
    return final_preds, route_mask, confusion_score
