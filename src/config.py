"""
Configuration module for EcoFair project.

Contains only global hyperparameters. All dataset-specific configuration
(paths, class names, safe/danger lists) is defined in front-end scripts.
"""

# Hyperparameters
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
ENTROPY_THRESHOLD = 0.65
SAFE_DANGER_GAP_THRESHOLD = 0.15

# Model pairs: (lite_model, heavy_model) for multi-pair pipeline
# Front scripts resolve these names to actual feature directories.
MODEL_PAIRS = [
    {'lite': 'MobileNetV2', 'heavy': 'ResNet50'},
    {'lite': 'MobileNetV3Small', 'heavy': 'DenseNet201'},
    {'lite': 'MobileNetV3Small', 'heavy': 'EfficientNetB6'},
]

# Version
VERSION = "0.1.0"
