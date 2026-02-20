"""
Configuration module for EcoFair project.

This module centralizes all global constants and configuration variables
to ensure the pipeline runs exactly as the original script.
"""

# Hyperparameters
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
ENTROPY_THRESHOLD = 0.65  # Initial default
SAFE_DANGER_GAP_THRESHOLD = 0.15  # Initial default

# HAM10000 Class Definitions
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
SAFE_CLASSES = ['nv', 'bkl', 'vasc', 'df']
DANGEROUS_CLASSES = ['mel', 'bcc', 'akiec']

# PAD-UFES-20 Class Definitions
PAD_CLASS_NAMES    = ['bcc', 'scc', 'mel', 'ack', 'nev', 'sek']
PAD_SAFE_CLASSES   = ['ack', 'nev', 'sek']
PAD_DANGEROUS_CLASSES = ['mel', 'bcc', 'scc']

# Risk Scoring - Localization Risk Scores
LOCALIZATION_RISK_SCORES = {
    'genital': 0.01,
    'acral': 0.01,
    'trunk': 0.04,
    'unknown': 0.09,
    'abdomen': 0.10,
    'foot': 0.11,
    'lower extremity': 0.18,
    'hand': 0.22,
    'back': 0.33,
    'upper extremity': 0.41,
    'chest': 0.45,
    'neck': 0.46,
    'ear': 0.56,
    'scalp': 0.58,
    'face': 0.74
}

# Paths
FEATURE_DIR = './output'
HEAVY_FEATURE_ROOT = '/kaggle/input/image-feature-extractor-heavy/output/HAM10000'
LITE_FEATURE_ROOT = '/kaggle/input/image-feature-extractor-lite/output/HAM10000'
DATA_PATH = '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv'

# Model Selection
SELECTED_HEAVY_MODEL = 'ResNet50'        # Options: EfficientNetB6, ResNet152V2, DenseNet201, InceptionResNetV2, ResNet50
SELECTED_LITE_MODEL = 'MobileNetV3Small'  # Options: MobileNetV3Small, MobileNetV3Large, EfficientNetB0, MobileNetV2, NASNetMobile

# Version
VERSION = "0.1.0"
