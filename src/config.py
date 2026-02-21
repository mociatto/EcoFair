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

# Paths — HAM10000
FEATURE_DIR = './output'
HEAVY_FEATURE_ROOT = '/kaggle/input/image-feature-extractor-heavy/output/HAM10000'
LITE_FEATURE_ROOT  = '/kaggle/input/image-feature-extractor-lite/output/HAM10000'
DATA_PATH = '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv'

# Paths — PAD-UFES-20 (features come from a separate Kaggle notebook output)
PAD_DATA_PATH          = '/kaggle/input/datasets/mahdavi1202/skin-cancer/metadata.csv'
PAD_HEAVY_FEATURE_ROOT = '/kaggle/input/notebooks/mostafaanoosha/image-feature-extractor-heavy/output/PAD-UFES-20'
PAD_LITE_FEATURE_ROOT  = '/kaggle/input/notebooks/mostafaanoosha/image-feature-extractor-lite/output/PAD-UFES-20'

# Model Selection
SELECTED_HEAVY_MODEL = 'ResNet50'        # Options: EfficientNetB6, ResNet152V2, DenseNet201, InceptionResNetV2, ResNet50
SELECTED_LITE_MODEL = 'MobileNetV3Small'  # Options: MobileNetV3Small, MobileNetV3Large, EfficientNetB0, MobileNetV2, NASNetMobile

# Version
VERSION = "0.1.0"
