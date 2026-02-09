"""
Training utilities for EcoFair project.

This module handles the training loop and class weighting logic.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback
from sklearn.utils.class_weight import compute_class_weight

from . import config


def compile_and_train(model, X_img, X_tab, y, X_val_img, X_val_tab, y_val, class_weight=None):
    """
    Compile and train a VFL model.
    
    Args:
        model: Keras model to train
        X_img: Training image features
        X_tab: Training tabular features
        y: Training labels (one-hot encoded)
        X_val_img: Validation image features
        X_val_tab: Validation tabular features
        y_val: Validation labels (one-hot encoded)
        class_weight: Optional dictionary mapping class indices to weights
    
    Returns:
        History: Training history object
    """
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
    tqdm_callback = TqdmCallback(verbose=1, leave=False)
    hist = model.fit([X_img, X_tab], y,
                     validation_data=([X_val_img, X_val_tab], y_val),
                     epochs=config.EPOCHS,
                     batch_size=config.BATCH_SIZE,
                     callbacks=[es, tqdm_callback],
                     class_weight=class_weight,
                     verbose=0)
    return hist


def get_class_weights(y_train, class_names=None):
    """
    Calculate balanced class weights for training.
    
    Handles classes not present in the training split by setting their weight to 10.0.
    
    Args:
        y_train: Training labels, either one-hot encoded (2D array) or integer labels (1D array)
        class_names: List of class names. If None, uses config.CLASS_NAMES
    
    Returns:
        dict: Dictionary mapping class indices to weights
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    # Convert one-hot to labels if necessary
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        y_train_labels = y_train.flatten() if len(y_train.shape) > 1 else y_train
    
    unique_classes = np.unique(y_train_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_labels)
    
    class_weight_dict = {}
    for i, class_name in enumerate(class_names):
        if i in unique_classes:
            weight_idx = np.where(unique_classes == i)[0][0]
            class_weight_dict[i] = class_weights[weight_idx]
        else:
            class_weight_dict[i] = 10.0
    
    return class_weight_dict
