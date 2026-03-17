"""
Model architecture definitions for EcoFair project.

This module defines the exact neural network components used in the
Vertical Federated Learning (VFL) architecture.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def build_image_adapter(feature_dim: int, embedding_dim: int = 128):
    """
    Build the image feature adapter component.
    
    Args:
        feature_dim: Dimension of input image features
        embedding_dim: Dimension of output embedding (default: 128)
    
    Returns:
        Keras Model: Image adapter model
    """
    inputs = layers.Input(shape=(feature_dim,), name='img_feat_input')
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(embedding_dim, activation='relu')(x)
    return models.Model(inputs, x, name='image_adapter')


def build_tabular_client(input_dim: int, embedding_dim: int = 128):
    """
    Build the tabular client component with risk-gated output.
    
    Crucial: The risk_gate (sigmoid) and gated_output logic is preserved exactly.
    The gate modulates the tabular embedding based on the risk score.
    
    Args:
        input_dim: Dimension of input tabular features (includes risk score as last feature)
        embedding_dim: Dimension of output embedding (default: 128)
    
    Returns:
        Keras Model: Tabular client model with risk gating
    """
    inputs = layers.Input(shape=(input_dim,), name='tabular_input')
    risk_input = layers.Lambda(lambda x: x[:, -1:], name='risk_extraction')(inputs)

    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    h_tab = layers.Dense(embedding_dim, activation='relu')(x)

    gate = layers.Dense(1, activation='sigmoid', name='risk_gate')(risk_input)
    gated_output = layers.Lambda(lambda t: t[0] * (1.0 + t[1]), name='gated_tabular')([h_tab, gate])

    return models.Model(inputs, gated_output, name='tabular_client')


def build_server_head(input_dim: int, num_classes: int):
    """
    Build the server head component for final classification.
    
    Args:
        input_dim: Dimension of fused input (image + tabular embeddings)
        num_classes: Number of output classes
    
    Returns:
        Keras Model: Server head model
    """
    inputs = layers.Input(shape=(input_dim,), name='fusion_input')
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return models.Model(inputs, outputs, name='server_head')


def build_vfl_model(img_adapter, tab_client, server_head):
    """
    Build the complete Vertical Federated Learning model.
    
    Combines image adapter, tabular client, and server head into a single model.
    
    Args:
        img_adapter: Image adapter model
        tab_client: Tabular client model
        server_head: Server head model
    
    Returns:
        Keras Model: Complete VFL model
    """
    img_in = layers.Input(shape=img_adapter.input_shape[1:], name='img_input')
    tab_in = layers.Input(shape=tab_client.input_shape[1:], name='tab_input')
    img_emb = img_adapter(img_in)
    tab_emb = tab_client(tab_in)
    fusion = layers.Concatenate()([img_emb, tab_emb])
    outputs = server_head(fusion)
    return models.Model([img_in, tab_in], outputs, name='vfl_model')
