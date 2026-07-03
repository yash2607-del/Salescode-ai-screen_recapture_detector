"""
Model architecture definition module using TensorFlow 2.x and Keras.
Prepares the EfficientNetV2B0 base model for transfer learning and fine-tuning.
"""
import tensorflow as tf
from typing import Tuple

def build_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate_1: float = 0.4,
    dropout_rate_2: float = 0.3,
    dense_units: int = 128
) -> tf.keras.Model:
    """
    Builds the model architecture with an EfficientNetV2B0 backbone
    and a custom classification head. The backbone is initially frozen.

    Args:
        input_shape (Tuple[int, int, int]): Shape of input images.
        dropout_rate_1 (float): Dropout rate before the dense layer.
        dropout_rate_2 (float): Dropout rate after the dense layer.
        dense_units (int): Number of units in the dense layer of the classification head.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    # Load the pretrained EfficientNetV2B0 model on ImageNet without top classifier layers
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    
    # Freeze the base model weights initially
    base_model.trainable = False

    # Define model inputs
    inputs = tf.keras.Input(shape=input_shape)

    # Note: EfficientNetV2 models handle preprocessing internally, but we can pass the scaled images directly.
    # We pass training=False to keep BatchNormalization layers in inference mode for the base model
    x = base_model(inputs, training=False)

    # Classifier head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate_1)(x)
    
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate_2)(x)
    
    # Sigmoid output for binary classification. Force float32 for mixed precision stability.
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs, name="screen_recapture_detector")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-4) -> tf.keras.Model:
    """
    Compiles the Keras model with Adam optimizer, binary crossentropy loss, and metrics.

    Args:
        model (tf.keras.Model): The model to compile.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        tf.keras.Model: The compiled model.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def unfreeze_backbone(model: tf.keras.Model, num_layers_to_unfreeze: int = 25) -> tf.keras.Model:
    """
    Unfreezes the last N layers of the EfficientNetV2B0 backbone for fine-tuning.
    Freeze all BatchNormalization layers to preserve accumulated mean/variance statistics.

    Args:
        model (tf.keras.Model): The Keras model containing the backbone.
        num_layers_to_unfreeze (int): Number of base model layers from the end to unfreeze.

    Returns:
        tf.keras.Model: The updated model.
    """
    # Extract the base model layer (usually the 2nd layer in the sequential architecture)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or (hasattr(layer, 'layers') and len(layer.layers) > 0):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("EfficientNetV2B0 base model layer not found in the model.")

    # Unfreeze base model
    base_model.trainable = True

    # Freeze everything except the last num_layers_to_unfreeze layers
    num_layers = len(base_model.layers)
    freeze_until = max(0, num_layers - num_layers_to_unfreeze)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # BatchNormalization trainable=True during fine-tuning with small datasets can disrupt model stability
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    print(f"Unfrozen the last {num_layers_to_unfreeze} layers of the base model backbone (excluding BatchNormalization).")
    return model
