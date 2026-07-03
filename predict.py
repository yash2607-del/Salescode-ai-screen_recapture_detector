"""
Main prediction script for screen-recapture-detector.
Supports single image prediction and batch folder prediction.
Optimized for quiet stdout outputs during single inference.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Suppress TensorFlow logs and warnings before loading TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import joblib

# Disable experimental warnings and standard output alerts from TF
tf.get_logger().setLevel("ERROR")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Import configurations
from src import config

# Standard supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_recommended_threshold() -> float:
    """
    Attempts to load the recommended threshold from metrics.json.
    Defaults to 0.50 if not found.
    """
    metrics_path = config.RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                return float(metrics.get("recommended_threshold", 0.50))
        except Exception:
            pass
    return 0.50


def preprocess_image(image_path: Path, image_size: int = 224) -> tf.Tensor:
    """
    Validates, loads, and preprocesses an image for inference.

    Args:
        image_path (Path): Path to the image file.
        image_size (int): Dimensions to resize the image.

    Returns:
        tf.Tensor: Batched preprocessed image tensor.
    """
    # 1. Path validation
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Error: Path '{image_path}' does not exist or is not a file.")

    # 2. Extension validation
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Error: Unsupported image format '{image_path.suffix}'. Supported: {SUPPORTED_EXTENSIONS}")

    # 3. Integrity verification (Pillow)
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
    except Exception as e:
        raise IOError(f"Error: Corrupted image file '{image_path.name}'. Details: {e}")

    # 4. TensorFlow Preprocessing (equivalent to training pipeline)
    img_bytes = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension [1, H, W, C]
    return img


def predict_single(model: tf.keras.Model, image_path: Path, calibrator=None) -> float:
    """Runs prediction on a single image and returns the probability."""
    preprocessed = preprocess_image(image_path, config.IMAGE_SIZE)
    # Predict without console logs
    prob = model(preprocessed, training=False).numpy().flatten()[0]
    
    if calibrator is not None:
        # LogisticRegression expects 2D array
        prob = calibrator.predict_proba([[prob]])[0, 1]
        
    return float(prob)


def predict_batch(model: tf.keras.Model, folder_path: Path, output_csv: Path, calibrator=None) -> None:
    """
    Runs prediction on all supported images in a folder and saves results to CSV.
    """
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    threshold = load_recommended_threshold()
    print(f"Loaded recommended decision threshold: {threshold:.2f}")

    # Collect valid images
    image_paths = []
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(file_path)

    if not image_paths:
        print(f"No valid images found in directory: {folder_path}")
        return

    print(f"Found {len(image_paths)} images. Processing batch predictions...")

    records = []
    # Loop over and predict (loading images dynamically)
    for path in image_paths:
        try:
            prob = predict_single(model, path, calibrator)
            prediction_label = "SCREEN" if prob >= threshold else "REAL"
            records.append({
                "filename": path.name,
                "probability": round(prob, 4),
                "prediction": prediction_label
            })
        except Exception as e:
            print(f"Warning: Skipping '{path.name}' due to error: {e}", file=sys.stderr)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved batch predictions to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Predict if an image is a screen recapture or a real photo.")
    parser.add_argument("input_path", type=str, help="Path to a single image file or a directory of images.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    model_path = config.CHECKPOINT_PATH

    if not model_path.exists():
        print(f"Error: Trained model checkpoint not found at '{model_path}'. Please run training first.", file=sys.stderr)
        sys.exit(1)

    # Load Keras Model once
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load calibrator if available
    calibrator = None
    if getattr(config, "CALIBRATOR_PATH", None) and config.CALIBRATOR_PATH.exists():
        try:
            calibrator = joblib.load(config.CALIBRATOR_PATH)
        except Exception as e:
            print(f"Warning: Could not load calibrator at {config.CALIBRATOR_PATH}: {e}", file=sys.stderr)

    if input_path.is_file():
        # Single image prediction: Print ONLY the formatted probability to stdout
        try:
            prob = predict_single(model, input_path, calibrator)
            print(f"{prob:.4f}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif input_path.is_dir():
        # Batch folder prediction
        output_csv = Path("predictions.csv")
        predict_batch(model, input_path, output_csv, calibrator)
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
