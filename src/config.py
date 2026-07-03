"""
Configuration settings for the screen-recapture-detector project.
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Image processing config
IMAGE_SIZE = 224
BATCH_SIZE = 16  # Use smaller batch size due to CPU/GPU constraints of local environment

# Training config
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15
LEARNING_RATE_STAGE1 = 3e-4
LEARNING_RATE_STAGE2 = 1e-5

# Model config
MODEL_NAME = "efficientnet_v2_b0"
NUM_CLASSES = 1  # Binary classification (0 = Real, 1 = Screen)
DROPOUT_RATE_1 = 0.4  # First dropout in classifier head
DROPOUT_RATE_2 = 0.3  # Second dropout in classifier head
DENSE_UNITS = 128
UNFREEZE_OPTIONS = [40]  # Search space for number of backbone layers to unfreeze

# Checkpoint and results paths
CHECKPOINT_PATH = MODELS_DIR / "best_model.keras"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.pkl"

# Reproducibility
RANDOM_SEED = 42
SEED = RANDOM_SEED

