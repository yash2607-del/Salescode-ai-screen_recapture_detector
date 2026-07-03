"""
Evaluation wrapper script for screen-recapture-detector.
Runs the evaluation pipeline from the root directory.
"""
import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.evaluate import evaluate_pipeline
from src import config

if __name__ == "__main__":
    print("Initiating screen-recapture-detector evaluation...")
    evaluate_pipeline(
        model_path=config.CHECKPOINT_PATH,
        test_data_dir=config.DATASET_DIR / "test",
        results_dir=config.RESULTS_DIR
    )
