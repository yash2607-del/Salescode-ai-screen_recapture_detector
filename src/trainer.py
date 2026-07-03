"""
Training loop and pipeline module using TensorFlow 2.x and Keras.
Manages two-stage training (backbone frozen vs. unfrozen), history tracking,
mixed precision, data loading, and metric visualizations.
"""
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
import joblib

# Add parent directory to path to enable execution from root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset import ScreenRecaptureDataset
from src.model import build_model, compile_model, unfreeze_backbone


def set_reproducibility(seed: int = 42) -> None:
    """
    Sets global seeds for reproducibility across Python, NumPy, and TensorFlow.

    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Configure deterministic execution if supported by the runtime
    tf.config.experimental.enable_op_determinism()
    print(f"Reproducibility seeds set to {seed}.")


def check_and_enable_mixed_precision() -> bool:
    """
    Detects GPU and enables mixed precision training (mixed_float16) if available.

    Returns:
        bool: True if mixed precision was successfully enabled, False otherwise.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Set mixed precision global policy
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled (mixed_float16) on GPU.")
            return True
        except Exception as e:
            print(f"Failed to enable mixed precision: {e}")
    else:
        print("No GPU detected. Running in standard float32 precision mode on CPU.")
    return False


def plot_metrics(history_csv_path: Path, output_dir: Path) -> None:
    """
    Reads the training log CSV and generates plots for Loss, Accuracy, AUC, and LR.

    Args:
        history_csv_path (Path): Path to the merged CSV history file.
        output_dir (Path): Output directory where figures will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(history_csv_path)
    epochs = df["epoch"] + 1  # Convert to 1-based indexing for plots

    # Plot specifications
    metrics_to_plot = [
        ("loss", "Loss", "loss.png", "royalblue", "darkorange"),
        ("accuracy", "Accuracy", "accuracy.png", "forestgreen", "crimson"),
        ("auc", "AUC", "auc.png", "darkorchid", "gold"),
    ]

    for metric, title, filename, train_color, val_color in metrics_to_plot:
        if metric in df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, df[metric], label=f"Train {title}", color=train_color, linewidth=2)
            val_metric = f"val_{metric}"
            if val_metric in df.columns:
                plt.plot(epochs, df[val_metric], label=f"Val {title}", color=val_color, linewidth=2)
            plt.title(f"Model {title} Curve")
            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=150)
            plt.close()
            print(f"Generated metric plot: {output_dir / filename}")

    # Plot learning rate
    if "lr" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["lr"], label="Learning Rate", color="darkcyan", linewidth=2)
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "learning_rate.png", dpi=150)
        plt.close()
        print(f"Generated learning rate plot: {output_dir / 'learning_rate.png'}")


class Trainer:
    """
    OOP Trainer class to manage dataset initialization, model compilation,
    two-stage training progression, logging, and evaluation.
    """
    def __init__(self, config_module=config):
        self.config = config_module
        set_reproducibility(self.config.RANDOM_SEED)
        self.mixed_precision_active = check_and_enable_mixed_precision()

        # Placeholders
        self.train_ds: Optional[tf.data.Dataset] = None
        self.val_ds: Optional[tf.data.Dataset] = None
        self.test_ds: Optional[tf.data.Dataset] = None
        self.model: Optional[tf.keras.Model] = None
        self.train_labels: Optional[list] = None
        self.val_labels: Optional[list] = None

    def load_datasets(self) -> None:
        """
        Loads train, val, and test datasets using the ScreenRecaptureDataset directory wrappers.
        """
        print("Loading datasets from disk splits...")
        
        train_dataset_wrapper = ScreenRecaptureDataset(
            data_dir=self.config.DATASET_DIR / "train",
            image_size=self.config.IMAGE_SIZE,
            seed=self.config.RANDOM_SEED
        )
        self.train_ds = train_dataset_wrapper.get_dataset(
            batch_size=self.config.BATCH_SIZE,
            is_training=True,
            shuffle=True
        )
        self.train_labels = train_dataset_wrapper.labels

        val_dataset_wrapper = ScreenRecaptureDataset(
            data_dir=self.config.DATASET_DIR / "val",
            image_size=self.config.IMAGE_SIZE,
            seed=self.config.RANDOM_SEED
        )
        self.val_ds = val_dataset_wrapper.get_dataset(
            batch_size=self.config.BATCH_SIZE,
            is_training=False,
            shuffle=False
        )
        self.val_labels = val_dataset_wrapper.labels

        test_dataset_wrapper = ScreenRecaptureDataset(
            data_dir=self.config.DATASET_DIR / "test",
            image_size=self.config.IMAGE_SIZE,
            seed=self.config.RANDOM_SEED
        )
        self.test_ds = test_dataset_wrapper.get_dataset(
            batch_size=self.config.BATCH_SIZE,
            is_training=False,
            shuffle=False
        )

        print("Datasets loaded successfully.")

    def run_pipeline(self) -> None:
        """
        Runs the complete two-stage transfer learning training loop.
        """
        if self.train_ds is None or self.val_ds is None:
            self.load_datasets()

        self.config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Build Model
        print("Building model architecture...")
        self.model = build_model(
            input_shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3),
            dropout_rate_1=self.config.DROPOUT_RATE_1,
            dropout_rate_2=self.config.DROPOUT_RATE_2,
            dense_units=self.config.DENSE_UNITS
        )

        # Compute class weights
        classes = np.unique(self.train_labels)
        cw = compute_class_weight('balanced', classes=classes, y=self.train_labels)
        class_weight_dict = {cls: weight for cls, weight in zip(classes, cw)}
        print(f"Computed class weights: {class_weight_dict}")

        # ==========================================
        # STAGE 1: Train Classifier Head Only
        # ==========================================
        print("\n" + "="*50)
        print("STAGE 1: Training classifier head (backbone frozen)")
        print("="*50)
        
        self.model = compile_model(self.model, learning_rate=self.config.LEARNING_RATE_STAGE1)
        
        stage1_checkpoint = self.config.MODELS_DIR / "stage1_best.keras"
        stage1_csv = self.config.RESULTS_DIR / "history_stage1.csv"

        callbacks_stage1 = [
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=str(stage1_checkpoint), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.CSVLogger(filename=str(stage1_csv), append=False)
        ]

        history_s1 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config.EPOCHS_STAGE1,
            class_weight=class_weight_dict,
            callbacks=callbacks_stage1
        )

        # ==========================================
        # STAGE 2: Fine-Tuning Backbone
        # ==========================================
        print("\n" + "="*50)
        print(f"STAGE 2: Fine-tuning unfreezing options {self.config.UNFREEZE_OPTIONS} layers")
        print("="*50)

        best_val_auc = -1.0
        best_history_s2 = None
        stage2_checkpoint = self.config.MODELS_DIR / "stage2_best.keras"
        stage2_csv = self.config.RESULTS_DIR / "history_stage2.csv"

        for unfreeze_layers in self.config.UNFREEZE_OPTIONS:
            print(f"\nEvaluating fine-tuning with {unfreeze_layers} layers unfrozen...")
            
            # Reset to best Stage 1 weights
            if stage1_checkpoint.exists():
                self.model = tf.keras.models.load_model(str(stage1_checkpoint))
            
            self.model = unfreeze_backbone(self.model, num_layers_to_unfreeze=unfreeze_layers)
            self.model = compile_model(self.model, learning_rate=self.config.LEARNING_RATE_STAGE2)

            # Use a temporary checkpoint for this option
            temp_checkpoint = self.config.MODELS_DIR / f"temp_stage2_{unfreeze_layers}.keras"
            
            callbacks_stage2 = [
                tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(filepath=str(temp_checkpoint), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.2, patience=3, min_lr=1e-7, verbose=1)
            ]

            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=self.config.EPOCHS_STAGE2,
                class_weight=class_weight_dict,
                callbacks=callbacks_stage2,
                verbose=1
            )
            
            # Evaluate the best model for this option on validation data
            if temp_checkpoint.exists():
                temp_model = tf.keras.models.load_model(str(temp_checkpoint))
                val_metrics = temp_model.evaluate(self.val_ds, return_dict=True, verbose=0)
                auc = val_metrics.get("auc", -1.0)
                print(f"Option {unfreeze_layers} layers unfrozen -> val_auc: {auc:.4f}")
                
                if auc > best_val_auc:
                    best_val_auc = auc
                    best_history_s2 = history
                    # Save as the best stage 2 checkpoint
                    temp_model.save(str(stage2_checkpoint))
                    # Save history to CSV
                    pd.DataFrame(history.history).to_csv(stage2_csv, index=False)
                    print(f"New best Stage 2 configuration found: {unfreeze_layers} layers!")
                
                # Cleanup temp
                import os
                if temp_checkpoint.exists():
                    os.remove(str(temp_checkpoint))

        history_s2 = best_history_s2

        # ==========================================
        # Post-Training Consolidation
        # ==========================================
        print("\nConsolidating results...")
        
        # Load the overall best model (evaluate checkpoints from both stages)
        best_model_path = stage2_checkpoint if stage2_checkpoint.exists() else stage1_checkpoint
        if best_model_path.exists():
            self.model = tf.keras.models.load_model(str(best_model_path))
            # Save as the ultimate best model
            self.model.save(str(self.config.CHECKPOINT_PATH))
            print(f"Final best model saved to {self.config.CHECKPOINT_PATH}")

        # Fit Platt Scaling calibrator on validation set
        if self.val_ds is not None and getattr(self.config, "CALIBRATOR_PATH", None):
            print("Fitting Platt Scaling calibrator on validation set...")
            val_probs = self.model.predict(self.val_ds, verbose=0).flatten()
            val_probs_2d = val_probs.reshape(-1, 1)
            val_labels_np = np.array(self.val_labels)
            
            calibrator = LogisticRegression(solver='lbfgs')
            calibrator.fit(val_probs_2d, val_labels_np)
            joblib.dump(calibrator, self.config.CALIBRATOR_PATH)
            print(f"Calibrator saved to {self.config.CALIBRATOR_PATH}")

        # Combine stage 1 and stage 2 training history robustly from Keras history objects
        try:
            df1 = pd.DataFrame(history_s1.history)
            df1["epoch"] = history_s1.epoch
            
            if history_s2 is not None:
                df2 = pd.DataFrame(history_s2.history)
                df2["epoch"] = history_s2.epoch
            else:
                df2 = pd.DataFrame()
            
            # Standardize learning rate column names
            for df in [df1, df2]:
                if not df.empty:
                    if "lr" in df.columns and "learning_rate" not in df.columns:
                        df["learning_rate"] = df["lr"]
                    elif "learning_rate" in df.columns and "lr" not in df.columns:
                        df["lr"] = df["learning_rate"]
            
            # Offset stage 2 epochs to make the timeline continuous
            if not df1.empty and not df2.empty:
                last_epoch_s1 = df1["epoch"].iloc[-1] + 1
                df2["epoch"] = df2["epoch"] + last_epoch_s1

            dfs_to_concat = [df for df in [df1, df2] if not df.empty]
            if dfs_to_concat:
                df_combined = pd.concat(dfs_to_concat, ignore_index=True)
                df_combined.to_csv(self.config.HISTORY_PATH, index=False)
                print(f"Merged training history saved to {self.config.HISTORY_PATH}")
                # Generate plots
                plot_metrics(self.config.HISTORY_PATH, self.config.RESULTS_DIR)
        except Exception as e:
            print(f"Error merging histories or generating plots: {e}")

        # Evaluation on test dataset
        if self.test_ds is not None:
            print("\nEvaluating final model on test set...")
            eval_results = self.model.evaluate(self.test_ds, return_dict=True)
            print("\nFinal Test Metrics:")
            print("-------------------------")
            for metric_name, val in eval_results.items():
                print(f"{metric_name.capitalize()}: {val:.4f}")
            print("-------------------------")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_datasets()
    trainer.run_pipeline()
