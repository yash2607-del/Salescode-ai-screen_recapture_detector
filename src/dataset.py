"""
Dataset module for loading, preprocessing, and splitting screen recapture data using TensorFlow.
Provides DatasetSplitter for splitting raw images and ImageDataset for tf.data pipelines.
"""
import random
import shutil
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import tensorflow as tf
from PIL import Image

class ImageDataset:
    """
    TensorFlow tf.data based dataset loader and preprocessor.
    Supports both training/evaluation (with labels) and inference (without labels).
    Note: EfficientNetV2B0 has a built-in rescaling layer (scale = 1/255.0),
    so images are kept in the [0, 255] float32 range.
    """
    def __init__(
        self,
        image_paths: List[Path],
        labels: Optional[List[int]] = None,
        image_size: int = 224,
        seed: int = 42
    ):
        """
        Args:
            image_paths (List[Path]): List of file paths to images.
            labels (List[int], optional): List of target labels. None for inference mode.
            image_size (int): Dimensions to resize images.
            seed (int): Random seed for augmentation layers.
        """
        self.image_paths = [str(p) for p in image_paths]
        self.labels = labels
        self.image_size = image_size
        self.seed = seed

    def get_dataset(
        self,
        batch_size: int = 32,
        is_training: bool = True,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset pipeline.
        Optimized with caching, prefetching, and parallel mapping.
        """
        if not self.image_paths:
            raise ValueError("Cannot create tf.data.Dataset with an empty list of image paths.")

        if self.labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)

        if shuffle and is_training:
            # Shuffle paths before loading
            dataset = dataset.shuffle(buffer_size=len(self.image_paths), seed=self.seed, reshuffle_each_iteration=True)

        def _load_and_resize(path, label=None):
            # Load file from disk
            img = tf.io.read_file(path)
            # Decode JPEG/PNG image to 3 channels
            img = tf.image.decode_jpeg(img, channels=3)
            # Resize image to target dimensions
            img = tf.image.resize(img, [self.image_size, self.image_size])
            # Keep in [0, 255] float32 range for EfficientNetV2 internal rescaling layer
            img = tf.cast(img, tf.float32)
            if label is not None:
                return img, label
            return img

        # Map loading and resizing in parallel
        dataset = dataset.map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

        # Cache decoded and resized images in memory to save CPU/disk cycles
        dataset = dataset.cache()

        if is_training:
            def _augment(img, label=None):
                # Random Horizontal Flip
                img = tf.image.random_flip_left_right(img)
                # Random Brightness (max_delta in pixel values, 0.2 * 255.0 = 51.0)
                img = tf.image.random_brightness(img, max_delta=51.0)
                # Random Contrast
                img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
                # Clip values to ensure they stay in [0, 255] range
                img = tf.clip_by_value(img, 0.0, 255.0)

                if label is not None:
                    return img, label
                return img

            # Reshuffle cached images every epoch
            dataset = dataset.shuffle(buffer_size=len(self.image_paths), seed=self.seed, reshuffle_each_iteration=True)
            # Map data augmentation
            dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset


class ScreenRecaptureDataset(ImageDataset):
    """
    Subclass of ImageDataset that loads images from a directory structure.
    """
    def __init__(self, data_dir: Path, image_size: int = 224, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.classes = ["real", "screen"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        image_paths = []
        labels = []
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for cls in self.classes:
            cls_dir = self.data_dir / cls
            if not cls_dir.is_dir():
                continue
            for file_path in cls_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    image_paths.append(file_path)
                    labels.append(self.class_to_idx[cls])

        super().__init__(image_paths=image_paths, labels=labels, image_size=image_size, seed=seed)


class DatasetSplitter:
    """
    Handles scanning, data validation, and stratified splitting of the raw dataset
    into train, validation, and test subsets.
    """
    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42
    ):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.split_ratio = split_ratio
        self.seed = seed
        self.classes = ["real", "screen"]
        self.supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        assert abs(sum(split_ratio) - 1.0) < 1e-9, "Split ratios must sum to 1.0"

    def _is_valid_image(self, path: Path) -> bool:
        """Helper to verify image file integrity."""
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
            return True
        except Exception:
            return False

    def scan_dataset(self) -> Dict[str, List[Path]]:
        valid_images = {cls: [] for cls in self.classes}
        for cls in self.classes:
            # We look directly under source_dir/real and source_dir/screen
            cls_dir = self.source_dir / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Required class directory not found: {cls_dir}")

            for file_path in cls_dir.rglob("*"):
                if file_path.is_file():
                    if file_path.suffix.lower() in self.supported_extensions:
                        if self._is_valid_image(file_path):
                            valid_images[cls].append(file_path)
                        else:
                            print(f"Warning: Corrupted image ignored: {file_path}")
        return valid_images

    def split(self) -> Dict[str, Dict[str, List[Path]]]:
        scanned_data = self.scan_dataset()
        splits = {
            "train": {cls: [] for cls in self.classes},
            "val": {cls: [] for cls in self.classes},
            "test": {cls: [] for cls in self.classes}
        }

        r_train, r_val, _ = self.split_ratio
        rng = random.Random(self.seed)

        for cls in self.classes:
            paths = list(scanned_data[cls])
            rng.shuffle(paths)

            n = len(paths)
            n_train = int(round(n * r_train))
            n_val = int(round(n * r_val))

            splits["train"][cls] = paths[:n_train]
            splits["val"][cls] = paths[n_train:n_train + n_val]
            splits["test"][cls] = paths[n_train + n_val:]

        return splits

    def execute_split(self) -> Dict[str, Dict[str, List[Path]]]:
        splits = self.split()

        # Clean and construct directories safely
        for split_name in ["train", "val", "test"]:
            for cls in self.classes:
                dest_dir = self.target_dir / split_name / cls
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy data files instead of moving them, keeping raw dataset read-only
        for split_name, class_data in splits.items():
            for cls, paths in class_data.items():
                dest_dir = self.target_dir / split_name / cls
                for p in paths:
                    shutil.copy2(p, dest_dir / p.name)

        return splits
