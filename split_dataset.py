"""
Script to automatically split the dataset into train, validation, and test sets.
Generates mock images if no raw dataset is present for demonstration purposes.
"""
import sys
import random
from pathlib import Path
from PIL import Image

# Add src to python path if executing from root directory
sys.path.append(str(Path(__file__).resolve().parent))

from src import config
from src.dataset import DatasetSplitter

def generate_mock_images(dataset_dir: Path, num_images: int = 40):
    """
    Generate mock images using pure Python/PIL to avoid heavy dependencies for setup.
    """
    print(f"Raw dataset not found. Generating {num_images} mock images per class for demonstration...")
    width, height = 224, 224
    for cls in ["real", "screen"]:
        cls_dir = dataset_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_images):
            # Create a random RGB image using random bytes
            data = bytes([random.randint(0, 255) for _ in range(width * height * 3)])
            img = Image.frombytes("RGB", (width, height), data)
            img.save(cls_dir / f"mock_{cls}_{i+1:03d}.jpg")

def main():
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "dataset"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Check if raw directories exist and contain images
    real_dir = dataset_dir / "real"
    screen_dir = dataset_dir / "screen"

    has_images = False
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if real_dir.exists() and screen_dir.exists():
        real_files = [p for p in real_dir.glob("*") if p.suffix.lower() in supported_extensions]
        screen_files = [p for p in screen_dir.glob("*") if p.suffix.lower() in supported_extensions]
        if len(real_files) > 0 and len(screen_files) > 0:
            has_images = True

    if not has_images:
        generate_mock_images(dataset_dir, num_images=40)

    print("Initiating dataset split...")
    splitter = DatasetSplitter(
        source_dir=dataset_dir,
        target_dir=dataset_dir,
        split_ratio=(0.70, 0.15, 0.15),
        seed=config.SEED
    )

    try:
        splits = splitter.execute_split()
    except Exception as e:
        print(f"Error executing split: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate statistics
    train_real = len(splits["train"]["real"])
    train_screen = len(splits["train"]["screen"])
    train_total = train_real + train_screen

    val_real = len(splits["val"]["real"])
    val_screen = len(splits["val"]["screen"])
    val_total = val_real + val_screen

    test_real = len(splits["test"]["real"])
    test_screen = len(splits["test"]["screen"])
    test_total = test_real + test_screen

    total_real = train_real + val_real + test_real
    total_screen = train_screen + val_screen + test_screen
    total_images = total_real + total_screen

    # Print statistics to stdout
    print("\nDataset Split Statistics:")
    print("--------------------------------------------------")
    print(f"Number of train images:      {train_total} (real: {train_real}, screen: {train_screen})")
    print(f"Number of validation images: {val_total} (real: {val_real}, screen: {val_screen})")
    print(f"Number of test images:       {test_total} (real: {test_real}, screen: {test_screen})")
    print("--------------------------------------------------")
    print(f"Total processed images:      {total_images}")
    print("\nClass Balance:")
    print(f"- Real:   {total_real} images ({total_real / total_images * 100:.2f}%)")
    print(f"- Screen: {total_screen} images ({total_screen / total_images * 100:.2f}%)")
    print("--------------------------------------------------")

    # Save to results/dataset_statistics.txt
    stats_file = results_dir / "dataset_statistics.txt"
    try:
        with open(stats_file, "w") as f:
            f.write("==================================================\n")
            f.write("Screen Recapture Detector - Dataset Statistics\n")
            f.write("==================================================\n")
            f.write(f"Total Valid Images: {total_images}\n\n")
            f.write("Images per class:\n")
            f.write(f"- Real: {total_real}\n")
            f.write(f"- Screen: {total_screen}\n\n")
            f.write("Train distribution:\n")
            f.write(f"- Total: {train_total} ({train_total / total_images * 100:.1f}% of total)\n")
            f.write(f"  * Real: {train_real} ({train_real / train_total * 100:.1f}% of split)\n")
            f.write(f"  * Screen: {train_screen} ({train_screen / train_total * 100:.1f}% of split)\n\n")
            f.write("Validation distribution:\n")
            f.write(f"- Total: {val_total} ({val_total / total_images * 100:.1f}% of total)\n")
            f.write(f"  * Real: {val_real} ({val_real / val_total * 100:.1f}% of split)\n")
            f.write(f"  * Screen: {val_screen} ({val_screen / val_total * 100:.1f}% of split)\n\n")
            f.write("Test distribution:\n")
            f.write(f"- Total: {test_total} ({test_total / total_images * 100:.1f}% of total)\n")
            f.write(f"  * Real: {test_real} ({test_real / test_total * 100:.1f}% of split)\n")
            f.write(f"  * Screen: {test_screen} ({test_screen / test_total * 100:.1f}% of split)\n")
        print(f"Saved dataset statistics to {stats_file.relative_to(base_dir)}")
    except Exception as e:
        print(f"Error saving statistics file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
