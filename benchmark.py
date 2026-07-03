"""
Benchmarking script for evaluating model inference speed, throughput, latency,
and resource consumption on CPU or GPU.
"""
import json
import os
import sys
import time
from pathlib import Path

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from PIL import Image

# Import configuration and prediction helpers
from src import config
from predict import preprocess_image


def get_inference_device() -> str:
    """Detects if GPU is available and used for inference."""
    gpus = tf.config.list_logical_devices("GPU")
    return "GPU" if gpus else "CPU"


def estimate_model_size_in_mb(model_path: Path) -> float:
    """Returns the size of the saved model on disk in Megabytes."""
    if model_path.exists():
        return model_path.stat().st_size / (1024 * 1024)
    return 0.0


def main():
    print("Initiating screen-recapture-detector performance benchmark...")
    
    model_path = config.CHECKPOINT_PATH
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at '{model_path}'. Run training first.", file=sys.stderr)
        sys.exit(1)

    # 1. Measure model loading time
    print("Loading model and checking inference device...")
    t0 = time.time()
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    model_loading_time = time.time() - t0

    device = get_inference_device()
    print(f"Model loaded in {model_loading_time:.4f} seconds.")
    print(f"Active inference device: {device}")

    # Locate a test image for benchmarking
    # Search in dataset folder
    test_images = list(config.DATASET_DIR.glob("**/*.jpg")) + list(config.DATASET_DIR.glob("**/*.png"))
    if not test_images:
        # Fallback: create a temporary image for benchmarking
        temp_img_path = Path("temp_bench_img.jpg")
        print(f"No test images found. Generating temporary image for benchmark at '{temp_img_path}'...")
        width, height = 224, 224
        data = bytes([128] * (width * height * 3))
        img = Image.frombytes("RGB", (width, height), data)
        img.save(temp_img_path)
        bench_image_path = temp_img_path
    else:
        bench_image_path = test_images[0]
        temp_img_path = None

    print(f"Benchmarking using image: {bench_image_path}")

    # Number of benchmarking runs
    iterations = 100
    preprocessing_times = []
    inference_times = []
    total_latencies = []

    # Warmup runs to compile graphs and stabilize TF state
    print("Performing warmup runs...")
    for _ in range(5):
        preprocessed = preprocess_image(bench_image_path, config.IMAGE_SIZE)
        _ = model(preprocessed, training=False).numpy()

    # Benchmark loop
    print(f"Running benchmark loop for {iterations} iterations...")
    for i in range(iterations):
        # Preprocessing time
        t_prep_start = time.time()
        preprocessed = preprocess_image(bench_image_path, config.IMAGE_SIZE)
        t_prep = time.time() - t_prep_start
        preprocessing_times.append(t_prep)

        # Inference time
        t_inf_start = time.time()
        _ = model(preprocessed, training=False).numpy()
        t_inf = time.time() - t_inf_start
        inference_times.append(t_inf)

        # Total latency per image
        total_latencies.append(t_prep + t_inf)

    # Clean up temp image if created
    if temp_img_path and temp_img_path.exists():
        temp_img_path.unlink()

    # Calculate statistics (in milliseconds)
    def get_stats(times_list):
        times_ms = np.array(times_list) * 1000
        return {
            "mean_ms": float(np.mean(times_ms)),
            "median_ms": float(np.median(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "std_ms": float(np.std(times_ms))
        }

    prep_stats = get_stats(preprocessing_times)
    inf_stats = get_stats(inference_times)
    latency_stats = get_stats(total_latencies)

    # Throughput (images per second based on mean latency)
    throughput = 1.0 / np.mean(total_latencies)

    # Model metrics
    model_size_mb = estimate_model_size_in_mb(model_path)
    total_params = model.count_params()
    
    # Memory estimation (VRAM / process RAM estimates)
    # 4 bytes per float32 parameter
    est_parameter_memory_mb = (total_params * 4) / (1024 * 1024)
    
    # Try importing psutil for host process memory estimation
    host_memory_mb = 0.0
    try:
        import psutil
        process = psutil.Process(os.getpid())
        host_memory_mb = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    results = {
        "device": device,
        "model_loading_time_seconds": model_loading_time,
        "model_size_mb": model_size_mb,
        "total_parameters": total_params,
        "estimated_parameter_memory_mb": est_parameter_memory_mb,
        "host_process_memory_mb": host_memory_mb,
        "iterations": iterations,
        "throughput_fps": throughput,
        "preprocessing_latency_ms": prep_stats,
        "inference_latency_ms": inf_stats,
        "total_latency_ms": latency_stats
    }

    # Save to results/benchmark.json
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.RESULTS_DIR / "benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save to results/benchmark.txt
    txt_path = config.RESULTS_DIR / "benchmark.txt"
    report_lines = [
        "==================================================",
        "Screen Recapture Detector - Inference Benchmark",
        "==================================================",
        f"Inference Device:              {device}",
        f"Model Size on Disk:            {model_size_mb:.2f} MB",
        f"Total Parameter Count:         {total_params:,}",
        f"Est. Parameter Memory (RAM):   {est_parameter_memory_mb:.2f} MB",
        f"Host Process RSS Memory:       {host_memory_mb:.2f} MB (if profiled)",
        f"Model Loading Time:            {model_loading_time:.4f} seconds",
        f"Benchmark Iterations:          {iterations}",
        f"Throughput:                    {throughput:.2f} images/second",
        "",
        "Latency Statistics (in milliseconds):",
        "--------------------------------------------------",
        "1. Image Preprocessing Latency:",
        f"  - Average (Mean):            {prep_stats['mean_ms']:.2f} ms",
        f"  - Median:                    {prep_stats['median_ms']:.2f} ms",
        f"  - Minimum:                   {prep_stats['min_ms']:.2f} ms",
        f"  - Maximum:                   {prep_stats['max_ms']:.2f} ms",
        f"  - Std Dev:                   {prep_stats['std_ms']:.2f} ms",
        "",
        "2. Backbone Inference Latency:",
        f"  - Average (Mean):            {inf_stats['mean_ms']:.2f} ms",
        f"  - Median:                    {inf_stats['median_ms']:.2f} ms",
        f"  - Minimum:                   {inf_stats['min_ms']:.2f} ms",
        f"  - Maximum:                   {inf_stats['max_ms']:.2f} ms",
        f"  - Std Dev:                   {inf_stats['std_ms']:.2f} ms",
        "",
        "3. Combined End-to-End Latency:",
        f"  - Average (Mean):            {latency_stats['mean_ms']:.2f} ms",
        f"  - Median:                    {latency_stats['median_ms']:.2f} ms",
        f"  - Minimum:                   {latency_stats['min_ms']:.2f} ms",
        f"  - Maximum:                   {latency_stats['max_ms']:.2f} ms",
        f"  - Std Dev:                   {latency_stats['std_ms']:.2f} ms",
        "=================================================="
    ]
    txt_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Output to stdout
    print("\n".join(report_lines))
    print(f"\nSaved benchmark reports to '{json_path}' and '{txt_path}'.")


if __name__ == "__main__":
    main()
