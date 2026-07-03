# Technical Report: Screen Recapture Detection

## 1. Why This Approach?
Deep Learning via Transfer Learning is the industry standard for complex image classification tasks. Rather than building handcrafted feature extractors (like FFT or LBP) which require constant manual tuning and heuristics, a Convolutional Neural Network (CNN) automatically learns the optimal spatial filters required to detect screen artifacts. By leveraging a pre-trained ImageNet model, we drastically reduce training time and ensure the early convolutional layers possess high-quality generalized feature extraction capabilities.

## 2. Dataset Collection
The initial prototyping utilized large-scale downloaded images to validate the pipeline. However, downloaded datasets often possess uniform compression artifacts and focal lengths that do not accurately represent real-world mobile capture behavior.
For the final production model, **all downloaded datasets were completely removed from the training pipeline**. The model was trained entirely on a self-collected dataset of approximately 100 images (~50 real physical environments, ~50 screen monitor recaptures) taken strictly using a mobile phone. This guarantees the model learns the genuine focal distortion, lighting dynamics, and sensor noise characteristics of an actual presentation attack.

## 3. Evaluation
The model's performance was evaluated on a dedicated Test set. Evaluation metrics were strictly derived from the self-collected images. The final evaluation logs Accuracy, Precision, Recall, F1, and ROC-AUC. 
During training, aggressive data augmentation (flipping, random rotations, brightness variations) was employed to prevent overfitting on the limited self-collected dataset.

## 4. Failure Cases
- **False Positives (Real flagged as Screen)**: Heavily textured natural items (tightly woven fabric, brick walls) generate high-frequency edges that trick the convolutional filters into detecting a pixel grid.
- **False Negatives (Screen flagged as Real)**: Ultra-high-PPI Retina displays photographed perfectly in-focus with high-end smartphone sensors may produce absolutely zero Moiré interference, effectively blinding the CNN.

## 5. Trade-Offs
By choosing Deep Learning over Classical Computer Vision:
- **Pros**: The model can learn abstract semantic contexts (e.g., recognizing a laptop keyboard or desk in the background) rather than relying purely on texture arrays. It requires less manual feature engineering.
- **Cons**: Inference latency is higher (~100ms vs 5ms for classical models). The model footprint is larger, making extreme edge-deployments (like IoT devices) slightly more complex. Explainability is also lower, requiring tools like Grad-CAM to interpret decisions.

## 6. How Accuracy Could Improve
Given the tiny size of the self-collected dataset (~100 images), the model currently risks overfitting to the specific monitor used for the screen recaptures, or the specific lighting of the room.
Accuracy and generalization would scale dramatically by increasing the dataset to 10,000+ images sourced from diverse hardware: OLED screens, CRT monitors, e-ink displays, glossy vs matte coatings, and captured using varying smartphone camera lenses.

## 7. Adapting to Evolving Attackers
As fraudsters adapt (e.g., by slightly blurring the screen to destroy the Moiré pattern or using matte privacy filters to kill glare), the model's static decision boundaries will degrade.
To counter this, the pipeline must be continuously retrained with adversarial examples (e.g., intentionally blurred screens) fed back into the training loop, forcing the CNN to find entirely new, subtle artifacts.

## 8. Threshold Selection
Thresholds should never default to `0.50` in security applications. 
The pipeline systematically sweeps thresholds from `0.05` to `0.95` in `0.01` intervals, explicitly selecting the threshold that strictly maximizes the **F1-Score** on the Validation set. This mathematically balances the trade-off between False Positives (blocking real users) and False Negatives (letting fraud through).

## 9. Optimizing for Mobile Deployment
To deploy this heavy CNN natively on-device (Android/iOS) for zero-latency offline inference:
1. The Keras (`.keras`) model can be explicitly converted to **TensorFlow Lite (.tflite)** format using the `TFLiteConverter`.
2. Post-Training Quantization (e.g., `INT8` quantization) can be applied during the conversion, reducing the model size by up to 4x and speeding up CPU inference significantly with minimal accuracy loss.
3. The resulting lightweight model can be loaded natively on mobile using the standard TensorFlow Lite delegates (or NNAPI/CoreML for hardware acceleration).
