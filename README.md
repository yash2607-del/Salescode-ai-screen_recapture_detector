
## Dataset
This model was trained exclusively on a self-collected dataset consisting of approximately 100 images:
- **~50 Real Photos**: Genuine photos of physical objects captured using a mobile phone.
- **~50 Screen Photos**: Photos taken of a digital monitor to purposefully capture screen glare and Moiré interference.

*(Note: While public datasets were utilized during initial experimentation, they have been securely removed from the final production repository. The final architecture relies entirely on the manually collected dataset to ensure real-world generalization).*

## Approach
This project utilizes a Deep Learning Transfer Learning Strategy. Instead of training a model from scratch, we leverage a pre-trained ImageNet model (e.g., MobileNetV2 or EfficientNet). The dense classification head is fine-tuned to recognize the high-frequency visual artifacts unique to LCD and OLED monitors.

## Training
- **Dataset Splitting**: The dataset is split rigidly into Training, Validation, and Test sets.
- **Data Augmentation**: Robust augmentation (flipping, rotation, brightness shifts) prevents the model from overfitting to the small dataset size.
- **Model Checkpointing**: The training loop saves the best performing weights based strictly on Validation Loss (`best_model.keras`).

## Evaluation
Run `python evaluate.py` to generate the exact evaluation metrics on the Test set.
The outputs include Accuracy, Precision, Recall, F1, ROC-AUC, and a complete Confusion Matrix saved to the `results/` folder.


## Cost
Assuming an AWS Lambda deployment (x86, 1024MB RAM at ~$0.0000166 per second), the inference latency translates to an estimated <$0.00001 per image (less than $10.00 per million images).

## Limitations
- Heavily textured physical items (like woven fabrics) can trick the convolutional filters into detecting false Moiré patterns.
- High-PPI Retina displays photographed perfectly in-focus with high-end cameras often produce zero visual artifacts, leading to False Negatives.

## Future Improvements
- Implement a tiny object detector (e.g., YOLOv8-nano) to rigidly crop out screen bezels before running the classification network.
- Expand the dataset to 1sc,000+ images sourced from diverse hardware: OLED screens, CRT monitors, e-ink displays, and varying smartphone camera lenses.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
To evaluate a single image, use the CLI tool:
```bash
python predict.py <path_to_image>
```

### Example
```bash
python predict.py dataset/test/screen/sample_01.jpg
```
*Output: `0.91`* (The probability that the image is a screen recapture).

## Streamlit Web Application
To run the interactive web interface:
```bash
streamlit run app.py
```

