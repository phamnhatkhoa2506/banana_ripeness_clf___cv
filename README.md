# Banana Ripeness Classification

This project uses YOLOv8 to classify the ripeness of bananas from images. The model can distinguish between different stages of banana ripeness.

## Project Structure

```
.
├── Banana-Ripeness-Classification-1/
│   ├── train/          # Training images
│   ├── valid/          # Validation images
│   └── test/           # Test images
├── runs/
│   └── classify/
│       └── train2/     # Training results and model weights
├── predict.py          # Script for making predictions
└── README.md          # This file
```

## Requirements

- Python 3.x
- PyTorch
- Ultralytics YOLOv8

You can install the required packages using:

```bash
pip install ultralytics torch
```

## Model Training

The model has been trained on a dataset of banana images, classified into different ripeness stages. The training results and model weights are stored in the `runs/classify/train2/weights/` directory.

## Making Predictions

To make predictions on new images, use the `predict.py` script:

```bash
python predict.py
```

The script will:
1. Load the trained model weights
2. Run validation on the test set
3. Make a prediction on a sample image
4. Output the predicted class and confidence score

## Model Performance

The model has been trained to classify bananas into different ripeness stages. The validation results show the model's performance on the test set.

## Dataset

The dataset is organized into the following structure:
- Training set: Contains images for model training
- Validation set: Used for model validation during training
- Test set: Used for final model evaluation

## License

[Add your license information here]

## Acknowledgments

- YOLOv8 by Ultralytics
- [Add any other acknowledgments here] 