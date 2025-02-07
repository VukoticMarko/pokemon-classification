# Pokemon Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify Pokemon images from a dataset of 1000 different Pokemon species and forms. The model leverages various data augmentation techniques and a deep learning architecture to achieve classification.

## Dataset
**Dataset Source**: [Pokemon Dataset - 1000](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000)  
- Contains 1000 Pokemon categories
- Images include various poses, battle states, and regional forms
- Original images resized to 128x128 pixels for processing

Extract the dataset into a folder and take folder named 'dataset`, rename it to 'pokemon-dataset' and place it in the root of the project.

**Note**: While the dataset is large in terms of the number of Pokemon categories, it lacks sufficient images for many Pokemon, which contributes to lower validation and test accuracy. This is a common challenge when dealing with highly imbalanced datasets.

## Project Structure
### Data Preparation
- 80-10-10 split for training, validation, and test sets
- Data augmentation includes:
  - Random flips (left-right & up-down)
  - Brightness/contrast adjustments
  - Rotation (90°)
  - Grayscale conversion
  - Random cropping/zooming
  - Hue variation
  - Gaussian noise injection

### Model Architecture
```python
Sequential([
    Conv2D(12, 3, activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(16, 3, activation='relu'),
    Dropout(0.1),
    # ... (multiple convolutional and pooling layers)
    Flatten(),
    Dense(20, activation='relu'),
    Dense(100, activation='relu'),
    Dense(1000, activation='softmax')
])
```

Key Features:

- 6 Convolutional layers with increasing filters (12-64)

- Batch Normalization for stable training

- Dropout layers (10-30%) for regularization

- Final softmax layer for 1000-class classification

### Training
- Adam optimizer with categorical crossentropy loss

- 1000 epochs with early stopping via ModelCheckpoint

- Best weights saved to model/weights/best_weights/

- Training metrics tracked:

  - Accuracy
  
  - Loss
  
  - Validation metrics

### Evaluation
- Test set evaluation after training

- Metrics saved to `model/metrics.txt`

- Visualization of training progress:

  - Accuracy plot (train vs validation)

  - Loss plot (train vs validation)

### Results
The model achieved the following metrics:
```bash
------------------------------
Train Accuracy: 0.8453
Train Loss: 0.5776
------------------------------
Validation Accuracy: 0.5938
Validation Loss: 3.3520
------------------------------
Test Accuracy: 0.5871
Test Loss: 3.5002
------------------------------
```

**Note:** The validation and test accuracy are lower than the training accuracy, which is expected given the dataset's limitations. The dataset contains 1000 Pokemon categories but lacks sufficient images for many of them, leading to overfitting on the training set and lower generalization performance on the validation and test sets.

### Requirements
Everything can be found in `requirements.txt`.
- Python 3.7+

- TensorFlow 2.x

- Keras

- OpenCV

- Matplotlib

- NumPy

### Usage
Download dataset from Kaggle

Place dataset in project directory as pokemon-dataset

Run the training script:
```bash
python train_network.py
```
### Visualization
Training progress is shown through two plots:

1. Model Accuracy: Comparison of training vs validation accuracy

2. Model Loss: Comparison of training vs validation loss

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
