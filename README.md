# CNN Image Classification on CIFAR-10

End-to-end image classification project using Convolutional Neural Networks (CNN) trained on the CIFAR-10 dataset with TensorFlow/Keras.

## Problem Statement
The objective of this project is to build an image classification model that accurately categorizes images from the CIFAR-10 dataset into one of 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. This involves preprocessing image data, designing a CNN architecture, training the model, and evaluating its performance. The project demonstrates a complete computer vision pipeline useful for image recognition tasks.

## Dataset Description
- **Source**: CIFAR-10 dataset, available through TensorFlow/Keras (`tf.keras.datasets.cifar10`).
- **Features**:
  - Images: 32x32 pixel RGB images.
  - Labels: 10 categorical classes.
- **Size**: 60,000 images (50,000 training, 10,000 testing).
- **Preprocessing**:
  - Normalization: Pixel values scaled to [0, 1] by dividing by 255.
- **Derived Features**: None additional; raw images used as input.

## CNN Model Architecture
- **Task Type**: Multi-class image classification (10 classes).
- **Framework**: TensorFlow/Keras.
- **Architecture**:
  - Convolutional layers: 3 Conv2D layers (32, 64, 64 filters) with ReLU activation.
  - Pooling: MaxPooling2D layers for downsampling.
  - Fully Connected: Flatten layer followed by Dense layers (64 units with ReLU, 10 units with softmax).
- **Input Shape**: (32, 32, 3) for RGB images.
- **Output**: Probability distribution over 10 classes.

## Training Details
- **Loss Function**: Sparse Categorical Crossentropy (suitable for integer labels).
- **Optimizer**: Adam optimizer.
- **Metrics**: Accuracy.
- **Epochs**: 10 epochs.
- **Validation**: Validation on test set during training.
- **Batch Size**: Default (not specified, uses Keras default).

## Evaluation Metrics
- **Accuracy**: Overall correct predictions on test set.
- **Training History**: Plots of training and validation accuracy over epochs.

## Results
Based on typical runs with this architecture:
- **Test Accuracy**: Around 70-75% after 10 epochs (may vary with random initialization).
- **Training Behavior**: Accuracy increases over epochs; potential overfitting if validation accuracy plateaus.
- **Visualization**: Accuracy curves show model learning progress.

## How to Run
1. **Prerequisites**: Python 3.8+, TensorFlow 2.x, Jupyter Notebook.
2. **Install Dependencies**:
   ```bash
   pip install tensorflow numpy matplotlib seaborn pandas
   ```
3. **Run the Notebook**:
   ```bash
   jupyter notebook cnn_image_classification.ipynb
   ```
4. Execute cells to load data, build model, train, and evaluate.

## Key Learnings
- **CNN Basics**: Convolutional layers extract spatial features; pooling reduces dimensionality.
- **Data Preprocessing**: Normalization improves training stability.
- **Model Training**: Monitoring validation metrics helps detect overfitting.
- **TensorFlow/Keras**: Sequential API simplifies model building.
- **Practical Insights**: CIFAR-10 is challenging due to small image size; deeper networks or data augmentation can improve performance.

## Project Structure
- `cnn_image_classification.ipynb`: Main notebook with code and comments.
- `README.md`: This file with project documentation.
