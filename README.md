
# Deep Learning Assignment 2: Image Classification on iNaturalist Dataset
Introduction
This assignment implements two approaches for classifying images from the iNaturalist dataset:

Part A: Building a Convolutional Neural Network (CNN) from scratch.

Part B: Fine-tuning pre-trained models (originally trained on ImageNet) for transfer learning.

Project Structure
Part A - Custom CNN Implementation
Objective: Design and train a CNN architecture without pre-trained weights.

Key Components:

Configurable hyperparameters (activation functions, regularization, data augmentation).

Modular implementation for flexibility in layer configurations.

GitHub Link

Part B - Transfer Learning with Pre-trained Models
Objective: Leverage architectures like InceptionV3, ResNet, and Xception for improved performance.

Key Features:

Partial/full fine-tuning of base models.

Customizable dense head with dropout and regularization.

GitHub Link

Hyperparameter Configurations
Shared Parameters
Dropout: [0, 0.2, 0.4]

Batch Size: [32, 64]

Data Augmentation: Enabled/Disabled

Part A-Specific
Activation Functions: ['relu', 'elu', 'selu']

Filters: Multi-scale configurations (e.g., [32, 64, 128, 256, 512]).

Regularization: L2 weight decay ([0, 0.00005, 0.0005]).

Part B-Specific
Base Models: ['inceptionv3', 'resnet', 'xception']

Freezing Ratios: [0%, 33.3%, 66.6%, 100%] (proportion of frozen base layers).

Dense Layers: Neurons per layer [128, 256, 512].

Implementation Details
Model Construction
CNN() Function
Purpose: Dynamically constructs CNN architectures.

Parameters:

Filter sizes, activation functions, input shape.

Regularization (dropout, L2 decay), batch normalization.

Dense layer neurons and output size.

Transfer Learning Setup
Workflow:

Load pre-trained base model (excluding top layers).

Add custom classification head with configurable dense layers and dropout.

Freeze specified portions of base layers to prevent overfitting.

Training & Evaluation
Training Process
Method:

Compile model with adaptive learning rate (default: 0.0001).

Train using fit() with validation split.

Augmentation: On-the-fly transformations (rotation, flipping) when enabled.

Evaluation
Metrics: Accuracy, loss, and per-class precision/recall.

Prediction: Use Predict() for test-set inference.

Results & Analysis
WandB Report
Link: Performance Metrics and Ablation Studies

Key Insights:

Impact of freezing strategies on transfer learning efficiency.

Trade-offs between data augmentation and training time.

Optimal dropout rates for preventing overfitting.

Reproducibility
Clone the repository and install dependencies.

Download the iNaturalist dataset.

Execute train.py with desired hyperparameters.
