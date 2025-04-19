# Deep Learning Assignment 2: Image Classification on iNaturalist Dataset  

## Introduction  
This project explores two approaches to classify images from the iNaturalist dataset:  
1. **Part A**: Building a Convolutional Neural Network (CNN) from scratch.  
2. **Part B**: Fine-tuning pre-trained models (trained on ImageNet) using transfer learning.  

---

## Project Structure  

### Part A - Custom CNN Implementation  
- **Objective**: Design and train a CNN from scratch without pre-trained weights.  
- **Key Features**:  
  - Configurable hyperparameters (activation functions, regularization, data augmentation).  
  - Modular architecture for flexible layer configurations.  
- **Code**: [GitHub Repository (Part A)](https://github.com/Vishnu000000/DeepLearning_A2/tree/main/A2_Part_A)  

### Part B - Transfer Learning with Pre-trained Models  
- **Objective**: Improve performance using architectures like InceptionV3, ResNet, and Xception.  
- **Key Features**:  
  - Partial/full fine-tuning of base models.  
  - Customizable dense layers with dropout and regularization.  
- **Code**: [GitHub Repository (Part B)](https://github.com/Vishnu000000/DeepLearning_A2/tree/main/A2_Part_B)  

---

## Hyperparameter Configurations  

### Shared Parameters  
- **Dropout**: `[0, 0.2, 0.4]`  
- **Batch Size**: `[32, 64]`  
- **Data Augmentation**: Enabled/Disabled  

### Part A-Specific  
- **Activation Functions**: `['relu', 'elu', 'selu']`  
- **Filters**: Multi-scale configurations (e.g., `[32, 64, 128, 256, 512]`).  
- **Regularization**: L2 weight decay (`[0, 0.00005, 0.0005]`).  

### Part B-Specific  
- **Base Models**: `['inceptionv3', 'resnet', 'xception']`  
- **Freezing Ratios**: `[0%, 33.3%, 66.6%, 100%]` (proportion of frozen base layers).  
- **Dense Layers**: Neurons per layer `[128, 256, 512]`.  

---

## Implementation Details  

### Model Construction  
#### `CNN()` Function  
- **Purpose**: Dynamically constructs CNN architectures.  
- **Parameters**:  
  - Filter sizes, activation functions, input shape.  
  - Regularization (dropout, L2 decay), batch normalization.  
  - Dense layer neurons and output size.  

#### Transfer Learning Setup  
- **Workflow**:  
  1. Load pre-trained base model (excluding top layers).  
  2. Add custom classification head with configurable dense layers and dropout.  
  3. Freeze specified portions of base layers to prevent overfitting.  

---

## Training & Evaluation  

### Training Process  
- **Method**:  
  - Compile model with adaptive learning rate (default: `0.0001`).  
  - Train using `fit()` with validation split.  
- **Augmentation**: On-the-fly transformations (rotation, flipping) when enabled.  

### Evaluation  
- **Metrics**: Accuracy, loss, and per-class precision/recall.  
- **Prediction**: Use `Predict()` for test-set inference.  

---

## Results & Analysis  
### WandB Report  
- **Link**: [Performance Metrics and Ablation Studies](https://wandb.ai/cs24m022-iit-madras-foundation/Deep_Learning_Assignment2_cs24m022/reports/Assignment-2--VmlldzoxMjM2NjYyNw)  
- **Key Insights**:  
  - Impact of freezing strategies on transfer learning efficiency.  
  - Trade-offs between data augmentation and training time.  
  - Optimal dropout rates for preventing overfitting.  

---

## Reproducibility  
 
