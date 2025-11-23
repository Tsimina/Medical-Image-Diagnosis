# Medical-Image-Diagnosis 

Deep-learning pipeline for **chest X-ray diagnosis** and **adversarial robustness** study.

This project trains a **MobileNetV2** classifier on the Kaggle *Chest X-Ray Images (Pneumonia)* dataset and evaluates how vulnerable it is to adversarial attacks such as **FGSM** and **One-Pixel**. It also implements two defence strategies:

- **Lipschitz regularization**
- **Defensive distillation** (with different temperatures)

The goal is to compare the **baseline model** vs **defended models** on clean and adversarial examples.

## Table of contents 

- [Features](#features)
- [Repository Structure](#repository-structure)

- [Dataset](#dataset)
  - [Dataset Structure](#dataset-structure)
  - [Importing the Dataset](#importing-the-dataset)

- [Models Implemented](#models-implemented)
  - [Baseline MobileNetV2](#baseline-mobilenetv2)
  - [Defensive Distillation](#defensive-distillation)
  - [Lipschitz Regularization](#lipschitz-regularization)

- [Training Scripts](#training-scripts)
  - [Train Baseline Model](#train-baseline-model)
  - [Train Distilled Model](#train-distilled-model)
  - [Train Lipschitz Model](#train-lipschitz-model)

- [Evaluation Scripts](#evaluation-scripts)
  - [Baseline Evaluation](#baseline-evaluation)
  - [Distilled Evaluation](#distilled-evaluation)
  - [Lipschitz Evaluation](#lipschitz-evaluation)

- [Adversarial Attacks](#adversarial-attacks)
  - [FGSM Attack](#fgsm-attack)
  - [One Pixel Attack](#one-pixel-attack)

- [FGSM Performance Results](#fgsm-performance-results)
- [One-Pixel Attack Results](#one-pixel-attack-results)

- [Future Work](#future-work)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

## Features

- Baseline **MobileNetV2** classifier on chest X-rays  
- Training & evaluation on **Normal vs Pneumonia** (binary classification)  
- **Adversarial attacks**
  - FGSM (Fast Gradient Sign Method)
  - One-Pixel attack
- **Defence methods**
  - Lipschitz regularization
  - Defensive distillation (teacher–student, with temperatures `T = 35` and `T = 50`, `α = 0.9`)
- Metrics & plots:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix
  - ROC curve & AUC
  - Accuracy vs ε under FGSM (baseline vs distilled)

## Repository structure

```text
Medical-Image-Diagnosis/
│
├── attacks/                 # FGSM, One-Pixel and other attack utilities
├── dataset/                 # Dataset placeholder / notes (images stored locally)
├── model_configuration/     # Configs (paths, hyperparameters, model factory, etc.)
│   ├── model_config.py      # Contains hyperparameters and paths to the train dataset
│   └── Mobilenetv2.py       # Model function
├── results/                 # Saved weights, metrics logs, ROC curves, evaluation attack plots, etc.
├── src/                     # Training & evaluation scripts (baseline + defences)
├── test_application/        # Simple tests for the saved models
|   ├── saved_models         # Saved model configs for baseline and defence  
├── utils/                   # Metrics, training helpers, distillation utilities
│
├── requirements.txt          # Python dependencies (pip install -r)
├── LICENSE                  # MIT License
└── README.md
```

## Dataset 

Our MobileNetV2 was trained on the Kaggle *Chest X-Ray Images (Pneumonia)*. You can download the dataset here: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). 

<img width="716" height="220" alt="xray" src="https://github.com/user-attachments/assets/eabb0b86-54f4-43f5-b3ed-b7546b762e9b" />


The dataset contains 5865 JPEG images of chest x-rays, being split by default into 3 folders train/test/val. The distribution for each data split can be seen in the graph attached below.

<img width="520" height="334" alt="Graph" src="https://github.com/user-attachments/assets/1e9d339f-2d68-436f-b33d-dab1e9f0fedf" />

 > Figure: Data split distribution - dark pink for the pneumonia chest x-rays and the lighter pink for the normal chest x-rays.

## Attacks

### One-Pixel attack

A black-box attack that finds a misclassification by changing only one pixel in the
image.
Does not require model gradients; instead it evaluates only the model’s outputs.
Uses an evolutionary algorithm (e.g., differential evolution) to search for:

      x′ = x + δ(i,j)
      
where δ(i,j) modifies a single pixel at position (i, j).

### FGSM 

A white-box attack that uses the model’s gradients to find the direction that increases
the loss the most.
Creates an adversarial example by adding a small, targeted perturbation to the input
image:

     x′ = x + ϵ · sign(∇x J(x, y ))
     
The parameter ϵ controls the strength of the perturbation.
Called “sign” because it uses only the sign of the gradient (+1 or −1) to decide the
direction of each pixel change.

## Model Details

This repository contains 3 models:
- Baseline MobileNetV2
- MobileNetV2 with Lipschitz Regularization
- MobileNetV2 Defensive Distillation model
  
### Baseline MobileNetV2 configuration
For the baseline configuration we utilised a clasic MobilenetV2 architecture.
  
<img width="850" height="297" alt="mobilenet_architecture" src="https://github.com/user-attachments/assets/e2b5f4b6-0623-4617-b65b-1a18de22bc6e" />
  > Figure: Dataset sample.

Preprocessing (train):
  - Grayscale: 3 channels, resize to 224$\times$224
  - Random horizontal flip, random rotation
  - Normalization with ImageNet mean/std
Training setup:
  - Optimizer: Adam
  - LR: 0.001
  - Loss: Cross-Entropy
  - Batch size: 16 (train), 32 (val/test), 25 epochs

### MobileNetV2 with Lipschitz Regularization

This model applies two techniques to improve robustness:

- **Spectral Normalization** on all convolutional and fully connected layers  
  Ensures the network has bounded Lipschitz constant.

- **Gradient Penalty (Lipschitz penalty)**  
  Adds a regularization term:  
  \[
  L_{lip} = \lambda \cdot \| \nabla_x f(x) \|_2^2
  \]  
  This reduces sensitivity to small perturbations and improves adversarial robustness.

- **Training Loss:**  
  \[
  L = L_{\text{cross-entropy}} + L_{\text{lip}}
  \]

### MobileNetV2 with Defensive Distillation

We trained the distilled model using:

- **Teacher model:** baseline MobileNetV2  
- **Student model:** same architecture, trained on *soft labels*
- **Temperature:** T = 50  
- **Soft + Hard loss:**  
  \[
  L = \alpha \cdot L_{soft}(T) + (1-\alpha) \cdot L_{hard}
  \]

Soft probabilities from the teacher smooth the decision boundaries, making the model harder to attack.

## Requirements
- Python 3.7 or later  
- PyTorch  
- NumPy, SciPy, Matplotlib 
- (Optional) CUDA for GPU acceleration

## Installation

**Clone the repository:**

```
git clone https://github.com/Tsimina/Medical-Image-Diagnosis.git
cd Medical-Image-Diagnosis
```

**Create and activate a virtual environment (optional)**

```
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate.bat   # Windows
```

**Install dependencies**
```
pip install -r requirements.txt
```

## Usage

**Get training dataset**

The dataset will be downloaded locally from Kaggle by running this script:

```
cd Medical-Image-Diagnosis
python -m dataset.import_data
```

> [!IMPORTANT] 
> The dataset is already split into subdirectories train/test/val, for the train and test application you just need to provide the ROOT of the directory.

**Train**

To train your own model you just need to update with the paths to your dataset and results directory in the *_config.py files, depending on the target you want to achive you can also change the hyperparameters. 
```
model_config ----------------------> train_model
                            |
MobileNetV2_model_function -|
                            |
model_metrics --------------|
```
- DATA_DIR – path to your chest X-ray dataset
- RESULTS_DIR – where to save logs, models, plots
- Path to trained model weights
- Training hyperparameters: EPOCHS, BATCH_TRAIN, BATCH_EVAL, LR
- Specific hyperparameters for each defense model configuration

To start the training process run the following command:
```
cd Medical-Image-Diagnosis
python -m src.train_<model_name>
```

**Example**
```
# Baseline
python -m src.train_baseline_mobilenetv2

# Lipschitz regularization
python -m src.train_Lipschitz_regularization

# Defensive Distillation
python -m src.train_deffensive_distilation
```

**Testing**

To test the saved models (or the ones you trained) you need to run the following command:
```
cd Medical-Image-Diagnosis
python -m test_application.test_<attack>_accuracy
```

> [!IMPORTANT] 
> For the directory that contains the test images you  just need to provide the ROOT of the file.

**Example**
```
# One-Pixel attack
python -m test_application.test_One_Pixel_attack_accuracy

# FGSM attack
python -m python -m test_application.test_FGSM_attack_accuracy
```

## Performance

**MobileNetV2 Baseline – Test Performance**

For our experiments, we observed that the training curves tend to plateau around epoch 25.
The baseline model registered very good results results on clean data:

| **Loss** | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|---------:|-------------:|--------------:|-----------:|-------------:|
|  0.6579  |     82.2%    |     86.2%     |    77.1%   |     78.8%    |

<img width="520" height="402" alt="roc_curve_mobilenetv2" src="https://github.com/user-attachments/assets/6d3a2f20-acc9-4283-9766-67c3521a4fb0" />

 > Figure: ROC Curve for MobileNetV2 (AUC = 0.93).

## First pair of attack-defense - One-Pixel attack with Lipschitz regularization

**MobileNetV2 with Lipschitz Regularization – Test Performance**

First we computed the metrics for baseline model that was attacked using the One-Pixel method.

| **Parameters** | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|----------------|--------------|---------------|------------|--------------|
| population = 60  
iterations = 120  
F = 0.5  
CR = 0.9 | 0.82 | 0.85 | 0.77 | 0.78 |
| population = 200  
iterations = 120  
F = 0.7  
CR = 1 | 0.819 | 0.84 | 0.768 | 0.773 |
| population = 60  
iterations = 120  
F = 0.5  
CR = 1 (3 pixels) | 0.819 | 0.84 | 0.767 | 0.769 |

**Accuracy Improvements of  Lipschitz Regularization vs. Baseline (One-Pixel Attack)**
| **Parameters** | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|----------------|--------------|---------------|------------|--------------|
| population = 60  
iterations = 120  
F = 0.5  
CR = 0.9 | 0.819 | 0.82 | 0.75 | 0.76 |
| population = 200  
iterations = 120  
F = 0.7  
CR = 1 | 0.817 | 0.819 | 0.743 | 0.758 |
| population = 60  
iterations = 120  
F = 0.5  
CR = 1 (3 pixels) | 0.816 | 0.815 | 0.74 | 0.755 |

**Lipschitz improves robustness against One-Pixel (+4.6 pp accuracy).**

## Second pair of attack-defense - FGSM with Defensive Distillation

**MobileNetV2 Defensive Distillation model - Test Performance**

Improvements are largest at moderate noise (ϵ = 0.03), reaching over +23 pp. Higher temperature (T = 50) provides a small but consistent advantage over T = 35.

**Accuracy Improvements of Defensive Distillation vs. Baseline (FGSM Attack)**

| **ε**  | **Baseline Acc**  | **T = 35 Acc** |   **Imp.**    | **T = 50 Acc** |   **Imp.**    |
|--------|------------------:|---------------:|--------------:|---------------:|--------------:|
|  0.03  |      28.7%        |     50.8%      |   +22.1 pp    |     52.2%      |   +23.5 pp    |
|  0.05  |      25.6%        |     41.2%      |   +15.6 pp    |     42.1%      |   +16.5 pp    |
|  0.08  |      25.6%        |     38.8%      |   +13.2 pp    |     39.1%      |   +13.5 pp    |

Both distilled models show slower accuracy degradation across all perturbation levels.

<img width="1445" height="362" alt="comparatie " src="https://github.com/user-attachments/assets/2919d9b8-13ab-4018-8676-f39e2b591f6a" />

  > Figure: performance comparison to different perturbation values.

## Second pair of attack-defense - FGSM with Lipschitz regularization ( just for fun :) )

**Accuracy Improvements of  Lipschitz Regularization vs. Baseline (FGSM Attack)**

We also tried to cross-validate the model trained with Lipschitz Regularization vs the Baseline model under FGSM attack.
| **ε**  | **Baseline Acc**  |   Lipschitz   |   **Imp.**    | 
|--------|------------------:|---------------:|--------------:|
|  0.03  |      28.7%        |     45.7%      |   +16.7 pp    |   
|  0.05  |      25.6%        |     36.7%      |   +11.1 pp    |     
|  0.08  |      25.6%        |     30.3%      |   +4.7 pp     |     

The model trained with Lipschitz Regularization is clearly more robust than the baseline, showing consistent accuracy improvements under FGSM attack across all tested ε values.

<img width="1637" height="562" alt="lip_fgms" src="https://github.com/user-attachments/assets/7470f8a8-502e-4bad-b042-dde07b207701" />

   > Figure: performance comparison to different perturbation values.

## Limitation

- Only two adversarial attacks were tested (FGSM, One-Pixel).  
- Future work: PGD, DeepFool, CW attack.

## Acknoledgments 
Contribuitors: Manolache Arianna, Stroe Teodora-Simina

