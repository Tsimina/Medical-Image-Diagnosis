# Medical-Image-Diagnosis 

Deep-learning pipeline for **chest X-ray diagnosis** and **adversarial robustness** study.

This project trains a **MobileNetV2** classifier on the Kaggle *Chest X-Ray Images (Pneumonia)* dataset and evaluates how vulnerable it is to adversarial attacks such as **FGSM** and **One-Pixel**. It also implements two defence strategies:

- **Lipschitz regularization**
- **Defensive distillation** (with different temperatures)

The goal is to compare the **baseline model** vs **defended models** on clean and adversarial examples.

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
├── results/                 # Saved weights, metrics logs, ROC curves, evaluation attckk plots, etc.
├── src/                     # Training & evaluation scripts (baseline + defences)
├── test_application/        # Simple tests for the saved models
|   ├── saved_models         # Saved model configs for baseline and defence  
├── utils/                   # Metrics, training helpers, distillation utilities
│
├── requirments.txt          # Python dependencies (pip install -r)
├── LICENSE                  # MIT License
└── (this) README.md
```

## Dataset 

Our MobileNetV2 was traint on the Kaggle *Chest X-Ray Images (Pneumonia)*. You can download the dataset here: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). 

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
      $x′ = x + δ(i,j)$
where δ(i,j) modifies a single pixel at position (i, j).

### FGSM 

A white-box attack that uses the model’s gradients to find the direction that increases
the loss the most.
Creates an adversarial example by adding a small, targeted perturbation to the input
image:
     $x′ = x + ϵ · sign(∇x J(x, y ))$
The parameter ϵ controls the strength of the perturbation.
Called “sign” because it uses only the sign of the gradient (+1 or −1) to decide the
direction of each pixel change.

## Model Details

This repository contains 3 models:
- Baseline MobileNetV2
- MobileNetV2 with Lipschitz Regularization
- MobileNetV2 Deffensive Distilation model
  
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

### MobileNetV2 Deffensive Distilation model
The Baseline MobileNetV2 serves as the teacher model.
  - Distillation temperatures: T=35$ and T=50
  - Mixing coefficient: alpha = 0.9 (focus on soft-label learning).
  - Training: 25 epochs, Adam (LR = 0.001), batch sizes 8 (train) / 16 (eval).

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
To train your own model you just need to update with the paths to your dataset and results directory in the *_config.py files, depending on the target you want to achive you can also change the hyperparameters. 
- DATA_DIR – path to your chest X-ray dataset
- RESULTS_DIR – where to save logs, models, plots
- BASELINE_WEIGHTS, DISTILLED_WEIGHTS
- Training hyperparameters: EPOCHS_BASELINE, EPOCHS_DISTILL, BATCH_TRAIN, BATCH_EVAL, LR
- Distillation: T (temperature), ALPHA
  
```
cd Medical-Image-Diagnosis
python -m src.train_<model_name>
```

## Testing
To test the saved models (or the ones you trained) you need to run the following command:
```
cd Medical-Image-Diagnosis
python -m test_application.test_<model_name>_accuracy
```

## Perfromance

### MobileNetV2 Baseline – Test Performance

For our experiments, we observed that the training curves tend to plateau around epoch 25.
The baseline model registerd very good results results on clean data:

| **Loss** | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|---------:|-------------:|--------------:|-----------:|-------------:|
|  0.6579  |    82.2%     |     86.2%     |   77.1%    |    78.8%     |

<img width="520" height="402" alt="roc_curve_mobilenetv2" src="https://github.com/user-attachments/assets/6d3a2f20-acc9-4283-9766-67c3521a4fb0" />
 > Figure: ROC Curve for MobileNetV2 (AUC = 0.93).

### MobileNetV2 with Lipschitz Regularization – Test Performance


### MobileNetV2 Deffensive Distilation model

Improvements are largest at moderate noise (ϵ = 0.03), reaching over +23 pp. Higher temperature (T = 50) provides a small but consistent advantage over T = 35.

### Accuracy Improvements of Defensive Distillation vs. Baseline (FGSM Attack)

| **ε**  | **Baseline Acc**  | **T = 35 Acc** |   **Imp.**    | **T = 50 Acc** |   **Imp.**    |
|--------|------------------:|---------------:|--------------:|---------------:|--------------:|
|  0.03  |      28.7%        |     50.8%      |   +22.1 pp    |     52.2%      |   +23.5 pp    |
|  0.05  |      25.6%        |     41.2%      |   +15.6 pp    |     42.1%      |   +16.5 pp    |
|  0.08  |      25.6%        |     38.8%      |   +13.2 pp    |     39.1%      |   +13.5 pp    |

Both distilled models show slower accuracy degradation across all perturbation levels.

<img width="1445" height="362" alt="comparatie " src="https://github.com/user-attachments/assets/2919d9b8-13ab-4018-8676-f39e2b591f6a" />
  > Figure: Perfromance comaprison to different perturbation values.

## Acknoledgments 
Contribuitors: Manolache Arianna, Stroe Teodora-Simina

