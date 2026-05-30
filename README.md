# Russian Toxic Comments Classification

A comparative study of classical machine learning, transformer-based models, and parameter-efficient fine-tuning methods for Russian toxic comment classification.

## Project Overview

This project investigates the task of automatic toxic comment detection in Russian. Several approaches were implemented and compared:

* TF-IDF + Logistic Regression (baseline)
* ruBERT-tiny
* DeepPavlov ruBERT-base + LoRA
* DeepPavlov ruBERT-base + AdaLoRA

The goal was to evaluate how parameter-efficient fine-tuning methods perform compared to both classical machine learning approaches and full transformer models.

## Dataset

The experiments were conducted on the Russian Language Toxic Comments dataset available on Kaggle:

https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments

Dataset statistics:

* Total comments: 14,412
* Toxic comments: 4,826
* Non-toxic comments: 9,586
* Train samples: 11,528
* Validation samples: 1,442
* Test samples: 1,442

## Models

### 1. TF-IDF + Logistic Regression

Classical machine learning baseline using sparse TF-IDF text representations and Logistic Regression classifier.

### 2. ruBERT-tiny

A lightweight transformer model for Russian NLP tasks. Used as a compact transformer baseline.

### 3. LoRA

Parameter-Efficient Fine-Tuning (PEFT) of DeepPavlov ruBERT-base using Low-Rank Adaptation.

Configuration:

* Rank (r): 32
* Alpha: 64
* Dropout: 0.05
* Target modules:

  * query
  * key
  * value
  * dense

Trainable parameters:

* 5.36M trainable
* 183.21M total
* 2.93% trainable parameters

### 4. AdaLoRA

Adaptive version of LoRA that dynamically redistributes rank budget during training.

Configuration:

* Initial rank: 32
* Target rank: 16
* Beta1: 0.85
* Beta2: 0.85
* t_init: 200
* t_final: 1000
* DeltaT: 10

Trainable parameters:

* 10.72M trainable
* 188.57M total
* 5.68% trainable parameters

## Results

| Model                           | Toxic F1 | Macro F1 |
|---------------------------------| -------: | -------: |
| TF-IDF + Logistic Regression    |     0.78 |     0.84 |
| DeepPavlov Fine-Tuning (Kaggle) |     0.83 |     0.88 |
| **DeepPavlov ruBERT + LoRA**    |     0.87 |     0.90 |
| **ruBERT-tiny**                 |     0.88 |     0.91 |
| **DeepPavlov ruBERT + AdaLoRA** | **0.89** | **0.92** |
| ruBERT-Toxic (VK)               |    0.922 |        — |

## Key Findings

* Transformer-based models significantly outperform classical machine learning approaches.
* ruBERT-tiny provides a strong balance between quality and computational efficiency.
* LoRA achieves competitive results while training only a small fraction of model parameters.
* AdaLoRA achieved the best performance among the models trained in this project.
* Parameter-efficient fine-tuning can approach state-of-the-art performance without full model fine-tuning.


## Technologies

* Python
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* PEFT
* DeepPavlov
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Future Work

* Larger Russian language models
* Additional PEFT methods
* Hyperparameter optimization
* Cross-dataset evaluation
* Ensemble approaches
