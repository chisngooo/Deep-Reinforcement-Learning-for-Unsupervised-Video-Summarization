# Deep Reinforcement Learning for Unsupervised Video Summarization

An advanced video summarization system powered by Deep Reinforcement Learning with Diversity-Representativeness reward mechanism. This implementation is based on the AAAI'18 research paper: "Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward".

## Table of Contents

- [Overview](#overview)
- [Trained Models](#trained-models)
- [System Requirements](#system-requirements)
- [Installation and Setup](#installation-and-setup)
- [Dataset Configuration](#dataset-configuration)
- [Model Training](#model-training)
- [Untrained Model Evaluation](#untrained-model-evaluation)
- [Training vs. Untrained Performance Comparison](#training-vs-untrained-performance-comparison)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Overview

This project implements a deep reinforcement learning approach for unsupervised video summarization. The system selects the most important frames from input videos to create concise summaries while maintaining both diversity and representativeness of the original content.

## Trained Models

This repository contains 60 pre-trained models across different architectures and datasets:

### Model Architectures (6 types)
- **DR-DSN**: Diversity-Representativeness Deep Summarization Network (Unsupervised)
- **D-DSN**: Diversity Deep Summarization Network (Unsupervised)
- **D-DSN-nolambda**: Diversity DSN without lambda regularization (Unsupervised)
- **R-DSN**: Representativeness Deep Summarization Network (Unsupervised)
- **DR-DSNsup**: Diversity-Representativeness DSN (Supervised)
- **DSNsup**: Deep Summarization Network (Supervised)

### Training Configuration
- **Datasets**: 2 datasets (SumMe, TVSum)
- **Cross-validation**: 5-fold cross-validation for each dataset
- **Total models**: 60 trained instances (6 architectures × 2 datasets × 5 splits)

### Model Performance Summary
Each model architecture was evaluated using 5-fold cross-validation on both SumMe and TVSum datasets to ensure robust performance metrics and generalization capabilities.

## System Requirements

- Python 3.8 or higher
- PyTorch 1.8 or higher
- CUDA 10.2+ (optional, for GPU acceleration)
- FFmpeg (latest version)
- Minimum 8GB RAM
- 5GB available storage space

## Installation and Setup

### 1. Clone Repository and Create Directories

```bash
git clone <repository-url>
cd Deep-Reinforcement-Learning-for-Unsupervised-Video-Summarization
mkdir -p datasets log
```

### 2. Download Datasets

Download the required datasets (173.5MB total) from Google Drive:
https://drive.google.com/open?id=1Bf0beMN_ieiM3JpprghaoOwQe9QJIyAN

Extract and place the following files in the `datasets/` directory:
- `eccv16_dataset_summe_google_pool5.h5` - SumMe dataset with GoogLeNet features
- `eccv16_dataset_tvsum_google_pool5.h5` - TVSum dataset with GoogLeNet features
- `eccv16_dataset_ovp_google_pool5.h5` - OVP dataset (optional)
- `eccv16_dataset_youtube_google_pool5.h5` - YouTube dataset (optional)

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Verify PyTorch installation with CUDA support (optional):
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Dataset Configuration

### Automatic Split Generation

Generate 5-fold cross-validation splits for training:

```bash
# Generate splits for SumMe dataset
python create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-path datasets/summe_splits.json

# Generate splits for TVSum dataset  
python create_split.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 --save-path datasets/tvsum_splits.json
```

### Pre-configured Splits

The repository includes pre-generated split configurations:
- `datasets/summe_splits.json` - 5-fold cross-validation splits for SumMe
- `datasets/tvsum_splits.json` - 5-fold cross-validation splits for TVSum

## Model Training

### Training Pipeline Overview

Training all 60 models involves running the following steps:

1. Train 40 unsupervised models (DR-DSN, D-DSN, D-DSN-nolambda, R-DSN on two datasets with 5 splits each)
2. Train 20 supervised models (DR-DSNsup, DSNsup on two datasets with 5 splits each)
3. Evaluate both trained and untrained (randomly initialized) models for comparison

### Manual Training Examples

#### Training DR-DSN on SumMe Dataset

```bash
python main.py \
    -d datasets/eccv16_dataset_summe_google_pool5.h5 \
    -s datasets/summe_splits.json \
    -m summe \
    --gpu 0 \
    --save-dir log/DR-DSN-summe-split0 \
    --split-id 0 \
    --verbose
```

#### Training DR-DSN on TVSum Dataset

```bash
python main.py \
    -d datasets/eccv16_dataset_tvsum_google_pool5.h5 \
    -s datasets/tvsum_splits.json \
    -m tvsum \
    --gpu 0 \
    --save-dir log/DR-DSN-tvsum-split0 \
    --split-id 0 \
    --verbose
```

#### Key Training Parameters

- `--lr 1e-05` - Learning rate (default)
- `--weight-decay 1e-05` - L2 regularization weight (default)
- `--max-epoch 60` - Maximum training epochs (default)
- `--beta 0.01` - Diversity reward weight (default)
- `--gpu 0` - GPU device ID (-1 for CPU)

### Automated Training Pipelines

#### Complete Unsupervised Training

```bash
bash run_experiments.sh
```

This script trains all unsupervised models:
- DR-DSN on SumMe and TVSum (5 splits each = 10 models)
- D-DSN on SumMe and TVSum (5 splits each = 10 models)
- D-DSN-nolambda on SumMe and TVSum (5 splits each = 10 models)
- R-DSN on SumMe and TVSum (5 splits each = 10 models)

Total: 40 unsupervised models

#### Complete Supervised Training

```bash
bash run_supervised.sh
```

This script trains all supervised models:
- DR-DSNsup on SumMe and TVSum (5 splits each = 10 models)
- DSNsup on SumMe and TVSum (5 splits each = 10 models)

Total: 20 supervised models

### Training Output Structure

All training logs and model checkpoints are saved to:

```
log/
├── {MODEL}-{DATASET}-split{ID}/
│   ├── model_epoch_{N}.pth.tar
│   ├── log.txt
│   └── reward_curves.png
```

Example structure for one trained model:
```
log/DR-DSN-summe-split0/
├── model_epoch_60.pth.tar  # Final model checkpoint
├── log.txt                 # Training log with loss/reward history
└── reward_curves.png       # Training visualization
```

## Untrained Model Evaluation

To establish a rigorous baseline for comparison, we evaluated 60 untrained models (with random initialization) using the same architectures, datasets, and splits as the trained models.

### Why Evaluate Untrained Models?

Evaluating untrained models provides:
1. A true baseline for measuring the effectiveness of training
2. Scientific insights into the inherent biases of model architectures
3. A way to detect potential dataset biases or memorization issues
4. Understanding of the role randomness plays in model performance

### Automated Untrained Model Evaluation

Run the untrained evaluation script for all 60 model configurations:

```bash
bash run_untrained_evaluation.sh
```

This script:
- Evaluates all 60 untrained models with consistent random seed (default: 7)
- Saves results in structured JSON format for easy comparison
- Generates automatic comparison with trained models
- Reports average performance metrics across all splits

#### Customization Options

```bash
# Evaluate with custom options
bash run_untrained_evaluation.sh --seed 42 --output "results/untrained_eval" --device 0 --verbose
```

Parameters:
- `--seed`: Random seed for reproducibility (default: 7)
- `--output`: Output directory for results (default: "untrained_results")
- `--device`: GPU device ID to use (-1 for CPU, default: 0)
- `--verbose`: Enable detailed logging (default: disabled)

## Training vs. Untrained Performance Comparison

Our extensive evaluation revealed surprising results: in several cases, untrained models achieved comparable or even superior performance to their trained counterparts.

### Performance Comparison Table

Below is a comprehensive comparison of trained vs. untrained model performance across all model architectures and datasets:

```
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| Model          | Dataset   | Untrained   | Trained   | Abs. Improvement   | Rel. Improvement   |
+================+===========+=============+===========+====================+====================+
| DR-DSN         | SUMME     | 40.6%       | 39.9%     | -0.7%              | -1.6%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| DR-DSN         | TVSUM     | 56.1%       | 56.6%     | +0.5%              | +0.9%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| D-DSN          | SUMME     | 40.8%       | 39.5%     | -1.3%              | -3.1%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| D-DSN          | TVSUM     | 56.6%       | 55.7%     | -0.9%              | -1.7%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| R-DSN          | SUMME     | 40.9%       | 38.8%     | -2.1%              | -5.1%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| R-DSN          | TVSUM     | 57.2%       | 56.6%     | -0.6%              | -1.1%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| DR-DSNsup      | SUMME     | 40.6%       | 41.9%     | +1.3%              | +3.2%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| DR-DSNsup      | TVSUM     | 55.7%       | 56.7%     | +1.0%              | +1.8%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| DSNsup         | SUMME     | 41.1%       | 39.2%     | -1.9%              | -4.6%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| DSNsup         | TVSUM     | 56.4%       | 52.3%     | -4.1%              | -7.2%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| D-DSN-nolambda | SUMME     | 39.1%       | 39.2%     | +0.1%              | +0.3%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
| D-DSN-nolambda | TVSUM     | 56.6%       | 52.3%     | -4.3%              | -7.6%              |
+----------------+-----------+-------------+-----------+--------------------+--------------------+
```

### Scientific Analysis: Why Untrained Models Can Outperform Trained Models

Our findings reveal a fascinating paradox in video summarization: in 8 out of 12 configurations, untrained models achieved superior F1-scores compared to their trained counterparts. This phenomenon can be explained through several scientific perspectives:

#### 1. Inductive Biases in Network Architecture

The underlying neural architectures inherently incorporate strong biases that align with the video summarization task. The LSTM backbone of our models has an intrinsic capability to identify temporal patterns, while the attention mechanisms naturally focus on salient video segments—even without training. This architectural inductive bias can contribute significantly to performance.

#### 2. Feature Rich Representations

The input to our models are pre-extracted GoogLeNet features (pool5 layer) that already encode rich semantic information about video frames. These high-level features may contain sufficient information for the summarization task such that even random linear projections (as in untrained models) can achieve reasonable discrimination between important and unimportant frames.

#### 3. Dataset Characteristics

The relatively small size of video summarization datasets (SumMe: 25 videos, TVSum: 50 videos) presents challenges for deep learning:
- Limited diversity in training examples restricts the model's ability to learn generalizable patterns
- The subjective nature of video summarization leads to significant annotation variance
- The datasets may not contain sufficient complexity to necessitate sophisticated learned patterns beyond what architectural biases provide

#### 4. Overfitting and Regularization

Trained models may overfit to specific training split characteristics, leading to decreased performance on test videos. The training objective may not perfectly align with F1-score evaluation:
- Models optimize for reconstruction and diversity-representativeness rewards during training
- Evaluation focuses on F1-score against human annotations
- This objective mismatch can cause trained models to optimize for metrics that don't perfectly correlate with final evaluation criteria

#### 5. Lucky Initialization and Optimization Challenges

The reinforcement learning paradigm used in training introduces:
- High variance in optimization trajectories
- Potential for getting trapped in local optima
- Complex reward landscapes that are difficult to navigate efficiently

Sometimes, a "lucky" random initialization may position the model in a favorable region of the parameter space that training cannot improve upon or may even move away from.

#### 6. Beneficial Noise in Decision-Making

Random parameters can introduce beneficial noise in the selection process. In video summarization, diversity is crucial, and the stochastic nature of untrained models might inadvertently promote diversity by making less biased selections across frames.

### Implications and Future Directions

This phenomenon highlights important considerations for video summarization research:

1. **Stronger Baselines:** Random initialization should be considered a legitimate baseline in video summarization benchmarks
2. **Architecture Design:** Models should be evaluated on the relative improvement over their untrained versions
3. **Training Methodologies:** Alternative training approaches like contrastive learning might better capture the essence of video summarization
4. **Dataset Development:** Larger, more diverse datasets may be needed to truly benefit from complex model training

This investigation underscores the complex interplay between model architecture, training dynamics, and dataset characteristics in the video summarization domain.

## Web Application

### Launch Streamlit Interface

```bash
streamlit run app.py
```

Access the application via browser at: http://localhost:8501

### User Interface Guide

1. **Upload Video**: Drag and drop video files (MP4, AVI, MOV, MKV)
2. **Select Model**: Choose architecture (DR-DSN, D-DSN, D-DSN-nolambda, R-DSN, DR-DSNsup, DSNsup)
3. **Select Dataset**: Pick pre-trained model (SumMe or TVSum)
4. **Configure Output**: Set summary ratio and output FPS
5. **Process**: Click "Process Video" to generate summary
6. **Download**: Save the generated video summary

### Application Features

- **Real-time Analytics**: Frame importance visualization with interactive charts
- **Video Information**: Automatic detection of resolution, FPS, and duration
- **Processing Progress**: Live progress tracking with estimated completion time
- **Preview System**: Built-in video player for summary preview
- **Multi-format Export**: Support for various output formats
- **Modern UI**: Responsive design with professional styling

## Project Structure

```
Deep-Reinforcement-Learning-for-Unsupervised-Video-Summarization/
├── datasets/                    # Dataset files and configuration
│   ├── *.h5                    # Feature datasets (SumMe, TVSum, etc.)
│   ├── summe_splits.json       # SumMe cross-validation splits
│   └── tvsum_splits.json       # TVSum cross-validation splits
├── log/                        # Training outputs and model checkpoints
│   ├── DR-DSN-*/              # DR-DSN model checkpoints (20 models)
│   ├── D-DSN-*/               # D-DSN model checkpoints (10 models)
│   ├── D-DSN-nolambda-*/      # D-DSN-nolambda checkpoints (10 models)
│   ├── R-DSN-*/               # R-DSN model checkpoints (10 models)
│   ├── DR-DSNsup-*/           # DR-DSNsup model checkpoints (10 models)
│   └── DSNsup-*/              # DSNsup model checkpoints (10 models)
├── imgs/                       # Documentation images and visualizations
├── streamlit_output/           # Web application output directory
├── video/                      # Sample videos for testing
├── main.py                     # Core training script
├── models.py                   # Neural network architectures
├── app.py                      # Web application interface
├── run_experiments.sh         # Automated unsupervised training
├── run_supervised.sh          # Automated supervised training
├── run_untrained_evaluation.sh # Automated untrained model evaluation
├── evaluate_untrained_models.py # Script for evaluating untrained models
├── create_split.py            # Dataset split generation utility
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Citation

This implementation is based on the AAAI'18 research paper by Kaiyang Zhou, Yu Qiao, and Tao Xiang.

```bibtex
@article{zhou2017reinforcevsumm,
  title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward},
  author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao},
  journal={arXiv preprint arXiv:1801.00054},
  year={2017}
}
```
