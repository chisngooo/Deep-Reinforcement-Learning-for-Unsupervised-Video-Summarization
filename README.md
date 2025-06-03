# Deep Reinforcement Learning for Unsupervised Video Summarization

An advanced video summarization system powered by Deep Reinforcement Learning with Diversity-Representativeness reward mechanism. This implementation is based on the AAAI'18 research paper: "Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward".

## Table of Contents

- [Overview](#overview)
- [Trained Models](#trained-models)
- [System Requirements](#system-requirements)
- [Installation and Setup](#installation-and-setup)
- [Dataset Configuration](#dataset-configuration)
- [Model Training](#model-training)
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

## Web Application

### Launch Streamlit Interface

```bash
streamlit run streamlit_app.py
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
├── streamlit_app.py           # Web application interface
├── run_experiments.sh         # Automated unsupervised training
├── run_supervised.sh          # Automated supervised training
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
