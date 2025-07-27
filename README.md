# POLYNET: Polyadenylation Site Prediction

## Overview
POLYNET is a deep learning project for predicting polyadenylation sites (PAS) in genomic sequences. It uses a convolutional neural network (CNN) to classify candidate nucleotide positions as true PAS or not, based on a fixed-length window of sequence data.

## Directory Structure
```
POLYNET/
├── src/
│   ├── train.py
│   ├── models/
│   │   └── POLYNET.py
│   ├── data/
│   │   ├── Pas_Dataset.py
│   │   └── processed/
│   │       ├── pos_201_train.fa
│   │       └── ...
│   └── utils/
│       └── encoding.py
├── scripts/
│   └── split_data.py
├── models/
│   └── POLYNET.pt
├── requirements.txt
├── README.md
└── notebooks/
```

## Setup
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd POLYNET
   ```
2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation
1. Place your raw FASTA files (e.g., `pos_201_hg19.fa`, `neg_201_hg19.fa`) in `src/data/`.
   - Note: These files were generated using PolyADB 3.0 and are sets of positive and negative examples for training. They are included as part of the repo for reproducibility purposes. 

2. Processed files in train/test/val splits are located in src/data/processed. 

## Training
Run the training script from the project root using the Python module flag:
```bash
python -m src.train
```
This will train the model, print training/validation metrics, and save the trained model to `models/POLYNET.pt`.

## Evaluation
After training, the script will automatically evaluate the model on the test set and print AUROC and AUPRC metrics.

## Notes
- Make sure to run all scripts from the project root directory so that relative imports and paths work correctly.
- You can modify hyperparameters (batch size, learning rate, epochs) in `src/train.py`.

## Requirements
See `requirements.txt` for a full list of dependencies.


