# Deterministic U-Net PoC for Retinal Vessel Segmentation

Proof-of-Concept (PoC) implementation of the core deterministic **U-Net** segmentation backbone on the multi-pathology **FIVES** dataset, developed and executed in **Google Colab**. This repository serves as the validated baseline for a modular retinal vessel segmentation pipeline with uncertainty estimation (designed for future integration of uncertainty estimation methods such as MC Dropout, Deep Ensembles, TTA, and Ensemble Distillation).

## Overview
- **Architecture:** 4-level U-Net (`base_ch=32`, `BatchNorm2d`, bilinear upsampling)
- **Input Resolution:** `768 × 768` px (downscaled from `2048 × 2048` px using `INTER_AREA` for images and `INTER_NEAREST` for masks)
- **Data Split:** Stratified 90/10 train/validation split (`540 / 60` images, `SEED = 881`) + official test set (`200` images: 50 Normal, 50 Diabetic Retinopathy, 50 Glaucoma, 50 AMD)
- **Training:** `25` epochs, `BCEDiceLoss` (`0.5 * BCE + 0.5 * soft Dice`), `AdamW` (`lr=1e-3`, `weight_decay=1e-4`), `CosineAnnealingLR`, batch size `4`, FP16 mixed precision (`AMP`), random flips & 90° rotations

## How to Run in Google Colab

1. **Download the FIVES Dataset**  
   Download the official **FIVES** (*Fundus Image Vessel Segmentation*) dataset (Jin et al., *Scientific Data*, 2022 — DOI: `10.1038/s41597-022-01564-3`).

2. **Upload to Google Drive**  
   Make sure the archive is named **exactly** `FIVES.zip` and upload it to the root folder of your Google Drive (`My Drive`):
   ```text
   MyDrive/FIVES.zip
   ```
   *(The notebook expects `ZIP_PATH = "/content/drive/MyDrive/FIVES.zip"` and automatically locates `train/Original` and `test/Original` after extraction.)*

3. **Run the Notebook**  
   - Open `unet_poc.ipynb` in **Google Colab**.
   - Enable GPU acceleration: **Runtime → Change runtime type → T4 GPU**.
   - Run all cells (**Runtime → Run all**) and grant Google Drive access when prompted.

## Results & Generated Outputs
Summary evaluation tables (**Overall Test Performance** and **Performance by Clinical Category** across Dice, IoU, Accuracy, Sensitivity, and Specificity) as well as the visual plots are generated and displayed directly inside the evaluation cells of `unet_poc.ipynb`.

Running the notebook also saves the following files in the Colab workspace:
- `best_unet.pth` — best model checkpoint based on validation Dice
- `results/training_curves.png` — training/validation loss and validation Dice curves
- `results/test_metrics.csv` — per-image evaluation metrics across all 200 test images
- `results/fives_unet_comparison.png` — qualitative comparison (input, ground truth, U-Net prediction)
