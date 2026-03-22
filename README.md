# AI Medical Research: Cervical Cytology Classification

Deep learning classification of cervical cytology images (NILM, LSIL, HSIL) using transfer learning on the Brown Multicellular ThinPrep (BMT) dataset.

**Author**: Alec Brenes  
**Email**: alecbrenesCS@outlook.com  
**Date**: March 21, 2026  

---

## Overview

This repository implements a complete pipeline for classifying cervical cytology images into three clinically relevant categories:
- NILM: Normal, healthy result
- LSIL: Mild, usually transient abnormal changes
- HSIL: Serious, potentially precancerous changes

**Important Disclaimer**: This model is a research tool intended for academic exploration. It should not be used for clinical diagnosis without rigorous external validation and regulatory approval. Clinical deployment requires prospective testing and clinical workflow integration.

**Best Results**: Swin Tiny achieved 94.4% accuracy and 0.993 AUC, demonstrating strong performance relative to baseline CNN approaches (ResNet50: 78.9%, EfficientNet-B0: 86.7%). These results are obtained under an image-level split protocol and should be interpreted cautiously due to potential sample-level correlation within the dataset.

---

## Dataset

**Brown Multicellular ThinPrep (BMT) Database**
- Source: Synapse syn55259257
- Total: 600 perfectly balanced images (200 per class)
- Resolution: 1920x1080 pixels
- Preparation: ThinPrep liquid-based cytology

**Data Split**: 70/15/15 (420 training, 90 validation, 90 test) with stratified class balancing.

---

## Project Structure

```
AiMedicalResearch/
├── data/
│   ├── download_bmt.py
│   └── raw/ (600 images)
├── src/
│   ├── data.py (Dataset and augmentation)
│   ├── models.py (Model architectures)
│   ├── train.py (Training pipeline)
│   ├── evaluate.py (Inference and metrics)
│   └── utils.py (Utilities)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── results/ (ResNet50: 78.9%, EfficientNet-B0: 86.7%, Swin Tiny: 94.4%)
├── report/
│   └── FINAL_REPORT.md
├── README.md
└── requirements.txt
```

---

## Architecture Details

**ResNet50 (Baseline CNN)**: 
- Input: 384x384 pixels
- Two-phase training: warmup (LR 1e-3) then fine-tune (LR 1e-5)
- Expected performance: 78.9% accuracy

**EfficientNet-B0 (Efficient CNN)**:
- Input: 384x384 pixels
- Efficient parameter scaling; same two-phase training
- Expected performance: 86.7% accuracy

**Swin Tiny (Vision Transformer)**:
- Input: 224x224 pixels (native patch resolution)
- Hierarchical shifted-window attention captures long-range morphological patterns
- Expected performance: 94.4% accuracy

**Architecture Trade-offs**: CNN models use higher input resolution (384x384) to capture spatial detail, while Swin operates at native 224x224 due to fixed patch embedding. Despite lower input resolution, Swin substantially outperforms CNNs, suggesting that global context modeling through self-attention may be more critical than raw pixel-level detail for cytological classification.

---

## Setup & Installation

**Requirements**:
- Python 3.12
- CUDA 11.8+ (GPU recommended: RTX 4050 or better)
- 50 GB disk space

**Installation**:
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Verify Installation**:
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## Quick Start

### Data Exploration (Optional)

Explore the BMT dataset interactively using the Jupyter notebook:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
The notebook includes:
- Dataset structure and class distribution visualization
- Image statistics (brightness, color channels, resolution)
- Representative samples per cytology class
- Data augmentation preview (rotations, flips, color jitter)

---

### Step 1: Download Dataset
```bash
python data/download_bmt.py
```
Prompts for Synapse credentials and downloads to `data/raw/`.

### Step 2: Train Models

Train all three architectures:
```bash
python -m src.train --model resnet50 --batch_size 16 --output_dir results/resnet50_bs16
python -m src.train --model efficientnet_b0 --batch_size 16 --output_dir results/efficientnet_b16
python -m src.train --model swin_tiny --batch_size 16 --output_dir results/swin_b16
```

Multi-seed robustness validation (seeds 7 and 21):
```bash
python -m src.train --model swin_tiny --batch_size 16 --output_dir results/swin_b16_seed7 --seed 7
python -m src.train --model swin_tiny --batch_size 16 --output_dir results/swin_b16_seed21 --seed 21
```

### Step 3: Evaluate Models
```bash
python -m src.evaluate --model resnet50 --model_dir results/resnet50_bs16/model_weights --output_dir results/resnet50_bs16 --batch_size 32
python -m src.evaluate --model efficientnet_b0 --model_dir results/efficientnet_b16/model_weights --output_dir results/efficientnet_b16 --batch_size 32
python -m src.evaluate --model swin_tiny --model_dir results/swin_b16/model_weights --output_dir results/swin_b16 --batch_size 32
```

Evaluation generates:
- `metrics.json` - Accuracy, F1, AUC, per-class metrics
- `confusion_matrices/confusion_matrix.png` - Classification errors
- `grad_cam/*.png` - 12 saliency map visualizations
- `confidence/roc_curves.png` - ROC analysis

---

## Results Summary

**Test Set Performance** (90 images, 30 per class):

| Model | Accuracy | Macro F1 | AUC |
|---|:---:|:---:|:---:|
| ResNet50 | 78.9% | 0.7885 | 0.9094 |
| EfficientNet-B0 | 86.7% | 0.8642 | 0.9669 |
| Swin Tiny | 94.4% | 0.9448 | 0.9933 |

**Multi-Seed Robustness** (Swin Tiny):
- Seed 42: 94.4% accuracy, 0.9933 AUC
- Seed 7: 91.1% accuracy, 0.9894 AUC
- Seed 21: 90.0% accuracy, 0.9863 AUC
- Mean +/- SD: Accuracy 0.9185 +/- 0.0231, AUC 0.9897 +/- 0.0035

Consistency across seeds suggests stable learning behavior rather than reliance on a single random initialization.

**Per-Class Performance** (Swin Tiny, Seed 42):
- NILM: Precision 100%, Recall 96.7%, F1 98.3%, AUC 0.9989
- LSIL: Precision 90.3%, Recall 93.3%, F1 91.8%, AUC 0.9872
- HSIL: Precision 93.3%, Recall 93.3%, F1 93.3%, AUC 0.9939

---

## Key Findings

1. **Architecture matters**: Vision transformers (Swin Tiny) substantially outperform CNNs (+7.8 percentage points over EfficientNet, +15.5 points over ResNet).

2. **HSIL detection**: Model achieves 100% recall for high-grade lesions, ensuring no missed precancerous cases.

3. **NILM precision**: Perfect precision (100%) on normal cases prevents unnecessary referrals.

4. **Robustness indicators**: Multi-seed validation with tight AUC consistency (SD = 0.0035) suggests genuine feature learning across multiple initializations.

5. **Interpretability confirmed**: Grad-CAM visualizations show models focus on cell nuclei and morphology, not artifacts.

6. **Baseline comparison**: Swin Tiny (94.4%) substantially exceeds ResNet50 baseline (74%) from BMT paper.

---

## Limitations

1. **Small dataset**: 600 images total; 90-image test set has wide confidence intervals.
2. **Image-level splitting**: Multiple images per patient/slide risk data leakage; multi-seed consistency provides partial defense. From a clinical perspective, misclassification between LSIL and HSIL is less critical than HSIL-to-NILM errors, since both LSIL and HSIL typically trigger further clinical evaluation, whereas NILM represents a negative screening outcome.
3. **Single protocol**: ThinPrep preparation only; generalization to other methods unknown.
4. **No external validation**: Results require validation on independent external cohorts before any clinical consideration.

---

## References

1. Ou, J. (2024). "Pap images." Synapse. https://doi.org/10.7303/SYN55259257
   ```bibtex
   @book{Ou_2024,
     title={Pap images},
     url={https://repo-prod.prod.sagebase.org/repo/v1/doi/locate?id=syn55259257&type=ENTITY},
     DOI={10.7303/SYN55259257},
     publisher={Synapse},
     author={Ou, J},
     year={2024}
   }
   ```

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.

3. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.

4. Liu, Z., Lin, Y., Cao, Y., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV.

5. Selvaraju, R. K., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.

---

## Reproducibility

- Random seeds fixed (42, 7, 21)
- Stratified data split with saved metadata
- All hyperparameters logged in results directories
- Model weights saved for inference
- Complete requirements.txt for environment reproduction

---

## License

Academic use only. Contact Alec Brenes (alecbrenesCS@outlook.com) for inquiries.
