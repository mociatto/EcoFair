# EcoFair — Energy-Aware, Fairness-Conscious Skin Lesion Classification

EcoFair is a Vertical Federated Learning (VFL) framework for skin lesion diagnosis that dynamically routes each patient case between a lightweight and a heavyweight model. It targets three simultaneous objectives: clinical accuracy, energy efficiency, and demographic fairness — without requiring access to raw images at inference time.

![EcoFair Framework](fig/framework.png)

---

## System Design

The pipeline is split into a feature extraction stage (run once offline) and an inference stage (run per patient). Two image encoders — a lite MobileNetV3 and a heavy ResNet50 — extract embeddings independently. A tabular client processes patient metadata (age, sex, localization). A VFL server head fuses the embeddings and produces a diagnosis probability vector.

At inference, a three-tier router decides per sample whether the lite path alone is sufficient:

1. **Uncertainty gate** — entropy of the lite output exceeds a normalised threshold (proportion of theoretical maximum entropy for the number of classes).
2. **Ambiguity gate** — the gap between aggregated safe-class and dangerous-class probability falls below a margin.
3. **Safety gate** — the patient's neurosymbolic risk score (age × localization malignancy rate) exceeds a clinical safety threshold.

Only samples that trigger at least one gate are escalated to the heavy model.

![System Design](fig/system_design.png)

---

## Model Architecture

Both the lite and heavy branches share identical VFL architecture: an image adapter, a tabular client, and a shared server head. The two branches differ only in which image encoder's features they consume. The shared head learns a single joint embedding space, keeping the total parameter count low.

![Model Architecture](fig/models.png)

---

## Datasets

EcoFair is validated on three publicly available dermatology datasets representing distinct acquisition conditions.

### HAM10000
Part of the [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019). Dermoscopic images collected under controlled clinical conditions across multiple institutions. Seven classes: melanoma, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, melanocytic nevi, and vascular lesions. Structured tabular metadata (age, sex, localization) with high completeness.

> Tschandl P. et al. *The HAM10000 dataset.* Sci. Data 5, 180161 (2018).

### BCN20000
Also part of the [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019). Dermoscopic images captured in a real-world clinical setting at Hospital Clínic de Barcelona. Eight classes, delivered as separate train and test CSV files. Acquisition conditions are heterogeneous — variable lighting, patient positioning, and device calibration — making it a natural domain-shift benchmark relative to HAM10000.

> Hernández-Pérez C. et al. *BCN20000: Dermoscopic lesions in the wild.* Sci. Data 11, 641 (2024).

### PAD-UFES-20
Available at [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1). Clinical images collected with consumer smartphones at a public dermatology clinic in Brazil serving low-income patients. Six classes, approximately 58 % biopsy-proven. Images vary in resolution, focus, and illumination due to different devices. Rich tabular metadata including Fitzpatrick skin type and lesion diameter alongside age, sex, and localization. Represents a resource-constrained, real-world deployment scenario.

> Pacheco A. G. C. et al. *PAD-UFES-20.* Mendeley Data (2020).

---

## Notebooks

Each dataset has a self-contained Kaggle notebook. All three are structurally identical — only the top-level constants (paths, class lists) differ — and each runs the full EcoFair pipeline end to end.

| Notebook | Dataset | Classes |
|---|---|---|
| [`main/ecofair-ham10k.ipynb`](main/ecofair-ham10k.ipynb) | HAM10000 | 7 |
| [`main/ecofair-pad-ufes-20.ipynb`](main/ecofair-pad-ufes-20.ipynb) | PAD-UFES-20 | 6 |
| [`main/ecofair-bcn20k.ipynb`](main/ecofair-bcn20k.ipynb) | BCN20000 | 8 |

Each notebook produces:
- 5-fold cross-validation accuracy for Pure Lite, Pure Heavy, and EcoFair
- Per-fold routing rate and energy cost per sample
- Confusion matrix comparison across the three decision strategies
- Class-wise accuracy and value-added routing breakdown
- Age- and gender-stratified accuracy
- Risk-stratified accuracy and battery lifetime projection
- Fairness audit (Equal Opportunity TPR and Demographic Parity across age and sex subgroups)

---

## Repository Structure

```
src/
├── config.py          # Global hyperparameters only
├── data_loader.py     # Dataset-agnostic feature & metadata alignment
├── features.py        # Neurosymbolic risk scoring & tabular feature engineering
├── models.py          # VFL model architecture
├── training.py        # Cross-validation pipeline
├── routing.py         # Entropy, threshold, and budget routing logic
├── fairness.py        # Subgroup fairness metrics
├── visualization.py   # All plotting functions
└── utils.py           # Seed control, energy stats loading

main/
├── ecofair-ham10k.ipynb
├── ecofair-pad-ufes-20.ipynb
└── ecofair-bcn20k.ipynb

main_ham.py            # Front-end script — HAM10000
main_pad.py            # Front-end script — PAD-UFES-20
main_bcn.py            # Front-end script — BCN20000
```

The `src/` backend is fully dataset-agnostic. All dataset-specific configuration (paths, class names, safe/dangerous lists, required metadata columns) is defined exclusively in the front-end scripts and injected into backend functions via arguments.
