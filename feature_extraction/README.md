# Feature Extraction

This folder contains the notebooks that produce the image embeddings and **energy measurements** required by the main EcoFair pipeline. Energy is tracked across the whole workflow: these extractors measure and store joules per sample for each model and each dataset. That data is written to a JSON file in each output directory and is used by the main notebooks for energy-aware metrics and battery projections. Run the extractors **before** any main notebook at the repo root (`ecofair-ham10k.ipynb`, `ecofair-pad-ufes-20.ipynb`, `ecofair-bcn20k.ipynb`).

## Notebooks

Each notebook supports multiple backbones. You choose one per run; the main pipeline typically uses one lite and one heavy option (e.g. MobileNetV3Small and ResNet50).

| Notebook | Supported models | Output |
|----------|------------------|--------|
| `image-feature-extractor-lite.ipynb` | MobileNetV3Small, MobileNetV3Large, EfficientNetB0, MobileNetV2, NASNetMobile | `features.npy`, `ids.npy`, `energy_stats.json` per dataset |
| `image-feature-extractor-heavy.ipynb` | ResNet50, EfficientNetB6, ResNet152V2, DenseNet201, InceptionResNetV2 | `features.npy`, `ids.npy`, `energy_stats.json` per dataset |

## How to use

1. **Per dataset and model:** Point each notebook at the dataset's image directory and select the encoder. Each run writes an **output directory** (e.g. `output/HAM10000/MobileNetV3Small/`) containing:
   - `features.npy` – shape `(n_images, embedding_dim)`
   - `ids.npy` – image identifiers in the same order as the rows in `features.npy`
   - `energy_stats.json` – measured energy (joules per sample) for this model and dataset, used by the main pipeline for reporting and battery projections.

2. **Paths:** In the main notebooks, set `LITE_DIR` and `HEAVY_DIR` to the output directories you produced (e.g. `.../output/BCN20000/MobileNetV3Small` and `.../output/BCN20000/ResNet50`). The main pipeline reads the energy stats from those directories.

3. **Order:** Run both extractors for a given dataset so the main pipeline can load lite and heavy features and their energy data and align them with metadata via the shared IDs.

Without these outputs, the main framework has no image features or energy measurements to train and evaluate on.
