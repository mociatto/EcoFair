# Feature Extraction

This folder contains the notebooks that produce the image embeddings required by the main EcoFair pipeline. Run them **before** running any of the main notebooks at the repo root (`ecofair-ham10k.ipynb`, `ecofair-pad-ufes-20.ipynb`, `ecofair-bcn20k.ipynb`).

## Notebooks

| Notebook | Encoder | Output |
|----------|---------|--------|
| `image-feature-extractor-lite.ipynb` | MobileNetV3Small | `features.npy`, `ids.npy` per dataset |
| `image-feature-extractor-heavy.ipynb` | ResNet50 | `features.npy`, `ids.npy` per dataset |

## How to use

1. **Per dataset:** Point each notebook at the dataset's image directory and run it. Each run writes a **directory** (e.g. `output/HAM10000/MobileNetV3Small/`) containing:
   - `features.npy` – shape `(n_images, embedding_dim)`
   - `ids.npy` – image identifiers (filenames or IDs) in the same order as the rows in `features.npy`

2. **Paths:** The main notebooks expect the paths to these output directories. Set `LITE_DIR` and `HEAVY_DIR` in the main notebook to the directories produced here (e.g. `.../output/BCN20000/MobileNetV3Small` and `.../output/BCN20000/ResNet50`).

3. **Order:** Run both extractors for a given dataset so that the main pipeline can load both lite and heavy features and align them with metadata via the shared IDs.

Without these outputs, the main framework has no image features to train or evaluate on.
