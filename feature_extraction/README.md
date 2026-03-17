# Feature Extraction

This folder contains two Jupyter notebooks that produce the image embeddings and hardware-measured model metrics required by the main EcoFair pipeline notebooks.

| Notebook | Role | Supported backbones |
|----------|------|---------------------|
| [`image-feature-extractor-lite.ipynb`](image-feature-extractor-lite.ipynb) | Lite CNN extraction (edge) | MobileNetV3Small, MobileNetV3Large, EfficientNetB0, MobileNetV2, NASNetMobile |
| [`image-feature-extractor-heavy.ipynb`](image-feature-extractor-heavy.ipynb) | Heavy CNN extraction (cloud) | ResNet50, EfficientNetB6, ResNet152V2, DenseNet201, InceptionResNetV2 |

Run both notebooks (e.g. on Kaggle) for each dataset before executing the pipeline notebooks. Each notebook iterates over the configured datasets and writes:

```text
output/<DATASET>/<MODEL>/
    features.npy          # shape (n_samples, embedding_dim)
    ids.npy               # image identifiers aligned with features.npy rows
    energy_stats.json     # model metrics and measured energy per sample
```

`energy_stats.json` contains: `model_name`, `dataset_name`, `resolution`, `num_samples`, `embedding_dim`, `model_parameters`, `model_size_mb`, `latency_per_sample_ms` (± std), `gpu_memory_footprint_mb`, `joules_per_sample`, and `total_energy_joules`. The main pipeline notebooks read these values directly for energy and capacity analyses.
