# SPAI: Spectral AI-Generated Image Detector

Official repository for the CVPR 2025 paper:
Any-Resolution AI-Generated Image Detection by Spectral Learning.

SPAI learns the spectral distribution of real images and detects AI-generated
images as out-of-distribution samples using spectral reconstruction similarity.

## Repository Status

This repository currently contains:

- Core SPAI package in `spai/`.
- Main config in `configs/spai.yaml`.
- A trained checkpoint in `spai/weights/spai.pth`.
- Unit tests in `tests/`.
- Utility scripts for data prep, crawling, Fourier analysis and reporting in `tools/` and `spai/tools/`.
- Hugging Face inference handler in `inference.py`.

## Project Structure

```text
.
├── configs/
│   └── spai.yaml
├── spai/
│   ├── data/                 # datasets, readers, augmentations, filestorage (LMDB)
│   ├── models/               # backbones, SID, MFM, losses, filters
│   ├── tools/                # CSV generation and dataset utilities
│   ├── weights/
│   │   └── spai.pth          # included checkpoint
│   ├── config.py             # yacs configuration
│   ├── hf_utils.py           # Hugging Face Hub upload/model card helpers
│   ├── main_mfm.py           # MFM pretraining entrypoint
│   └── ...
├── tests/
│   ├── data/
│   └── models/
├── tools/                    # analysis, crawling, preprocessing, HF execution logs
├── inference.py              # HF EndpointHandler + local single-image inference
└── requirements.txt
```

## Installation

Recommended environment:

```bash
conda create -n spai python=3.11
conda activate spai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

Notes:

- Training code may require NVIDIA APEX.
- `requirements.txt` includes packages for training, inference, ONNX, crawling and Hugging Face utilities.

## Configuration and Weights

- Main config: `configs/spai.yaml`
- Default included checkpoint: `spai/weights/spai.pth`

The included config is set for SID finetuning/inference with arbitrary-resolution processing.

## Inference

### 1) Hugging Face Endpoint Handler (recommended)

File: `inference.py`

The `EndpointHandler` supports these input formats:

- image URL (`http://...` / `https://...`)
- local path
- base64 string
- raw bytes
- PIL image
- dict with one of keys: `url`, `path`, `b64`, `bytes`

Output format:

```json
{
  "score": 0.8732,
  "predicted_label": 1,
  "predicted_label_name": "ai-generated",
  "threshold": 0.5
}
```

Label convention used in repository tooling:

- `0` -> real
- `1` -> ai-generated

Run locally:

```bash
python inference.py --image "/path/to/image.jpg" --model-dir .
```

Environment overrides:

- `SPAI_THRESHOLD` (default `0.5`)
- `SPAI_CONFIG` (custom config path)
- `SPAI_CHECKPOINT` (local path or HF ref in the form `repo_id::path/in/repo`)
- `SPAI_FORCE_CPU=1` (force CPU)

### 2) Python usage

```python
from inference import EndpointHandler

handler = EndpointHandler(path=".")
result = handler({"inputs": "https://example.com/image.jpg"})
print(result)
```

## Training

### MFM pretraining entrypoint

Use:

```bash
python spai/main_mfm.py --cfg configs/spai.yaml --data-path /path/to/data.csv --output output/mfm
```

`spai/main_mfm.py` also supports optional Hugging Face push flags:

- `--push-to-hub`
- `--hub-repo-id`
- `--hub-token`
- `--hub-create-model-card`

## Dataset CSV Format

Core dataset readers in `spai/data/data_finetune.py` expect CSVs with at least:

- `image`: image path
- `class`: class id
- `split`: one of `train`, `val`, `test`

Paths are resolved relatively to a configurable CSV root directory.

## LMDB Dataset File Storage

Module: `spai/data/filestorage.py`

Available commands:

```bash
python spai/data/filestorage.py add-csv --help
python spai/data/filestorage.py add-db --help
python spai/data/filestorage.py verify-csv --help
python spai/data/filestorage.py list-db --help
```

Use this workflow when you want to package many files into LMDB for faster or centralized IO.

## Utility Scripts

### Repository-level tools (`tools/`)

- `tools/simple_crawler.py`: crawl and download images with metadata.
- `tools/web_image_crawler.py`: crawl URLs/CSVs, download images, filter ad-like images.
- `tools/image_quality_processor.py`: quality filtering, deduplication and reports.
- `tools/preprocess_for_spai.py`: image preprocessing before SPAI.
- `tools/create_spai_metadata.py`: build metadata CSV from an image folder.
- `tools/extract_fourier_features.py`: compute Fourier-derived features.
- `tools/visualize_fourier.py`: Fourier spectrum visualizations.
- `tools/visualize_noise_decomposition.py`: advanced noise decomposition visualizations.
- `tools/analyze_spai_results.py`: plots/analysis for prediction results.
- `tools/analyze_normalization_impact.py`: study resize normalization impact.
- `tools/hf_log_execution.py`: generate execution artifacts and optionally upload to HF datasets.

Example:

```bash
python tools/hf_log_execution.py --results-csv output/preds.csv --output-dir output/hf_artifacts
```

### Package tools (`spai/tools/`)

- `spai.tools.create_dir_csv`: create train/val/test CSV from directories.
- `spai.tools.create_dmid_ldm_train_val_csv`: create DMID/LDM training CSV.
- `spai.tools.augment_dataset`: augment a dataset and export updated CSV.
- `spai.tools.reduce_csv_column`: conditional column reduction/aggregation.
- `spai/tools/create_synthbuster_csv.py`: Synthbuster CSV generation utility.

Examples:

```bash
python -m spai.tools.create_dir_csv --help
python -m spai.tools.create_dmid_ldm_train_val_csv --help
python -m spai.tools.augment_dataset --help
python -m spai.tools.reduce_csv_column --help
```

For `create_synthbuster_csv.py`, use a `PYTHONPATH` that includes `spai/` due its import style:

```bash
PYTHONPATH=spai python spai/tools/create_synthbuster_csv.py --help
```

## Tests

Run all tests:

```bash
pytest tests -q
```

Current test folders:

- `tests/data/`
- `tests/models/`

## Acknowledgments

This work was partly supported by Horizon Europe projects ELIAS and vera.ai,
and computational resources from GRNET.

Parts of the implementation build upon ideas/code from:
https://github.com/Jiahao000/MFM

## License

Source code is licensed under Apache 2.0.
Third-party datasets and dependencies keep their own licenses.

## Contact

For questions: d.karageorgiou@uva.nl

## Citation

```text
@inproceedings{karageorgiou2025any,
  title={Any-resolution ai-generated image detection by spectral learning},
  author={Karageorgiou, Dimitrios and Papadopoulos, Symeon and Kompatsiaris, Ioannis and Gavves, Efstratios},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18706--18717},
  year={2025}
}
```