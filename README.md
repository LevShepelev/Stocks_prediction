# 📈 Stock Prediction Project

This repository contains an **end‑to‑end educational pipeline** for time‑series
forecasting of Russian stocks using deep learning (LSTM) with a modern MLOps
stack:

- **PyTorch + PyTorch Lightning** for model authoring and training
- **Hydra & OmegaConf** for hierarchical configuration management
- **ONNX + TensorRT** for portable, high‑performance inference
- **NVIDIA Triton Inference Server** to serve the model in production‐like
  conditions
- **Poetry** for dependency management and packaging
- **pre‑commit** (Black + isort + flake8 + mypy) for code quality

---

## 🗂️ Repository Structure

```text
.
├── data/                 # Raw & processed datasets (⛔ Git‑ignored)
├── exports/
│   └── onnx/             # Exported ONNX models
├── model_repo/           # Triton model repository (⛔ Git‑ignored)
├── notebooks/            # Exploration & prototyping notebooks
├── stocks_prediction/
│   ├── conf/             # Hydra configs (train_config.yaml, …)
│   ├── datamodules/      # LightningDataModule implementations
│   ├── models/           # LSTM/GRU model definitions
│   ├── train_loop.py     # ⚙️ Main training script (Hydra‑aware)
│   ├── predict_and_plot.py
│   └── …
├── convert_and_deploy_triton.py   # ONNX → TensorRT → Triton helper
├── Dockerfile.triton              # (optional) Build a custom Triton image
├── pyproject.toml
└── README.md                      # ← you are here
```

---

## 🏗️ Setup

> **Prerequisites**
>
> - Python ³.11 with CUDA‑enabled GPU & drivers (for training / TRT)
> - Docker ≥ 24 with NVIDIA Container Runtime
> - (Optional) CUDA / cuDNN locally for JIT previews

```bash
# 1. Clone & install deps
git clone https://github.com/your‑org/Stocks_prediction.git
cd Stocks_prediction

# 2. Install poetry & project packages
curl -sSL https://install.python-poetry.org | python -
poetry install --with dev

# 3. Activate virtual‑env
poetry shell

# 4. Set environment variables (API keys, tracking URLs…)
cp .env.example .env  # then edit values
```
⬇️ Download Data with DVC

This project uses DVC to manage and version the raw OHLCV datasets.

# Ensure DVC is installed in your environment
poetry run pip install dvc

# Configure your DVC remote (only first time):
# e.g., an S3 bucket, Azure Blob, Google Drive, etc.
# dvc remote add -d storage <remote-url>

# Pull the raw and processed data files
poetry run dvc pull -r storage data/

Note: Replace <remote-url> and storage with your DVC remote name and URL as specified by your project maintainers.


## 🚂 Training

Hydra drives the pipeline; all hyper‑parameters live in
**`stocks_prediction/conf/`**.

```bash
# Train with the default config
poetry run python stocks_prediction/train_loop.py
# Log files, checkpoints & TensorBoard go to ./outputs/<DATE_TIME>/
```

### 🔄 Resume or Fine‑Tune

```bash
poetry run python stocks_prediction/train_loop.py \
  +checkpoint_path=outputs/2025‑05‑31/15‑42‑10/checkpoints/epoch=04.ckpt
```

---

## 📤 Export to ONNX

Training script already exports the **best checkpoint** to ONNX
(`exports/onnx/`).

---

## ⚙️ Convert to TensorRT & Deploy to Triton

```bash
# 1️⃣ Convert ONNX → TensorRT engine and generate Triton directory
poetry run python convert_and_deploy_triton.py \
  --onnx exports/onnx/stocklstm_GAZP_10m_2002‑01‑01_2024‑05‑01.onnx \
  --repo-dir ../model_repo/

# The script creates
#   model_repo/
#       stocklstm/
#           1/model.plan   (TensorRT engine)
#           config.pbtxt   (I/O schema)
```

---

## 🚀 Run Triton Inference Server

```bash
# Pull NVIDIA Triton image (≈3 GB)
docker pull nvcr.io/nvidia/tritonserver:25.04-py3

# Launch server mapping HTTP :8000, gRPC :8001, metrics :8002
# (replace $(pwd)/../model_repo with your absolute path)
docker run --rm --gpus=all --name triton_server \
  -p8011:8000 -p8012:8001 -p8013:8002 \
  -v "$(pwd)/../model_repo:/models" \
  nvcr.io/nvidia/tritonserver:25.04-py3 \
  tritonserver --model-repository=/models
```

> Visit **`http://localhost:8011/v2/health/ready`** ⇒ `OK` when ready.

### 🐚 Shell script helper

A reusable helper to (re)start Triton locally is provided in
**`scripts/start_triton.sh`**:

```bash
#!/usr/bin/env bash
# scripts/start_triton.sh
set -euo pipefail

# Absolute path to repo root
ROOT_DIR="$(git rev-parse --show-toplevel)"

# Default ports can be overridden via env vars
HTTP_PORT="${HTTP_PORT:-8011}"
GRPC_PORT="${GRPC_PORT:-8012}"
METRICS_PORT="${METRICS_PORT:-8013}"

# Launch (remove --rm if you want persistent logs)
docker run --rm --gpus=all --name triton_server \
  -p"${HTTP_PORT}":8000 -p"${GRPC_PORT}":8001 -p"${METRICS_PORT}":8002 \
  -v "${ROOT_DIR}/model_repo:/models" \
  nvcr.io/nvidia/tritonserver:25.04-py3 \
  tritonserver --model-repository=/models "$@"
```

Make it executable and run:

```bash
chmod +x scripts/start_triton.sh
./scripts/start_triton.sh
```

---

## 🔮 Batch Inference & Plotting

For lightweight local experiments without Triton:

```bash
poetry run python predict_and_plot.py   --data_dir ../data/raw/
```

### 🔌 Triton Client Example

```bash
poetry run python inference/triton/client_lstm.py --data_dir ../data/raw/
```

---

## 🛠️ Development

| Task                   | Command                                 |
| ---------------------- | --------------------------------------- |
| **Format + Lint**      | `poetry run pre-commit run --all-files` |
| **Build wheel**        | `poetry build`                          |
| **Publish (internal)** | `poetry publish --username __token__`   |

### Continuous Integration

Add in `.github/workflows/ci.yml` (sample provided) to automate lint → test →
build.

---

## 📄 License

This project is licensed under the **MIT License** – see [`LICENSE`](LICENSE)
for details.

---

## 🙋‍♂️ FAQ

| Question           | Answer                                                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Dataset source?    | 10 Minute‑level OHLCV data for MOEX tickers, collected via [ISS API](https://iss.moex.com) (script `data/download_moex.py`). |
| CPU‑only training? | Possible; set `trainer.accelerator=cpu`, but expect ×20 slower.                                                              |
| Windows supported? | Training works under WSL 2; TensorRT/Triton require Linux.                                                                   |

---

> Feel free to open an issue or discussion if you hit a snag. Happy forecasting!
