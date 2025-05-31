# 📈 Stock Prediction MLOps Project

This repository contains an **end‑to‑end educational pipeline** for time‑series forecasting of Russian stocks using deep learning (LSTM/GRU) with a modern MLOps stack:

* **PyTorch + PyTorch Lightning** for model authoring and training
* **Hydra & OmegaConf** for hierarchical configuration management
* **ONNX + TensorRT** for portable, high‑performance inference
* **NVIDIA Triton Inference Server** to serve the model in production‐like conditions
* **Poetry** for dependency management and packaging
* **pre‑commit** (Black + isort + flake8 + mypy) for code quality
* **pytest** for unit tests

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
> * Python ³.11 with CUDA‑enabled GPU & drivers (for training / TRT)
> * Docker ≥ 24 with NVIDIA Container Runtime
> * (Optional) CUDA / cuDNN locally for JIT previews

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

---

## 🚂 Training

Hydra drives the pipeline; all hyper‑parameters live in **`stocks_prediction/conf/`**.

```bash
# Train with the default config
poetry run python stocks_prediction/train_loop.py

# Override any parameter on the CLI (Hydra syntax)
poetry run python stocks_prediction/train_loop.py \
  model=stockgru \
  datamodule.batch_size=256 \
  trainer.max_epochs=50

# Log files, checkpoints & TensorBoard go to ./outputs/<DATE_TIME>/
```

### 🔄 Resume or Fine‑Tune

```bash
poetry run python stocks_prediction/train_loop.py \
  +checkpoint_path=outputs/2025‑05‑31/15‑42‑10/checkpoints/epoch=04.ckpt
```

---

## 📤 Export to ONNX

Training script already exports the **best checkpoint** to ONNX (`exports/onnx/`).
To re‑export any Lightning checkpoint:

```bash
poetry run python -m stocks_prediction.export_onnx \
  --ckpt_path outputs/…/model.ckpt \
  --seq_len 60 --feature_dim 5
```

---

## ⚙️ Convert to TensorRT & Deploy to Triton

```bash
# 1️⃣ Convert ONNX → TensorRT engine and generate Triton directory
poetry run python convert_and_deploy_triton.py \
  --onnx exports/onnx/stocklstm_GAZP_10m_2002‑01‑01_2025‑05‑19.onnx \
  --max-batch 32 \
  --seq-len 60 \
  --feature-dim 5 \
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

---

## 🔮 Batch Inference & Plotting

For lightweight local experiments without Triton:

```bash
poetry run python stocks_prediction/predict_and_plot.py \
  --data_dir ../data/raw/ \
  --ckpt_path outputs/…/best.ckpt
```

### 🔌 Triton Client Example

```python
import numpy as np
import tritonclient.http as http

client = http.InferenceServerClient(url="localhost:8011")
X = np.load("sample_batch.npy").astype(np.float32)  # (B, T, F)
inputs  = http.InferInput("INPUT__0", X.shape, "FP32")
inputs.set_data_from_numpy(X)
result = client.infer("stocklstm", inputs=[inputs])
print(result.as_numpy("OUTPUT__0"))
```

---

## 🛠️ Development

| Task                   | Command                                 |
| ---------------------- | --------------------------------------- |
| **Format + Lint**      | `poetry run pre-commit run --all-files` |
| **Type‑check**         | `poetry run mypy stocks_prediction/`    |
| **Tests**              | `poetry run pytest -q`                  |
| **Build wheel**        | `poetry build`                          |
| **Publish (internal)** | `poetry publish --username __token__`   |

### Continuous Integration

Add in `.github/workflows/ci.yml` (sample provided) to automate lint → test → build.

---

## 📊 Monitoring & Logging

* **Weights & Biases** integration is toggled via `wandb.enabled=true` in Hydra.
* Training logs: `outputs/<run>/train.log` (rotating, JSON).
* Triton exposes Prometheus metrics at **`:8002/metrics`**; a Grafana dashboard `docker‑compose.monitoring.yml` is included.

---

## 🧹 Cleaning Artifacts

```bash
# Remove Hydra outputs (confirm *first*)
find outputs -maxdepth 1 -mtime +7 -type d -exec rm -r {} +
# Purge old TensorBoard event files
rm -rf lightning_logs/
```

---

## 📄 License

This project is licensed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---

## 🙋‍♂️ FAQ

| Question           | Answer                                                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Dataset source?    | Minute‑level OHLCV data for MOEX tickers, collected via [ISS API](https://iss.moex.com) (script `data/download_moex.py`). |
| CPU‑only training? | Possible; set `trainer.accelerator=cpu`, but expect ×20 slower.                                                           |
| Windows supported? | Training works under WSL 2; TensorRT/Triton require Linux.                                                                |

---

> Feel free to open an issue or discussion if you hit a snag. Happy forecasting! \:rocket:
