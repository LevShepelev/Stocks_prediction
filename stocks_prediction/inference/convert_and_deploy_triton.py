#!/usr/bin/env python
"""
Convert an ONNX StockLSTM model → TensorRT engine (.plan) and scaffold a Triton
model-repo entry, driven by Google Fire and Hydra/OmegaConf.

Example
-------
python convert_and_deploy_triton_fire.py \
    --onnx artifacts/stocklstm.onnx \
    --config conf/inference/stocklstm.yaml \
    --trt_path artifacts/stocklstm.plan \
    --model_name stocklstm \
    --repo_dir model_repository
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import fire
import tensorrt as trt
from importlib.metadata import version
from omegaconf import DictConfig, OmegaConf

# ----------------------------------------------------------------------------- #
# Logging
# ----------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)
LOGGER.info("TensorRT %s", version("tensorrt"))

TRT_LOGGER: trt.Logger = trt.Logger(trt.Logger.INFO)


# ----------------------------------------------------------------------------- #
# Core logic (unchanged)
# ----------------------------------------------------------------------------- #
def build_engine(
    onnx_path: str,
    trt_path: str,
    max_batch: int,
    seq_len: int,
    feature_dim: int,
) -> None:
    """Compile ONNX → TensorRT (.plan) and write it to *trt_path*."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                LOGGER.error(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    in_name = network.get_input(0).name
    profile.set_shape(
        in_name,
        (1, seq_len, feature_dim),
        (max_batch, seq_len, feature_dim),
        (max_batch, seq_len, feature_dim),
    )
    config.add_optimization_profile(profile)

    LOGGER.info("Building TensorRT engine …")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT build failed")

    with open(trt_path, "wb") as f:
        f.write(engine_bytes)
    LOGGER.info("TensorRT engine written → %s", trt_path)


def create_triton_model_repo(
    model_name: str,
    trt_path: str,
    seq_len: int,
    feature_dim: int,
    max_batch: int,
    repo_dir: str,
) -> None:
    """Place *.plan* and *config.pbtxt* into a Triton model-repository folder."""
    model_dir = Path(repo_dir) / model_name / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

    dest_plan = model_dir / "model.plan"
    if dest_plan.exists():
        dest_plan.unlink()
    os.replace(trt_path, dest_plan)

    config_pbtxt = f"""
name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch}
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ {seq_len}, {feature_dim} ]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [ {{ kind: KIND_GPU }} ]
""".lstrip()
    (model_dir.parent / "config.pbtxt").write_text(config_pbtxt, encoding="utf-8")
    LOGGER.info("Triton repo updated → %s", model_dir.parent)


# ----------------------------------------------------------------------------- #
# CLI entry-point (Fire) + Hydra-backed config
# ----------------------------------------------------------------------------- #
def convert_and_deploy(
    *,
    onnx: str,
    config: str = "conf/inference/stocklstm.yaml",
    trt_path: str = "stocklstm.plan",
    model_name: str = "stocklstm",
    repo_dir: str = "model_repository",
) -> None:
    """
    Convert *onnx* → TensorRT and prepare a Triton repo entry.

    All static model parameters (seq_len, feature_dim, max_batch, …) are read
    from **config** (Hydra/OmegaConf YAML) so they remain in sync with training.
    """
    cfg: DictConfig = OmegaConf.load(config)
    seq_len = int(cfg.seq_len)
    feature_dim = int(cfg.feature_dim)
    max_batch = int(cfg.max_batch)

    LOGGER.info(
        "Loaded cfg | seq_len=%d feature_dim=%d max_batch=%d",
        seq_len,
        feature_dim,
        max_batch,
    )

    build_engine(
        onnx_path=onnx,
        trt_path=trt_path,
        max_batch=max_batch,
        seq_len=seq_len,
        feature_dim=feature_dim,
    )
    create_triton_model_repo(
        model_name=model_name,
        trt_path=trt_path,
        seq_len=seq_len,
        feature_dim=feature_dim,
        max_batch=max_batch,
        repo_dir=repo_dir,
    )


def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    """Dispatch CLI → convert_and_deploy via Fire."""
    fire.Fire(convert_and_deploy)


if __name__ == "__main__":  # pragma: no cover
    main()
