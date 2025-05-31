#!/usr/bin/env python3
"""
onnx_server_fire.py
-------------------
Serve a StockLSTM ONNX model via FastAPI.  All settings (model path, host,
port, providers, log-level …) are loaded from a Hydra YAML, and you can
override any of them on the command line thanks to Google Fire.

Example
-------
poetry run python inference/onnx_server_fire.py \
    host=0.0.0.0 port=8020 log_level=debug
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List

import fire
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


###############################################################################
# Pydantic schemas
###############################################################################
class PredictRequest(BaseModel):
    # Accept 2-D (batch × seq_len) or 3-D (batch × seq_len × feat_dim)
    input_sequence: List[List[Any]]


class PredictResponse(BaseModel):
    prediction: List[float]


###############################################################################
# Helper – load ONNX
###############################################################################
def _load_onnx(path: str, providers: list[str]) -> ort.InferenceSession:
    if not Path(path).is_file():
        raise FileNotFoundError(f"ONNX model not found at {path}")
    try:
        sess = ort.InferenceSession(path, providers=providers)
        logging.info("ONNX loaded from %s  | providers=%s", path, sess.get_providers())
        return sess
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to load ONNX: %s", exc)
        raise


###############################################################################
# Fire entry-point
###############################################################################
def serve(
    *,
    config: str = "conf/server/onnx_stocklstm.yaml",
    model_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    providers: list[str] | None = None,
    log_level: str | None = None,
) -> None:
    cfg: DictConfig = OmegaConf.load(config)
    for k, v in {
        "model_path": model_path,
        "host": host,
        "port": port,
        "providers": providers,
        "log_level": log_level,
    }.items():
        if v is not None:
            cfg[k] = v

    logging.basicConfig(
        level=getattr(logging, str(cfg.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Config: %s", OmegaConf.to_yaml(cfg, resolve=True))

    # --------- FASTAPI LIFESPAN HANDLER ---------
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.session = _load_onnx(cfg.model_path, cfg.providers)
        yield
        # (Optional: do cleanup here)

    app = FastAPI(
        title="StockLSTM ONNX Server",
        description="Serve predictions from a StockLSTM ONNX model",
        version="1.0.0",
        lifespan=lifespan,  # <-- use lifespan
    )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        try:
            arr = np.asarray(req.input_sequence, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            elif arr.ndim != 3:
                raise ValueError("Input must be a 2-D or 3-D float array")

            inp_name = app.state.session.get_inputs()[0].name
            preds = app.state.session.run(None, {inp_name: arr})[0].squeeze()
            return PredictResponse(
                prediction=(
                    preds.tolist() if isinstance(preds, np.ndarray) else [float(preds)]
                )
            )
        except ValueError as ve:
            raise HTTPException(status_code=422, detail=str(ve)) from ve
        except Exception as exc:
            logger.exception("Inference error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    uvicorn.run(app, host=cfg.host, port=int(cfg.port), log_level=cfg.log_level)


def main() -> None:  # pragma: no cover
    fire.Fire(serve)


if __name__ == "__main__":  # pragma: no cover
    main()
