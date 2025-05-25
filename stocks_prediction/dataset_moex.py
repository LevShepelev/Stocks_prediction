# dataset_moex.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MoexStockDataset(Dataset):
    """
    Return
        X : (seq_len, n_features)
        y : (1,)       -- future close
        [optional time marks tensors for FEDformer]
    """

    # ------------------------------------------------------------------ #
    #  constructor                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        root_dir: Path | str,
        seq_len: int,
        horizon: int = 1,
        *,
        label_len: int | None = None,
        pred_len: int | None = None,
        features: Sequence[str] | None = None,
        target: str = "close",
        stride: int = 1,
        single_ticker: bool = False,
        return_marks: bool = False,
        tickers: Sequence[str] | None = None,
    ) -> None:
        super().__init__()

        # --- core dims -----------------------------------------------------
        self.seq_len = seq_len
        self.horizon = horizon
        self.label_len = label_len or seq_len // 2
        self.pred_len = pred_len or horizon

        # --- flags & columns ----------------------------------------------
        self.stride = stride
        self.single_ticker = single_ticker
        self.return_marks = return_marks
        self.feature_cols = list(
            features or ("open", "high", "low", "close", "volume")
        )
        self.target_col = target

        # --- dataframe -----------------------------------------------------
        self.df = self._load_parquets(Path(root_dir), tickers)
        self._make_scaler()

        # --- tensors -------------------------------------------------------
        self.X, self.y, self.mark_enc, self.mark_dec = self._build_pairs()

    # ------------------------------------------------------------------ #
    #  loading / preprocessing                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_parquets(directory: Path, tickers: Sequence[str] | None) -> pd.DataFrame:
        paths = [directory] if directory.is_file() else sorted(directory.glob("*.parquet"))

        if tickers:
            paths = [p for p in paths if p.stem.split("_")[0].upper() in tickers]
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {directory}")

        dfs = []
        for p in paths:
            df = pd.read_parquet(p)
            df["begin"] = pd.to_datetime(df["begin"], utc=True)
            dfs.append(df)

        return (
            pd.concat(dfs)
            .sort_values(["ticker", "begin"])
            .reset_index(drop=True)
        )

    def _make_scaler(self) -> None:
        if self.single_ticker:
            mean = self.df[self.feature_cols].mean().to_numpy()
            std = self.df[self.feature_cols].std().replace(0, 1).to_numpy()
            float_block = (self.df[self.feature_cols].to_numpy(np.float32) - mean) / std
            self.df[self.feature_cols] = float_block
        else:
            for tck, grp in self.df.groupby("ticker"):
                mean = grp[self.feature_cols].mean().to_numpy()
                std = grp[self.feature_cols].std().replace(0, 1).to_numpy()
                idx = grp.index
                self.df.loc[idx, self.feature_cols] = (grp[self.feature_cols].astype("float32") - mean) / std

    # ------------------------------------------------------------------ #
    #  pair generation                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _time_feats(ts: pd.Series) -> np.ndarray:
        return np.stack(
            [
                ts.dt.month / 12.0,
                ts.dt.day / 31.0,
                ts.dt.dayofweek / 7.0,
                ts.dt.hour / 23.0,
                (ts.dt.minute // 10) / 5.0,
            ],
            axis=-1,
        ).astype(np.float32)

    def _build_pairs(self):
        seqs, tgts, m_enc, m_dec = [], [], [], []

        groups = (
            [("_single", self.df)] if self.single_ticker else self.df.groupby("ticker")
        )

        for _, grp in groups:
            arr = grp[self.feature_cols + [self.target_col]].to_numpy(np.float32)
            times = grp["begin"].reset_index(drop=True)

            last_start = len(arr) - self.seq_len - self.horizon + 1
            for i in range(0, last_start, self.stride):
                window = arr[i : i + self.seq_len]
                seqs.append(window[:, :-1])  # drop target col
                target = arr[i + self.seq_len + self.horizon - 1, -1]
                tgts.append(target)

                if self.return_marks:
                    enc_ts = times.iloc[i : i + self.seq_len]
                    dec_ts = times.iloc[
                        i + self.seq_len - self.label_len : i + self.seq_len + self.pred_len
                    ]
                    m_enc.append(self._time_feats(enc_ts))
                    m_dec.append(self._time_feats(dec_ts))

        X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        y = torch.tensor(np.array(tgts, dtype=np.float32).reshape(-1, 1))
        if self.return_marks:
            mark_enc = torch.tensor(np.stack(m_enc), dtype=torch.float32)
            mark_dec = torch.tensor(np.stack(m_dec), dtype=torch.float32)
            return X, y, mark_enc, mark_dec

        return X, y, None, None

    # ------------------------------------------------------------------ #
    #  PyTorch API                                                       #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.return_marks:
            return self.X[idx], self.mark_enc[idx], self.mark_dec[idx], self.y[idx]
        return self.X[idx], self.y[idx]
