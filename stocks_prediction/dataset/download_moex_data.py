#!/usr/bin/env python
"""
Download 10-minute candles from MOEX ISS via apimoex.

Usage:
    python download_moex_data.py \
        --tickers assets/moex_tickers.txt \
        --start 2023-01-01 \
        --end   2024-05-01 \
        --out-dir data/raw
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import time
from pathlib import Path
from typing import List

import apimoex
import pandas as pd
import requests
from tqdm import tqdm

INTERVAL = 10  # 10-minute candles, MOEX code “10” :contentReference[oaicite:0]{index=0}


def parse_ticker_file(file_path: Path) -> List[str]:
    """
    Extract ticker symbols from *any* human-written list.
    We take the first token before a comma, semicolon or whitespace.
    """
    tickers: list[str] = []
    pattern = re.compile(r"^[\s–—\-]*([A-ZА-Я]+)[,;\s]")

    with file_path.open(encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                tickers.append(m.group(1).upper())

    if not tickers:
        raise ValueError("No tickers recognised in file!")

    return tickers


def fetch_one(
    session: requests.Session,
    ticker: str,
    start: str,
    end: str,
    interval: int = INTERVAL,
) -> pd.DataFrame:
    """
    Download candles for a single ticker and return as DataFrame.
    """
    logging.info("Downloading %s %s", ticker, interval)
    candles = apimoex.get_market_candles(
        session=session, security=ticker, interval=interval, start=start, end=end
    )
    if not candles:
        raise RuntimeError(f"MOEX returned 0 rows for {ticker}…")

    df = pd.DataFrame(candles)
    df["ticker"] = ticker
    df["begin"] = pd.to_datetime(df["begin"])
    return df.sort_values("begin")


def main() -> None:
    argp = argparse.ArgumentParser()
    argp.add_argument("--tickers", required=True, type=Path)
    argp.add_argument("--start", required=True, help="YYYY-MM-DD")
    argp.add_argument("--end", required=True, help="YYYY-MM-DD")
    argp.add_argument("--out-dir", default="../data/raw", type=Path)
    argp.add_argument("--sleep", default=0.2, type=float, help="seconds between calls")
    args = argp.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        filename="app.log",         # path to your log file
        filemode="a",               # "a" for append, "w" to overwrite each run
        encoding="utf-8"            # ensures proper character encoding
    )
    (out_dir := args.out_dir).mkdir(parents=True, exist_ok=True)
    tickers = parse_ticker_file(args.tickers)
    logging.info("Downloading %d tickers: %s", len(tickers), ", ".join(tickers))

    with requests.Session() as session:
        for ticker in tqdm(tickers, desc="MOEX"):
            try:
                df = fetch_one(session, ticker, args.start, args.end)
            except Exception as ex:
                logging.warning("Skip %s → %s", ticker, ex)
                continue

            fname = f"{ticker}_{INTERVAL:02d}m_{args.start}_{args.end}.parquet"
            df.to_parquet(out_dir / fname, index=False)
            time.sleep(args.sleep)

    logging.info("Done. Files written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
