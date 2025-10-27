#!/usr/bin/env python3
"""
CodeBERT Feature Extraction (Refactored)
=======================================

Extract fixed-size embeddings from text values using CodeBERT (RoBERTa).
- Loads texts from a CSV (choose the column).
- Batches through the model with mean/CLS pooling.
- Saves embeddings to .npy (or .pt) with a metadata sidecar.

Examples
--------
# Basic: read 'values_concat' from CSV and write embeddings.npy
python codebert_features.py \
  --input col_yago_train.csv \
  --text-col values_concat \
  --out embeddings_train.npy

# Change batch size / max length / pooling, save as .pt
python codebert_features.py \
  --input col_yago_test.csv \
  --text-col values_concat \
  --out embeddings_test.pt \
  --pooling cls \
  --batch-size 32 \
  --max-len 128

# Select model and device explicitly
python codebert_features.py \
  --input col_yago_train.csv \
  --text-col values_concat \
  --out train.npy \
  --model microsoft/codebert-base \
  --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# --------------------------
# Pooling & embedding helpers
# --------------------------
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings using the attention mask.
    last_hidden_state: [B, T, H]
    attention_mask:    [B, T]
    returns:           [B, H]
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Return the <s> (CLS) embedding at position 0 for each sequence.
    last_hidden_state: [B, T, H]
    returns:           [B, H]
    """
    return last_hidden_state[:, 0, :]


def codebert_embed(
    texts: Iterable[str],
    model_name: str = "microsoft/codebert-base",
    device: str = "auto",
    batch_size: int = 64,
    max_len: int = 256,
    pooling: str = "mean",
    fp16: bool = False,
) -> np.ndarray:
    """
    Compute embeddings for a list/iterable of strings using CodeBERT.

    Parameters
    ----------
    texts : Iterable[str]
    model_name : str
        HF model id (default: microsoft/codebert-base)
    device : {"auto","cpu","cuda"}
    batch_size : int
    max_len : int
    pooling : {"mean","cls"}
    fp16 : bool
        If True and device is CUDA, run the model in float16.

    Returns
    -------
    np.ndarray of shape [N, H]
    """
    texts = ["" if t is None else str(t) for t in texts]

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    if fp16 and device == "cuda":
        model.half()

    # Choose pooling function
    if pooling == "mean":
        pool_fn = lambda out, mask: mean_pool(out.last_hidden_state, mask)
    elif pooling == "cls":
        pool_fn = lambda out, mask: cls_pool(out.last_hidden_state)
    else:
        raise ValueError(f"Unknown pooling '{pooling}'. Use 'mean' or 'cls'.")

    outputs: List[torch.Tensor] = []
    # inference_mode avoids grad overhead and is safer than no_grad for inference
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"CodeBERT({pooling})"):
            batch = texts[i : i + batch_size]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            out = model(**toks)
            pooled = pool_fn(out, toks["attention_mask"])  # [B, H]
            outputs.append(pooled.detach().to("cpu"))

    return torch.cat(outputs, dim=0).numpy()


# -------------
# I/O utilities
# -------------
def load_texts_from_csv(path: str, text_col: str) -> List[str]:
    import pandas as pd  # defer import to keep base deps light
    df = pd.read_csv(path)
    if text_col not in df.columns:
        cols = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{text_col}' not in CSV. Available: {cols}")
    return df[text_col].astype(str).tolist()


def save_embeddings(out_path: str, arr: np.ndarray) -> None:
    out_path = str(out_path)
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    if out_path.lower().endswith(".pt"):
        torch.save(torch.from_numpy(arr), out_path)
    else:
        # default .npy
        if not out_path.lower().endswith(".npy"):
            out_path += ".npy"
        np.save(out_path, arr)


def save_metadata(out_path: str, meta: dict) -> None:
    meta_path = out_path
    if meta_path.lower().endswith(".pt"):
        meta_path = meta_path[:-3] + ".meta.json"
    elif meta_path.lower().endswith(".npy"):
        meta_path = meta_path[:-4] + ".meta.json"
    else:
        meta_path = meta_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# -----
#  CLI
# -----
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract CodeBERT embeddings from a CSV text column.")
    p.add_argument("--input", required=True, help="Input CSV path.")
    p.add_argument("--text-col", required=True, help="Text column to embed (e.g., values_concat).")
    p.add_argument("--out", required=True, help="Output embeddings file (.npy or .pt).")

    p.add_argument("--model", default="microsoft/codebert-base", help="HF model id.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p.add_argument("--max-len", type=int, default=256, help="Tokenization max length.")
    p.add_argument("--pooling", default="mean", choices=["mean", "cls"], help="Pooling strategy.")
    p.add_argument("--fp16", action="store_true", help="Use fp16 on CUDA.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    texts = load_texts_from_csv(args.input, args.text_col)
    emb = codebert_embed(
        texts=texts,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        pooling=args.pooling,
        fp16=args.fp16,
    )

    save_embeddings(args.out, emb)
    save_metadata(
        args.out,
        {
            "input": args.input,
            "text_col": args.text_col,
            "output": args.out,
            "num_rows": len(texts),
            "dim": int(emb.shape[1]) if emb.ndim == 2 else None,
            "model": args.model,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "pooling": args.pooling,
            "fp16": bool(args.fp16),
        },
    )
    print(f"[OK] Wrote embeddings to {args.out} (shape={emb.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
