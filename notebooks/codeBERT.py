#!/usr/bin/env python3
"""
CodeBERT Feature Extraction (Refactored)
=======================================

This script extracts fixed‑length embeddings from a column of text in a CSV
using a pre‑trained CodeBERT (RoBERTa) model. It is designed to be simple
to use from the command line and flexible enough for a variety of
workflows. The script accepts several parameters to control how the
embeddings are computed, including the model name, device selection,
batch size, maximum sequence length, and pooling strategy.

Key Features
------------
* Reads text values from a specified column in a CSV file.
* Uses HuggingFace's Transformers library to load a CodeBERT model and
  tokenizer.
* Supports mean pooling or CLS token pooling to obtain a single
  vector per input.
* Allows specifying the compute device (CPU/GPU) and enabling fp16 on CUDA.
* Saves the resulting embeddings to a NumPy `.npy` file or a PyTorch
  `.pt` tensor file.
* Writes a side‑car JSON file containing metadata about the run (e.g.,
  model name, number of rows, embedding dimension).

Example Usage
-------------
```
# Extract embeddings from the 'values_concat' column of col_yago_train.csv
# and save as train_embeddings.npy along with train_embeddings.meta.json
python codebert.py \
  --input col_yago_train.csv \
  --text-col values_concat \
  --out train_embeddings.npy

# Extract embeddings using CLS pooling and a shorter max sequence length
python codebert.py \
  --input col_yago_test.csv \
  --text-col values_concat \
  --out test_embeddings.pt \
  --pooling cls \
  --max-len 128 \
  --batch-size 32
```

"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np  # type: ignore
import torch  # type: ignore
from transformers import AutoTokenizer, AutoModel  # type: ignore
from tqdm import tqdm  # type: ignore


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean‑pool token embeddings across the sequence length.

    Parameters
    ----------
    last_hidden_state : torch.Tensor
        The hidden states returned from the transformer model of shape
        ``[batch_size, sequence_length, hidden_size]``.
    attention_mask : torch.Tensor
        A tensor of shape ``[batch_size, sequence_length]`` indicating which
        tokens are real (1) versus padding (0).

    Returns
    -------
    torch.Tensor
        A tensor of shape ``[batch_size, hidden_size]`` containing the
        mean‑pooled embeddings.
    """
    # Expand attention_mask to match the last_hidden_state shape
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    # Sum the hidden states where mask == 1
    summed = (last_hidden_state * mask).sum(dim=1)
    # Count the number of valid tokens to avoid dividing by zero
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Return the embeddings from the CLS token (first token) of each sequence."""
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
    """Compute embeddings for a collection of strings using CodeBERT.

    This function lazily batches the provided iterable of texts and passes
    them through the transformer model. It then applies the chosen pooling
    strategy to obtain a single embedding per input string.

    Parameters
    ----------
    texts : Iterable[str]
        An iterable of texts to embed.
    model_name : str, optional
        The HuggingFace model identifier to load. Defaults to
        ``microsoft/codebert-base``.
    device : str, optional
        Which device to run on: ``"auto"`` selects CUDA if available,
        otherwise CPU. You can also explicitly set ``"cpu"`` or ``"cuda"``.
    batch_size : int, optional
        The number of texts per inference batch. Defaults to 64.
    max_len : int, optional
        Maximum length in tokens for truncation/padding. Defaults to 256.
    pooling : str, optional
        Pooling strategy: ``"mean"`` or ``"cls"``. Defaults to ``"mean"``.
    fp16 : bool, optional
        If True and a CUDA device is selected, perform inference in
        half precision (fp16). Defaults to False.

    Returns
    -------
    np.ndarray
        A NumPy array of shape ``[num_texts, hidden_size]`` containing the
        embeddings.
    """
    # Normalize the input texts (replace None with empty strings)
    text_list = ["" if t is None else str(t) for t in texts]

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Optionally cast model to half precision
    if fp16 and device == "cuda":
        model.half()

    # Select pooling function
    if pooling == "mean":
        pool_fn = lambda out, mask: mean_pool(out.last_hidden_state, mask)
    elif pooling == "cls":
        pool_fn = lambda out, mask: cls_pool(out.last_hidden_state)
    else:
        raise ValueError(f"Unknown pooling '{pooling}'. Use 'mean' or 'cls'.")

    outputs = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(text_list), batch_size), desc=f"CodeBERT({pooling})"):
            batch = text_list[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            result = model(**encoded)
            pooled = pool_fn(result, encoded["attention_mask"])
            outputs.append(pooled.detach().to("cpu"))

    return torch.cat(outputs, dim=0).numpy()


def load_texts_from_csv(path: str, text_col: str) -> List[str]:
    """Load a column of text from a CSV file as a list of strings.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    text_col : str
        Name of the column containing the text to embed.

    Returns
    -------
    List[str]
        List of text values.
    """
    import pandas as pd  # local import to keep global import footprint small
    df = pd.read_csv(path)
    if text_col not in df.columns:
        cols = ", ".join(df.columns)
        raise ValueError(f"Column '{text_col}' not in CSV. Available columns: {cols}")
    return df[text_col].astype(str).tolist()


def save_embeddings(out_path: str, arr: np.ndarray) -> None:
    """Persist an embedding array to disk in either NumPy or PyTorch format."""
    out_path = str(out_path)
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    # Save as .pt if extension is .pt (case‑insensitive)
    if out_path.lower().endswith(".pt"):
        torch.save(torch.from_numpy(arr), out_path)
    else:
        # Default to .npy, adding extension if necessary
        if not out_path.lower().endswith(".npy"):
            out_path += ".npy"
        np.save(out_path, arr)


def save_metadata(out_path: str, meta: dict) -> None:
    """Write a JSON metadata file adjacent to the embeddings file."""
    if out_path.lower().endswith(".pt"):
        meta_path = out_path[:-3] + ".meta.json"
    elif out_path.lower().endswith(".npy"):
        meta_path = out_path[:-4] + ".meta.json"
    else:
        meta_path = out_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="Extract CodeBERT embeddings from a CSV column")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--text-col", required=True, help="Name of the column containing text values")
    parser.add_argument("--out", required=True, help="Output file for embeddings (.npy or .pt)")
    parser.add_argument("--model", default="microsoft/codebert-base", help="HuggingFace model identifier")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device to run the model on")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum sequence length for tokenization")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Pooling strategy: mean or cls")
    parser.add_argument("--fp16", action="store_true", help="Use half precision when running on CUDA")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Load texts
    texts = load_texts_from_csv(args.input, args.text_col)
    # Compute embeddings
    embeddings = codebert_embed(
        texts=texts,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        pooling=args.pooling,
        fp16=args.fp16,
    )
    # Save embeddings and metadata
    save_embeddings(args.out, embeddings)
    save_metadata(
        args.out,
        {
            "input": args.input,
            "text_col": args.text_col,
            "output": args.out,
            "num_rows": len(texts),
            "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
            "model": args.model,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "pooling": args.pooling,
            "fp16": bool(args.fp16),
        },
    )
    print(f"[OK] Wrote embeddings to {args.out} (shape={embeddings.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
