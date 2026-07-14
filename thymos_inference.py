#!/usr/bin/env python3
"""Run Thymos-6D Python inference from a published model directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

LABELS = ("valence", "arousal", "dominance", "connectionNeed", "openness", "confidence")


class AffectHead(nn.Module):
    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dimension, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 6),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding.float())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("model"))
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    encoder = SentenceTransformer(str(args.model / "text_encoder"), device="cpu")
    encoder.eval()
    head = AffectHead(encoder.get_embedding_dimension())
    head.load_state_dict(torch.load(args.model / "head.pt", map_location="cpu", weights_only=True))
    head.eval()
    with torch.inference_mode():
        embedding = encoder.encode(args.text, convert_to_tensor=True, normalize_embeddings=False).unsqueeze(0)
        values = head(embedding).squeeze(0).tolist()
    print(json.dumps(dict(zip(LABELS, values)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
