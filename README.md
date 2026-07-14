---
language:
- zh
- en
license: apache-2.0
library_name: pytorch
base_model: Qwen/Qwen3-Embedding-0.6B
tags:
- emotion
- affective-computing
- text-classification
- six-dimensional-affect
---

# Thymos-6D

Thymos is a six-dimensional text affect model. It maps a message to a bounded
continuous state rather than a single emotion category:

`[valence, arousal, dominance, connectionNeed, openness, confidence]`

The model fine-tunes the full `Qwen/Qwen3-Embedding-0.6B` encoder with a
`1024 -> 256 -> 6` sigmoid head. The exported `model.pt` is a TorchScript
bundle for DJL integration; `text_encoder/` and `head.pt` are included for
Python inference and reproducibility.

## Results

The held-out test split contains 819 records. Mean absolute error is
`0.078755` across the six dimensions.

Per-dimension MAE:

| Dimension | MAE |
| --- | ---: |
| valence | 0.05160 |
| arousal | 0.06291 |
| dominance | 0.05799 |
| connectionNeed | 0.06470 |
| openness | 0.07713 |
| confidence | 0.15821 |

The training corpus was generated and reviewed synthetically. These metrics
are not a substitute for evaluation on consented, representative human data.
The model is not a clinical or mental-health diagnostic tool. Sarcasm,
negation, code-switching, cultural context and ambiguous short messages may
be unreliable. Treat `confidence` as an uncertainty signal, not a probability
that the prediction is correct.

## Python inference

```powershell
python -m pip install -r requirements.txt
python thymos_inference.py --model model --text "别担心，我只是有点累"
```

## Re-training

The raw training corpus is intentionally not included. Provide a local JSONL
corpus with the fields `sampleId`, `text`, and the six output names, then run:

```powershell
python scripts/train-user-affect-model.py --corpus path/to/corpus.jsonl --output model
```

Training requires CUDA and BF16. CPU fallback is disabled by design.

## License and attribution

The repository code is licensed under AGPL-3.0-or-later. The model is a
fine-tuned derivative of Qwen3-Embedding-0.6B and remains subject to the
upstream model license and terms. See the Qwen model card before redistribution
or commercial use.
