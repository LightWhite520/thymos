---
language:
- zh
- en
license: agpl-3.0
library_name: pytorch
base_model: Qwen/Qwen3-Embedding-0.6B
tags:
- emotion
- affective-computing
- text-classification
- six-dimensional-affect
---

# Thymos-6D

Published model bundle: [0x4C57/Thymos-6D](https://huggingface.co/0x4C57/Thymos-6D)

Thymos is a six-dimensional text affect model. It maps a message to a bounded
continuous state rather than a single emotion category:

`[valence, arousal, dominance, connectionNeed, openness, confidence]`

The model fine-tunes the full `Qwen/Qwen3-Embedding-0.6B` encoder with a
`1024 -> 256 -> 6` sigmoid head. The exported `model.pt` is a TorchScript
bundle for DJL integration; `text_encoder/` and `head.pt` are included for
Python inference and reproducibility.

## Results

The held-out test split contains 1,024 records. Mean absolute error is
`0.069860` across the six dimensions.

Per-dimension MAE:

| Dimension | MAE |
| --- | ---: |
| valence | 0.04775 |
| arousal | 0.05858 |
| dominance | 0.05564 |
| connectionNeed | 0.05587 |
| openness | 0.07358 |
| confidence | 0.12774 |

The training corpus was generated and reviewed synthetically. These metrics
are not a substitute for evaluation on consented, representative human data.
The model is not a clinical or mental-health diagnostic tool. Sarcasm,
negation, code-switching, cultural context and ambiguous short messages may
be unreliable. Treat `confidence` as an uncertainty signal, not a probability
that the prediction is correct.

## JVM inference

Thymos exposes the same DJL inference chain used by OpenEden: Hugging Face
tokenization, fixed-length Qwen inputs, TorchScript inference, six-dimensional
validation, and a caller-supplied fallback.

```kotlin
import io.thymos.ThymosAffectAnalyzer
import java.nio.file.Path

ThymosAffectAnalyzer.fromBundle(Path.of("model")).use { analyzer ->
    val state = analyzer.analyze("别担心，我只是有点累")
    println(state)
}
```

Download the runtime bundle and invoke the standalone CLI:

```powershell
.\gradlew.bat ensureModel
.\gradlew.bat run --args="--model-dir model --text 别担心，我只是有点累"
```

The Maven publication coordinates are `io.openeden:thymos:0.1.0`. Consumers
must provide the DJL PyTorch native runtime for their platform; the Windows
distribution includes the tested `win-x86_64` CPU runtime.

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

The repository code and Thymos model artifacts are licensed under
AGPL-3.0-or-later. The model is a fine-tuned derivative of
Qwen3-Embedding-0.6B; upstream copyright, attribution, license notices and
terms continue to apply. See the Qwen model card before redistribution or
commercial use.
