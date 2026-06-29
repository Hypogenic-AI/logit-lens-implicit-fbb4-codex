# Logit Lens with Implicit Tokens

This workspace tests whether a logit-lens-style decoder can output implicit vocabulary items, not just explicit tokenizer IDs. The experiment builds a 128-entry implicit vocabulary from multi-token WikiText-2 words and decodes held-out Qwen2.5-0.5B hidden states at the final subtoken position.

## Key Findings

- A layer-local implicit-token lens reached 19.5% test top-1 accuracy over 128 whole-word labels, versus 12.5% for the final-subtoken majority baseline and 0.78% chance.
- The ITL beat the best raw-final-prototype baseline by 5.5 percentage points on test, bootstrap 95% CI [2.1, 8.8], paired randomization p = 0.0023.
- The Tuned-Lens-style ridge-to-final-prototype ablation failed at this data scale, reaching only 3.7% test top-1 at its validation-selected layer.
- The result supports a bounded claim: supervised implicit word prototype tables can work as a lens, but this is not yet unsupervised latent vocabulary discovery.

See [REPORT.md](REPORT.md) for the full methodology, results, figures, and limitations.

## Reproduce

The project uses the isolated `.venv` already present in this workspace.

```bash
source .venv/bin/activate
HF_HOME=./artifacts/hf_cache CUDA_VISIBLE_DEVICES=1 python src/run_implicit_token_lens.py --batch-size 64
```

The run uses cached HuggingFace assets under `artifacts/hf_cache` and saves outputs under `results/` and `figures/`. A cached rerun reproduced identical hashes for the reported summary and metric files.

## File Structure

- `src/run_implicit_token_lens.py`: full experiment pipeline.
- `planning.md`: motivation, novelty assessment, and experimental design.
- `REPORT.md`: final research report.
- `results/metrics_layerwise.csv`: layer-wise metrics for all methods.
- `results/summary.json`: primary statistical comparison.
- `results/additional_comparisons.json`: paired comparisons against selected baselines.
- `results/candidate_vocab.json`: implicit word entries and tokenizer pieces.
- `figures/layer_top1_accuracy.png`: main layer-wise accuracy plot.
- `literature_review.md`, `resources.md`: pre-gathered research context and resource catalog.
