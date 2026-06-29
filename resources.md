# Resources Catalog

## Summary

This document catalogs resources gathered for the project "Logit Lens with Implicit Tokens." The paper-finder service was attempted first but timed out on all bounded queries, so resources were gathered through arXiv, ACL Anthology, OpenReview/HuggingFace/GitHub, and targeted manual search.

## Papers

Total papers downloaded: 14

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Eliciting Latent Predictions from Transformers with the Tuned Lens | Belrose et al. | 2023 | `papers/2303.08112_tuned_lens.pdf` | Affine per-layer lens baseline |
| From Tokens to Words: On the Inner Lexicon of LLMs | Kaplan et al. | 2025 | `papers/2410.05864_tokens_to_words_inner_lexicon.pdf` | Direct latent vocabulary evidence |
| Patchscopes | Ghandeharioun et al. | 2024 | `papers/2401.06102_patchscopes.pdf` | Patch-based hidden-state decoding |
| Future Lens | Pal et al. | 2023 | `papers/2023.conll-1.37_future_lens.pdf` | Multi-token future decoding |
| Backward Lens | Katz et al. | 2024 | `papers/2024.emnlp-main.142_backward_lens.pdf` | Gradient/VJP vocabulary projection |
| LogitLens4LLMs | Wang | 2025 | `papers/2503.11667_logitlens4llms.pdf` | Modern LLM logit-lens toolkit |
| The Power of Scale for Prompt Tuning | Lester et al. | 2021 | `papers/2104.08691_prompt_tuning_scale.pdf` | Soft prompt baseline |
| Prefix-Tuning | Li and Liang | 2021 | `papers/2101.00190_prefix_tuning.pdf` | Virtual-prefix token baseline |
| P-Tuning v2 | Liu et al. | 2022 | `papers/2110.07602_p_tuning_v2.pdf` | Deep prompt tuning baseline |
| AutoPrompt | Shin et al. | 2020 | `papers/2010.15980_autoprompt.pdf` | Discrete prompt-search baseline |
| FFN Layers Are Key-Value Memories | Geva et al. | 2021 | `papers/2012.14913_ffn_key_value_memories.pdf` | FFN memory and vocabulary projection |
| Universality and Limitations of Prompt Tuning | Wang et al. | 2023 | `papers/2305.18787_universality_limitations_prompt_tuning.pdf` | Prompt tuning theory |
| When Do Prompting and Prefix-Tuning Work? | Petrov et al. | 2024 | `papers/2310.19698_prompting_prefix_tuning_theory.pdf` | Prefix/prompt expressivity limits |
| Logit Prisms | Nguyen | 2024 | `papers/logit_prisms_nguyen_2024.pdf` | Component-level logit decomposition |

See `papers/README.md` for details.

## Datasets

Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiText-2 Raw | HuggingFace `Salesforce/wikitext` | 36,718 train / 3,760 val / 4,358 test | LM and word-splitting analysis | `datasets/wikitext2_raw/` | Filter empty rows |
| LAMBADA OpenAI | HuggingFace `EleutherAI/lambada_openai` | 5,153 test | Long-context next-word prediction | `datasets/lambada_openai/` | Test split only |
| GLUE SST-2 | HuggingFace `nyu-mll/glue`, `sst2` | 67,349 train / 872 val / 1,821 test | Soft prompt/task baseline | `datasets/glue_sst2/` | Test labels are hidden/placeholder in GLUE style |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

Total repositories cloned: 9

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| tuned-lens | https://github.com/AlignmentResearch/tuned-lens | Tuned lens training/evaluation | `code/tuned-lens/` | Core explicit-vocab lens baseline |
| Tokens2Words | https://github.com/schwartz-lab-NLP/Tokens2Words | Inner lexicon and vocab expansion | `code/Tokens2Words/` | Most direct implementation |
| future-lens | https://github.com/KoyenaPal/future-lens | Future-token hidden-state decoding | `code/future-lens/` | GPT-J oriented |
| BackwardLens | https://github.com/shacharKZ/BackwardLens | Gradient vocabulary projection | `code/BackwardLens/` | Notebook/demo repo |
| LogitLens4LLMs | https://github.com/zhenyu-02/LogitLens4LLMs | Logit lens for modern HF LLMs | `code/LogitLens4LLMs/` | Llama/Qwen helpers |
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | Activation hooks and patching | `code/TransformerLens/` | General experiment substrate |
| peft | https://github.com/huggingface/peft | Soft/prefix prompt baselines | `code/peft/` | Use library APIs |
| patchscopes | https://github.com/cywinski/patchscopes | Patchscopes code/site | `code/patchscopes/` | Use `patchscopes/code/` subtree |
| logit-prisms | https://github.com/neuralblog/logit-prisms | Logit prism reference assets | `code/logit-prisms/` | No package code found |

See `code/README.md` for details.

## Resource Gathering Notes

### Search Strategy
- Started with paper-finder, per instructions. Diligent and fast queries timed out.
- Used manual search over arXiv, ACL Anthology, OpenReview, HuggingFace dataset cards, GitHub, and paper project pages.
- Prioritized direct relevance to implicit vocabulary, hidden-state decoding, and continuous virtual tokens.

### Selection Criteria
- Directly supports logit-lens or vocabulary-space interpretation.
- Provides evidence for hidden token-like representations beyond tokenizer entries.
- Provides executable code, datasets, or baseline methods for experiments.
- Small enough to download and validate within the workspace.

### Challenges Encountered
- Paper-finder service did not return within 30 seconds on bounded fast queries and was killed/allowed to timeout.
- HuggingFace `datasets` v5 requires namespaced dataset IDs for older datasets; fixed by using `Salesforce/wikitext` and `nyu-mll/glue`.
- Some cloned repos have heavy notebook/GPU pinned environments, so only shared minimum dependencies were installed in the main venv.
- Patchscopes clone is large because it includes site/static assets and multiple PAIR projects.

### Gaps and Workarounds
- The Pile was not downloaded due size; use streaming or sampled subsets later.
- PubMed abstracts and Arabic Wiki40B were not downloaded; use only if domain/multilingual vocabulary expansion becomes central.
- No full model checkpoints were downloaded. Experiments should begin with small HF models and cache models separately.

## Recommendations for Experiment Design

1. Primary datasets: WikiText-2 for word-level hidden-state experiments; LAMBADA for long-context implicit continuation tests; SST-2 for soft prompt baselines.
2. Baselines: direct logit lens, tuned lens, Patchscope token identity, mean embedding for multi-token words, AutoPrompt, and PEFT prompt tuning.
3. Metrics: retrieval accuracy, precision@1, surprisal, KL to final logits, layer of first retrieval, and ablation effect on downstream prediction.
4. Code to reuse: start with TransformerLens for hooks, tuned-lens for affine translators, Tokens2Words for inner-lexicon/vocab expansion logic, and PEFT for learned soft tokens.

## Research Execution Outputs

The automated research run produced a working implicit-token lens prototype using `Qwen/Qwen2.5-0.5B` and WikiText-2. The primary deliverables are:

- `REPORT.md`: final research report with methodology, figures, statistical tests, limitations, and conclusions.
- `README.md`: concise project overview and reproduction command.
- `planning.md`: motivation, novelty assessment, and experimental plan.
- `src/run_implicit_token_lens.py`: full reproducible experiment pipeline.
- `results/summary.json`: primary comparison, layer-wise tests, and candidate-vocabulary statistics.
- `results/additional_comparisons.json`: paired comparisons against selected raw-prototype and mean-subtoken baselines.
- `results/metrics_layerwise.csv`: per-layer metrics for all methods.
- `results/candidate_vocab.json`: the 128 multi-token implicit word entries used in the experiment.
- `figures/layer_top1_accuracy.png`, `figures/layer_top5_accuracy.png`, `figures/layer_mrr.png`: generated visualizations.

Headline result: the layer-local implicit-token lens reached 19.5% test top-1 accuracy over 128 whole-word labels, compared with 12.5% for the final-subtoken majority baseline and 0.78% chance.
