# Cloned Repositories

## tuned-lens
- URL: https://github.com/AlignmentResearch/tuned-lens
- Location: `code/tuned-lens/`
- Purpose: Official library for training and evaluating tuned lenses.
- Key files: `pyproject.toml`, `tuned_lens/`, docs and notebooks.
- Notes: Install with `pip install tuned-lens` or from source. Requires PyTorch. Useful as the first baseline for decoding hidden states with learned affine translators.

## Tokens2Words
- URL: https://github.com/schwartz-lab-NLP/Tokens2Words
- Location: `code/Tokens2Words/`
- Purpose: Implementation for "From Tokens to Words" inner-lexicon analysis and finetuning-free vocabulary expansion.
- Key files: `src/tokens2words/run_patchscopes.py`, `run_vocab_expansion_eval.py`, `representation_translator.py`, `vocab_modifier.py`, `word_retriever.py`.
- Notes: Most directly relevant code. Requirements pin a notebook-heavy environment; use a separate repo-specific venv if full reproduction is needed.

## future-lens
- URL: https://github.com/KoyenaPal/future-lens
- Location: `code/future-lens/`
- Purpose: Code and data workflow for decoding future tokens from a single hidden state.
- Key files: `linear_methods/linear_hs.py`, `causal_methods/train.py`, `causal_methods/test.py`, `demo/FutureLensDemonstration.ipynb`.
- Notes: Requires GPT-J-scale experiments for full reproduction. Useful methodologically for an implicit-token lens that decodes multi-token continuations.

## BackwardLens
- URL: https://github.com/shacharKZ/BackwardLens
- Location: `code/BackwardLens/`
- Purpose: Demo code for projecting gradients and backward-pass VJPs into vocabulary space.
- Key files: `backward_lens_demo.ipynb`, `hook_collect_hidden_states.py`, `llm_utils.py`, `opt_utils.py`.
- Notes: Useful for interpreting learned implicit-token parameters or gradient directions.

## LogitLens4LLMs
- URL: https://github.com/zhenyu-02/LogitLens4LLMs
- Location: `code/LogitLens4LLMs/`
- Purpose: Logit-lens workflow for modern HuggingFace Llama and Qwen models.
- Key files: `main.py`, `activation_analyzer.py`, `model_factory.py`, `model_helper/`.
- Notes: Supports Llama-2-7B, Llama-3.1-8B, Qwen-2.5-7B according to README. Useful for batch layer-wise visualization.

## TransformerLens
- URL: https://github.com/TransformerLensOrg/TransformerLens
- Location: `code/TransformerLens/`
- Purpose: General mechanistic interpretability library for activation caching, patching, and hooks.
- Key files: `transformer_lens/`, `pyproject.toml`, `demos/`.
- Notes: Current README recommends `TransformerBridge.boot_transformers("gpt2")`. Good substrate for custom logit/implicit-token lens experiments.

## peft
- URL: https://github.com/huggingface/peft
- Location: `code/peft/`
- Purpose: Parameter-efficient fine-tuning methods including prompt tuning, prefix tuning, and LoRA.
- Key files: `src/peft/`, `examples/`, `pyproject.toml`.
- Notes: Use for soft-prompt baselines and learned virtual-token experiments.

## patchscopes
- URL: https://github.com/cywinski/patchscopes
- Location: `code/patchscopes/`
- Purpose: PAIR interpretability repository containing Patchscopes paper/site code and related notebooks.
- Key files: `patchscopes/code/our_patchscopes.py`, `patchscopes/code/next_token_prediction.ipynb`, `attribute_extraction.ipynb`, `patch_cross_model.ipynb`.
- Notes: Large checkout because it includes site/static assets and related PAIR projects. Use the `patchscopes/code/` subtree for experiments.

## logit-prisms
- URL: https://github.com/neuralblog/logit-prisms
- Location: `code/logit-prisms/`
- Purpose: Static paper/site assets for Logit Prisms.
- Key files: `docs/index.pdf`, `docs/index.html`.
- Notes: No reusable Python package found in the clone; treat as reference material for decomposition formulas.

## Environment Notes

The workspace venv contains shared packages for downstream experiments:

- `torch`
- `transformers`
- `accelerate`
- `datasets`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pypdf`
- `arxiv`
- `requests`

Per-repo pinned requirements were not installed globally because several include full notebook stacks, GPU-specific packages, or stale pins.

