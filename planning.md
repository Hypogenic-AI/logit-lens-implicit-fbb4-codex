# Logit Lens with Implicit Tokens: Research Plan

## Motivation & Novelty Assessment

### Why This Research Matters
Standard logit lenses can only decode intermediate states into the model's explicit tokenizer vocabulary, but recent evidence suggests that LLMs form word-, phrase-, and entity-like representations that are not single vocabulary entries. A lens that exposes these implicit entries would make detokenization and early semantic composition more measurable, and could support vocabulary expansion, debugging, and mechanistic interpretability workflows that currently lose information at tokenizer boundaries.

### Gap in Existing Work
The gathered literature shows strong evidence for implicit lexical items: token erasure in multi-token words and entities, whole-word reconstruction in the inner-lexicon paper, future-token information in hidden states, and continuous prompt vectors that behave like virtual tokens. However, existing logit/tuned-lens methods still output explicit token IDs, while Patchscopes can verbalize hidden states but do not provide a reusable vocabulary-like output layer. The missing step is a quantitative lens whose output categories are implicit multi-token lexical items.

### Our Novel Contribution
This project tests an "implicit token lens" (ITL): a per-layer affine/ridge translator whose unembedding table is made of multi-token word prototypes rather than explicit tokenizer rows. Each implicit token entry is a whole-word centroid estimated from final-layer hidden states of real model activations. The lens is evaluated on held-out occurrences of known multi-token word types from WikiText-2.

### Experiment Justification
- Experiment 1: Build and validate a candidate implicit vocabulary from WikiText-2 multi-token words. This establishes that the output classes are not single tokenizer entries and controls for trivial final-subtoken leakage.
- Experiment 2: Collect layer-wise hidden states from a real open causal LM. This is necessary because the hypothesis concerns real model internals, not simulated behavior.
- Experiment 3: Compare the proposed ITL to explicit-token and compositional embedding baselines. This tests whether an implicit-token output space adds information beyond subtoken identity or mean subtoken embeddings.
- Experiment 4: Analyze layer-wise decoding and statistical confidence. This connects the method to token erasure/inner-lexicon claims by asking where full-word identity becomes decodable.

## Research Question
Can we construct a logit-lens-style decoder that maps intermediate LLM hidden states to implicit vocabulary items, operationalized as multi-token word types that are not explicit tokenizer entries?

## Background and Motivation
The paper "Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs" reports that last-token representations of named entities and multi-token words rapidly lose constituent-token information in early layers, suggesting that models convert token sequences into higher-level latent items. "From Tokens to Words" similarly argues that LLMs reconstruct whole-word representations from subword sequences. A direct logit lens cannot output such items because its output matrix is restricted to tokenizer vocabulary rows; this plan tests a vocabulary-like extension whose rows are latent word prototypes.

## Hypothesis Decomposition
- H1: Multi-token word occurrences cluster by whole-word identity in final-layer hidden-state space strongly enough to define reusable implicit-token prototypes.
- H2: Earlier-layer hidden states can be translated into that implicit-token prototype space with a simple affine/ridge map.
- H3: The implicit-token lens predicts held-out whole-word identities better than baselines based only on explicit final-subtoken identity or mean subtoken embeddings.
- H4: Decoding accuracy varies systematically by layer, with middle/late layers outperforming very early layers.

Independent variables are lens method, layer, and candidate vocabulary construction. Dependent variables are top-1/top-5 whole-word retrieval accuracy, mean reciprocal rank, confidence intervals, and paired method comparisons.

## Proposed Methodology

### Approach
Use a modern open HuggingFace causal LM with accessible hidden states, preferably `Qwen/Qwen2.5-0.5B` for a 2024-era real model that fits on the available GPUs. From WikiText-2, extract frequent alphabetic word types that tokenize into at least two tokens under the model tokenizer. For each occurrence, feed the context ending at the target word and store the hidden state at the final subtoken position for every layer.

Construct the implicit vocabulary from training occurrences: for each word type, average normalized hidden states to form prototype rows. The primary lens uses one layer-local prototype table per layer, analogous to an implicit vocabulary/unembedding matrix for whole-word entries. A ridge-to-final-prototype translator, closer to Tuned Lens' affine translator, is retained as an ablation; a smoke test showed it is underdetermined with only 10 training contexts per word.

### Experimental Steps
1. Load WikiText-2 and Qwen tokenizer; extract candidate multi-token word occurrences with per-word train/validation/test splits.
2. Filter candidate words to reduce trivial leakage: prefer final-subtoken groups with at least two word types, and require enough occurrences per word.
3. Collect hidden states in batches on GPU with fixed random seed and deterministic splits.
4. Build final-layer word prototypes from training occurrences.
5. Train/evaluate per-layer implicit prototype tables and ridge-to-final-prototype translators on validation/test occurrences.
6. Evaluate baselines on the same splits: final-subtoken majority baseline, mean-subtoken embedding lens, raw final-prototype nearest-neighbor without translator, and explicit LM-head next-token diagnostic.
7. Generate layer-wise plots, bootstrap confidence intervals, and paired permutation/McNemar-style tests for the preregistered middle layer and best validation-selected ITL layer.

### Baselines
- Majority class baseline: most frequent implicit word in the training set.
- Final-subtoken majority baseline: predict the most frequent training word sharing the target occurrence's final tokenizer ID.
- Mean-subtoken embedding lens: score a layer state against the mean input embedding of each candidate word's subtoken sequence.
- Raw prototype lens: directly compare layer states to final-layer word prototypes without a trained translator.
- Ridge-to-final implicit lens: translate each layer into final-layer prototype space; retained as an ablation because it may overfit when per-word sample counts are small.
- LM-head diagnostic: apply the ordinary explicit vocabulary head to intermediate states and measure whether top predictions recover the current whole word; expected to be weak because the LM head predicts next-token distributions, not implicit current-word classes.

### Evaluation Metrics
- Top-1 and top-5 implicit word retrieval accuracy.
- Mean reciprocal rank for the correct implicit word.
- Layer of peak validation/test performance.
- Bootstrap 95% confidence intervals for accuracy.
- Paired accuracy differences and approximate randomization p-values between ITL and strongest baseline.

### Statistical Analysis Plan
The primary confirmatory comparison is ITL versus the strongest non-ITL baseline at the validation-selected layer, evaluated on the held-out test split. Use paired bootstrap confidence intervals for the accuracy difference and an approximate paired randomization test with alpha = 0.05. Layer-wise comparisons are descriptive and will be treated as exploratory unless corrected with Benjamini-Hochberg false discovery rate control.

## Expected Outcomes
Support for the hypothesis: ITL substantially exceeds subtoken and mean-embedding baselines on held-out occurrences, with a positive confidence interval for the paired accuracy difference. Partial support: final-layer prototypes decode well but early/mid-layer translators do not, implying implicit tokens exist but the proposed lens is insufficient. Refutation for this setup: ITL performs no better than subtoken/compositional baselines or only succeeds through final-subtoken leakage.

## Timeline and Milestones
- Resource review and planning: complete before implementation.
- Environment and data validation: 10-20 minutes.
- Candidate extraction and hidden-state collection: 20-45 minutes depending on model download/cache state.
- Lens training and evaluation: 20-40 minutes.
- Analysis, visualization, and documentation: 30-60 minutes.

## Potential Challenges
- Model download or tokenizer compatibility may fail; fallback is `EleutherAI/pythia-160m` or `gpt2`, explicitly documented as a smaller/older mechanistic baseline.
- Qwen's large tokenizer may make many common words single tokens; mitigation is selecting rare but frequent-enough WikiText words and reducing the candidate count if needed.
- Final-subtoken identity may trivially identify words; mitigation is grouping candidate words by shared final subtoken where possible and reporting the final-subtoken majority baseline.
- Hidden-state extraction can be memory intensive; mitigation is using batch size 32-64 on an RTX A6000, storing only final-position layer states, and using float32/float16 as appropriate.

## Success Criteria
The research succeeds if it produces a reproducible pipeline, real hidden-state experiments, documented candidate implicit vocabulary, quantitative comparison against baselines, statistical uncertainty estimates, and a clear conclusion about whether this prototype implicit-token logit lens works on the selected model/dataset.
