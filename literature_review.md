# Literature Review: Logit Lens with Implicit Tokens

## Review Scope

### Research Question
Can large language models be interpreted as having implicit token-like representations outside the explicit tokenizer vocabulary, and can a logit-lens-style method decode or use these implicit tokens?

### Inclusion Criteria
- Methods that project hidden states, weights, gradients, or residual-stream components into vocabulary space.
- Evidence that LLMs internally reconstruct word, phrase, or future-token representations beyond one tokenizer item.
- Soft/virtual prompt methods where continuous vectors function as token-like inputs.
- Code or datasets that can support experiments on hidden-state decoding and implicit vocabulary.

### Exclusion Criteria
- General interpretability work without a decoding/projection method.
- Tokenizer engineering papers without model-internal evidence.
- Very large datasets unsuitable for this resource-gathering phase.

### Search Log

| Date | Query | Source | Result |
|------|-------|--------|--------|
| 2026-06-29 | `logit lens implicit tokens large language models` | paper-finder | Timed out; manual fallback used |
| 2026-06-29 | `tuned lens logit lens transformer interpretability` | paper-finder | Timed out |
| 2026-06-29 | `soft prompt tuning continuous prompt tokens language models` | paper-finder | Timed out |
| 2026-06-29 | `inner lexicon latent vocabulary large language models` | paper-finder | Timed out |
| 2026-06-29 | logit lens, tuned lens, inner lexicon, Patchscopes, Future Lens, soft prompt tuning | arXiv, ACL Anthology, OpenReview, HuggingFace, GitHub | 14 PDFs downloaded, 9 repos cloned |

## Research Area Overview

The relevant literature splits into three connected lines. First, logit-lens methods decode intermediate hidden states into distributions over the explicit model vocabulary. Tuned Lens improves reliability by learning per-layer affine translators before the unembedding. Second, Patchscopes and Future Lens show that hidden states may contain information not well represented by a single next-token distribution, including entity identity and several future tokens. Third, prompt tuning and prefix tuning show that continuous vectors can act like "virtual tokens" even though they are not tokenizer entries. The inner-lexicon paper directly connects these lines by showing that LLMs reconstruct whole-word representations from subword inputs and can use those representations for finetuning-free vocabulary expansion.

## Key Papers

### From Tokens to Words: On the Inner Lexicon of LLMs
- Authors: Kaplan, Oren, Reif, Schwartz
- Year/Source: ICLR 2025, arXiv:2410.05864
- Key contribution: Presents evidence that LLMs compose subword sequences into whole-word hidden representations at the last token and maintain a latent vocabulary beyond the tokenizer.
- Methodology: Uses k-NN probes for word vs. nonword separability, input-embedding logit lens for artificially split single-token words, Patchscopes for multi-token words, FFN vocabulary projections, attention analysis, and targeted FFN ablations.
- Datasets: Gutenberg word/nonword set, WikiText-103, PubMed abstracts, Arabic Wiki40B.
- Results: Last-token representations become decodable as full words after early/middle layers; 23% of multi-token words are never successfully decoded. FFN updates often retrieve full-word concepts before they appear in hidden states; ablating identified FFN updates sharply reduces retrieval. Detokenized vectors can initialize new input/output embeddings with little loss in next-token accuracy.
- Relevance: This is the central prior work. It operationalizes "implicit vocabulary" and suggests the right first experiment: extend logit/tuned lens to decode continuous whole-word states instead of only explicit vocabulary IDs.

### Eliciting Latent Predictions from Transformers with the Tuned Lens
- Authors: Belrose et al.
- Year/Source: 2023, arXiv:2303.08112
- Key contribution: Introduces trained affine translators from each layer's representation basis to the final unembedding basis.
- Methodology: Minimize KL divergence between lens output and final model output on frozen models; compare perplexity/bias with logit lens; introduce causal basis extraction.
- Datasets/Baselines: Pile, RedPajama, Anthropic HH, LM evaluation tasks; logit lens baseline.
- Results: Tuned lens is more predictive and less biased than direct logit lens across model families; trajectories can support prompt-injection detection and example-difficulty analysis.
- Relevance: A strong baseline and likely scaffold for an implicit-token lens. The affine translator idea can be reused to map hidden states into an implicit-token embedding space.

### Patchscopes
- Authors: Ghandeharioun et al.
- Year/Source: ICML 2024, arXiv:2401.06102
- Key contribution: Generalizes vocabulary projection and activation patching into source-model/target-prompt configurations that decode hidden representations in natural language.
- Methodology: Patch hidden states into target prompts designed for token identity, feature extraction, entity description, cross-model explanation, and reasoning correction.
- Datasets/Baselines: Pile eval, WikiText-103, PopQA, entity/relation tasks; compares logit lens, tuned lens, linear probes.
- Results: Token Identity Patchscope is training-free and robust across layers; Patchscopes can inspect early layers and provide more expressive outputs than token probabilities.
- Relevance: Provides the main alternative when explicit-vocabulary projection is too restrictive. An implicit-token lens could use Patchscope outputs as labels or validators.

### Future Lens
- Authors: Pal, Sun, Yuan, Wallace, Bau
- Year/Source: CoNLL 2023
- Key contribution: Shows individual hidden states can encode information about tokens beyond the immediate next token.
- Methodology: Linear hidden-state prediction, direct vocabulary prediction, fixed-prompt causal intervention, and learned soft-prompt causal intervention.
- Datasets/Baselines: 100k Pile token samples for training, 1k test token samples; GPT-J-6B.
- Results: Learned prompts recover future-token information best; middle-layer states can predict token N+1 with 48.4% precision@1 and still recover tokens farther ahead above baselines.
- Relevance: Important if an implicit token corresponds to a multi-token phrase or future continuation, not just one explicit token.

### Backward Lens
- Authors: Katz, Belinkov, Geva, Wolf
- Year/Source: EMNLP 2024
- Key contribution: Projects gradients and VJPs from the backward pass into vocabulary space.
- Methodology: Shows gradients are low-rank combinations of forward inputs and backward VJPs; projects those spanning vectors with logit lens; proposes "imprint and shift" for how MLP updates store edited information.
- Datasets/Baselines: CounterFact-style edits, GPT-2 and Llama-2 examples.
- Results: Backward VJPs often project to editing targets or related concepts; update norms reveal where model editing concentrates.
- Relevance: Useful for interpreting the training dynamics of learned implicit tokens, translator matrices, or soft prompts.

### Prompt Tuning, Prefix Tuning, and P-Tuning v2
- Papers: Lester et al. 2021; Li and Liang 2021; Liu et al. 2022
- Key contribution: Learned continuous vectors can function like task-conditioning tokens while the model remains frozen.
- Methodology: Learn soft prompts or deep prefixes through backpropagation; compare to full fine-tuning and discrete prompts.
- Datasets/Baselines: SuperGLUE/GLUE, table-to-text, summarization, sequence labeling; full fine-tuning and prompt design baselines.
- Results: Prompt tuning becomes competitive at large scale; prefix tuning stores roughly 0.1% task-specific parameters; P-Tuning v2 improves universality for NLU.
- Relevance: These are the clearest examples of token-like vectors outside the vocabulary matrix. They provide baselines for creating artificial implicit tokens and testing whether a lens can decode them.

### AutoPrompt
- Authors: Shin et al.
- Year/Source: EMNLP 2020
- Key contribution: Gradient-guided search for discrete prompt tokens.
- Relevance: Discrete-token baseline against which continuous implicit-token methods should be compared.

### FFN Key-Value Memories and Logit Prisms
- Papers: Geva et al. 2021; Nguyen 2024
- Key contribution: FFNs and model components can be interpreted by projecting values or contributions into vocabulary space.
- Relevance: The inner-lexicon paper suggests FFNs retrieve whole-word concepts. Component-level decomposition may localize where implicit-token evidence is written.

## Common Methodologies

- Direct logit lens: apply the final unembedding to intermediate hidden states. Fast but brittle and biased across layers/models.
- Tuned lens: train affine translators per layer using KL to final logits. Strong baseline for explicit vocabulary decoding.
- Patchscopes: patch a representation into a target prompt and let an LM verbalize or continue. More expressive and better for hidden states not aligned with explicit tokens.
- Input-embedding lens: compare hidden states to input embedding vectors rather than output unembedding, useful for current-token identity rather than next-token prediction.
- Soft/virtual token learning: learn continuous prompt vectors, prefixes, or vocabulary-expansion embeddings while the model is frozen.
- FFN and gradient vocabulary projection: project FFN updates, values, or VJPs to identify concept directions and storage mechanisms.

## Standard Baselines

- Direct logit lens over explicit vocabulary.
- Tuned lens over explicit vocabulary.
- Patchscope token-identity decoding.
- Mean embedding for multi-token words, as used in vocabulary expansion baselines.
- Discrete prompt search, e.g. AutoPrompt.
- Soft prompt/prefix tuning via PEFT.

## Evaluation Metrics

- Top-1 retrieval accuracy for recovered token, word, or phrase.
- Precision@1 and surprisal for next/future token prediction.
- KL divergence to final-layer output for lens training.
- Token-level accuracy for vocabulary expansion.
- Ablation effect on retrieval or downstream factual prediction.
- Layer of first successful retrieval and cumulative retrieval rate.

## Datasets in the Literature and Workspace

- WikiText-103/WikiText-2: language modeling contexts and word splitting. Workspace includes WikiText-2 raw for small experiments.
- LAMBADA: long-context final-word prediction. Workspace includes OpenAI test split.
- GLUE/SST-2: compact sentiment task for prompt tuning and AutoPrompt-style baselines.
- PubMed abstracts and Arabic Wiki40B: useful later for domain and multilingual vocabulary expansion; not downloaded due size/scope.
- Pile: used by Tuned Lens, Patchscopes, and Future Lens. Not downloaded because it is very large; use streaming or sampled subsets if needed.

## Gaps and Opportunities

- Existing logit-lens methods decode into explicit vocabularies; the inner-lexicon paper shows some hidden states correspond to whole words that are not explicit tokens.
- Patchscopes can decode these hidden states but do not produce a reusable vocabulary-like lens by themselves.
- Soft prompt vectors act as implicit tokens but are usually evaluated on task performance, not decoded or aligned to interpretable latent vocabulary items.
- Current evidence is strongest for words; phrase-level, entity-level, and learned soft-token vocabularies remain open.

## Recommendations for Experiments

- Start with a small model available in HuggingFace, such as GPT-2 small or Pythia small, then move to Llama-style models only after the pipeline works.
- Build a candidate implicit vocabulary from multi-token words in WikiText-2: collect word strings, token spans, last-token hidden states, and context.
- Compare four decoders: direct logit lens, tuned lens, input-embedding nearest neighbor, and Patchscope token identity.
- Train an affine "implicit lens" from hidden states to a candidate implicit-token embedding table. Use Patchscope success or exact word identity as labels.
- Add soft prompt vectors trained on SST-2 and test whether the same lens can assign them stable nearest words, task labels, or latent concepts.
- Evaluate with retrieval accuracy, layer of first retrieval, KL to final logits, and downstream ablation/patching effects.

