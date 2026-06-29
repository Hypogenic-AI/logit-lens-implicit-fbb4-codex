# Working Title

Layer-Local Implicit Token Lenses Recover Multi-Token Words

## Abstract Sketch

- Context/Problem: Standard logit lenses decode intermediate language-model states only into explicit tokenizer items.
- Gap/Challenge: Prior work argues that LLMs build implicit word-like representations, but existing lenses do not expose these items as a reusable vocabulary.
- Our approach: Build an implicit token lens (ITL) from layer-local prototypes for 128 WikiText-2 multi-token word types in Qwen2.5-0.5B.
- Key results: The validation-selected ITL reaches 19.53% test top-1 accuracy, above 12.50% final-subtoken majority and 0.78% chance; it also beats the strongest raw final-prototype baseline by 5.47 points.
- Significance: A supervised implicit vocabulary table can decode whole-word identity beyond subtoken leakage, but a Tuned-Lens-style ridge translator is not reliable at this scale.

## 1. Introduction

- Hook: Tokenizer boundaries limit what a logit lens can report about internal language-model states.
- Background: Token erasure and inner-lexicon work suggest that multi-token words are composed into whole-word representations.
- Gap: Tuned Lens and standard logit lenses output explicit tokenizer IDs; Patchscopes verbalize activations but do not provide a reusable implicit vocabulary table.
- Approach: Define implicit tokens as multi-token word types, build layer-local prototype rows, and score held-out occurrences by cosine similarity.
- Quantitative preview: ITL layer 5 reaches 19.53% test top-1, 25.59% top-5, and 0.240 MRR; it beats final-subtoken majority by 7.03 points with a 95% bootstrap CI of [2.53, 11.72].
- Contributions:
  - Propose a prototype implicit-token output space for logit-lens-style decoding.
  - Construct a leakage-controlled candidate vocabulary with 16 suffix groups and 8 words per group.
  - Compare against final-subtoken, mean-embedding, raw final-prototype, and ridge-to-final baselines.
  - Analyze layer-wise behavior and representative errors.

## 2. Related Work

- Vocabulary-space lenses: logit lens, Tuned Lens, Future Lens; our output space is whole-word labels instead of explicit token IDs.
- Implicit lexical representations: token erasure and inner-lexicon results motivate the candidate vocabulary and layer-wise analysis.
- Natural-language and continuous-token inspection: Patchscopes and prompt/prefix tuning show hidden states and continuous vectors can carry token-like information, but do not evaluate fixed implicit vocabulary retrieval.
- Model/data citations: Qwen2.5 and WikiText-2 support the experimental setup.

## 3. Methodology

- Problem setup: Given hidden state h_l at the final subtoken position of a target word, retrieve the whole word from an implicit vocabulary V_I.
- Candidate vocabulary: 128 lowercased alphabetic word types, all tokenized into at least two Qwen pieces; 16 final-subtoken groups, 8 words per group.
- Data: 10 train, 2 validation, and 4 test occurrences per word, for 2048 total examples.
- ITL: For each layer and word, average normalized training states into a normalized prototype row; score by cosine similarity.
- Baselines: final-subtoken majority, majority word, raw final prototype, mean subtoken embedding, ridge-to-final ITL, and an explicit final-subtoken diagnostic.
- Metrics: top-1, top-5, MRR; validation-selected layers; paired bootstrap, randomization, and McNemar tests.

## 4. Results

- Main table: ITL best validation/test top-1 and MRR among whole-word methods; raw final prototype has higher top-5 but lower top-1.
- Statistical table: ITL beats final-subtoken majority, raw final prototype, and mean subtoken embedding with positive confidence intervals.
- Layer plots: ITL peaks at layer 5; raw final prototypes peak late; explicit final-subtoken diagnostic is strongest at layer 0.
- Error table: Confusions often stay within suffix groups, but cross-group semantic/contextual errors occur.

## 5. Discussion

- Interpretation: Results support a narrow decodability claim: layer-local whole-word prototypes expose information beyond final-subtoken identity.
- Limitations: Fixed supervised vocabulary, one small model, one English dataset, occurrence-level splits, possible domain/topic correlations, no causal intervention, underpowered ridge ablation.
- Implications: Implicit token tables can complement tokenizer-bound lenses and motivate unsupervised discovery plus causal validation.

## 6. Conclusion

- Summary: Built and evaluated a logit-lens-style implicit vocabulary table for multi-token words.
- Key insight: Layer-local prototypes work better than final-space translation at this data scale.
- Future work: Discover implicit items by clustering token-erasure footprints, validate with Patchscopes, and scale to larger models, named entities, phrases, and causal interventions.

## Figure and Table Plan

| ID | Type | Location | Description | Status |
| --- | --- | --- | --- | --- |
| Figure 1 | Method schematic | Methodology | Candidate construction, prototype building, cosine decoding | Planned |
| Table 1 | Setup | Methodology | Model, dataset, vocabulary, splits, compute | Planned |
| Table 2 | Main results | Results | Validation-selected methods and test metrics | Planned |
| Table 3 | Statistical comparisons | Results | Accuracy differences, CIs, and p-values | Planned |
| Figure 2 | Layer plots | Results | Top-1 and MRR by layer | Planned |
| Table 4 | Errors | Results | Representative ITL failures | Planned |

## Citation Checklist

| Topic | Paper | Status |
| --- | --- | --- |
| Token erasure | Feucht et al. 2024 | Planned |
| Inner lexicon | Kaplan et al. 2025 | Planned |
| Tuned Lens | Belrose et al. 2023 | Planned |
| Patchscopes | Ghandeharioun et al. 2024 | Planned |
| Future Lens | Pal et al. 2023 | Planned |
| Prompt/prefix tuning | Lester et al. 2021; Li and Liang 2021; Liu et al. 2022 | Planned |
| Dataset/model | Merity et al. 2016; Qwen Team 2025 | Planned |
