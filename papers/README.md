# Downloaded Papers

PDFs are stored in this directory. Chunked PDFs and manifests for skimming/deep reading are in `papers/pages/`.

1. [Eliciting Latent Predictions from Transformers with the Tuned Lens](2303.08112_tuned_lens.pdf)
   - Authors: Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, Jacob Steinhardt
   - Year: 2023; arXiv:2303.08112
   - Why relevant: Core logit-lens successor. Trains affine per-layer translators to decode hidden states into vocabulary distributions.

2. [From Tokens to Words: On the Inner Lexicon of LLMs](2410.05864_tokens_to_words_inner_lexicon.pdf)
   - Authors: Guy Kaplan, Matanel Oren, Yuval Reif, Roy Schwartz
   - Year: 2025; ICLR; arXiv:2410.05864
   - Why relevant: Most direct match. Shows LLMs reconstruct whole-word hidden representations from subword tokens and argues for a latent vocabulary beyond the tokenizer.

3. [Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models](2401.06102_patchscopes.pdf)
   - Authors: Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, Mor Geva
   - Year: 2024; ICML; arXiv:2401.06102
   - Why relevant: General method for decoding hidden representations through patched target prompts, useful when a vector does not map cleanly to one vocabulary token.

4. [Future Lens: Anticipating Subsequent Tokens from a Single Hidden State](2023.conll-1.37_future_lens.pdf)
   - Authors: Koyena Pal, Jiuding Sun, Andrew Yuan, Byron Wallace, David Bau
   - Year: 2023; CoNLL
   - Why relevant: Shows single hidden states can encode multiple future tokens; useful for implicit tokens that may represent multi-token continuations.

5. [Backward Lens: Projecting Language Model Gradients into the Vocabulary Space](2024.emnlp-main.142_backward_lens.pdf)
   - Authors: Shahar Katz, Yonatan Belinkov, Mor Geva, Lior Wolf
   - Year: 2024; EMNLP Best Paper
   - Why relevant: Extends vocabulary projection to gradients and backward-pass vectors; useful for interpreting learned implicit-token or lens parameters.

6. [LogitLens4LLMs: Extending Logit Lens Analysis to Modern Large Language Models](2503.11667_logitlens4llms.pdf)
   - Author: Zhenyu Wang
   - Year: 2025; arXiv:2503.11667
   - Why relevant: Toolkit paper for applying logit lens to current HuggingFace Llama and Qwen models.

7. [The Power of Scale for Parameter-Efficient Prompt Tuning](2104.08691_prompt_tuning_scale.pdf)
   - Authors: Brian Lester, Rami Al-Rfou, Noah Constant
   - Year: 2021; EMNLP; arXiv:2104.08691
   - Why relevant: Establishes learned soft prompts as continuous "tokens" that condition frozen LMs.

8. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](2101.00190_prefix_tuning.pdf)
   - Authors: Xiang Lisa Li, Percy Liang
   - Year: 2021; ACL; arXiv:2101.00190
   - Why relevant: Treats learned prefixes as virtual tokens attended to by later tokens.

9. [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](2110.07602_p_tuning_v2.pdf)
   - Authors: Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, Jie Tang
   - Year: 2022; ACL; arXiv:2110.07602
   - Why relevant: Strong baseline for deep continuous prompt tuning across NLU tasks.

10. [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](2010.15980_autoprompt.pdf)
    - Authors: Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, Sameer Singh
    - Year: 2020; EMNLP; arXiv:2010.15980
    - Why relevant: Discrete prompt-search baseline for comparing implicit/continuous token methods.

11. [Transformer Feed-Forward Layers Are Key-Value Memories](2012.14913_ffn_key_value_memories.pdf)
    - Authors: Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy
    - Year: 2021; EMNLP; arXiv:2012.14913
    - Why relevant: Shows FFN values induce output vocabulary distributions, a mechanism likely involved in reconstructing implicit lexical items.

12. [Universality and Limitations of Prompt Tuning](2305.18787_universality_limitations_prompt_tuning.pdf)
    - Authors: Yihan Wang, Jatin Chauhan, Wei Wang, Cho-Jui Hsieh
    - Year: 2023; NeurIPS; arXiv:2305.18787
    - Why relevant: Theoretical capabilities and limits of soft-prompt tokens.

13. [When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations](2310.19698_prompting_prefix_tuning_theory.pdf)
    - Authors: Aleksandar Petrov, Philip H. S. Torr, Adel Bibi
    - Year: 2024; ICLR; arXiv:2310.19698
    - Why relevant: Shows context-based tuning can be less expressive than full fine-tuning despite continuous embeddings.

14. [Logit Prisms: Decomposing Transformer Outputs for Mechanistic Interpretability](logit_prisms_nguyen_2024.pdf)
    - Author: Thong T. Nguyen
    - Year: 2024
    - Why relevant: Decomposes logit contributions from residual, attention, and MLP components; useful for locating sources of implicit-token evidence.

