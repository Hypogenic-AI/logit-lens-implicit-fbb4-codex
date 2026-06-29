# Downloaded Datasets

Data files are stored locally for immediate experimentation and are excluded from git by `datasets/.gitignore`. Small sample files are kept under each dataset's `samples/` directory.

## Dataset 1: WikiText-2 Raw

### Overview
- Source: HuggingFace `Salesforce/wikitext`, config `wikitext-2-raw-v1`
- Size: train 36,718 rows; validation 3,760; test 4,358
- Format: HuggingFace DatasetDict saved with `save_to_disk`
- Location: `datasets/wikitext2_raw/`
- Task: language modeling, hidden-state/logit-lens probing, multi-token word analysis
- License: WikiText is distributed from verified Good and Featured Wikipedia articles; HuggingFace card lists CC BY-SA.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
dataset.save_to_disk("datasets/wikitext2_raw")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext2_raw")
```

### Sample Data

See `datasets/wikitext2_raw/samples/examples.json`.

### Notes
- WikiText rows include empty strings and section headings. Filter `text.strip()` before token-level experiments.
- Recommended for reproducing the inner-lexicon paper's single-token splitting and context-window experiments at small scale.

## Dataset 2: LAMBADA OpenAI

### Overview
- Source: HuggingFace `EleutherAI/lambada_openai`
- Size: test 5,153 rows
- Format: HuggingFace DatasetDict saved with `save_to_disk`
- Location: `datasets/lambada_openai/`
- Task: long-context final-word and next-token prediction
- License: check upstream HuggingFace card before redistribution beyond local use.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("EleutherAI/lambada_openai")
dataset.save_to_disk("datasets/lambada_openai")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/lambada_openai")
```

### Sample Data

See `datasets/lambada_openai/samples/examples.json`.

### Notes
- Good benchmark for whether an implicit-token lens can recover a final word or phrase from a preceding hidden state.
- Only a test split is provided in this variant.

## Dataset 3: GLUE SST-2

### Overview
- Source: HuggingFace `nyu-mll/glue`, config `sst2`
- Size: train 67,349 rows; validation 872; test 1,821
- Format: HuggingFace DatasetDict saved with `save_to_disk`
- Location: `datasets/glue_sst2/`
- Task: sentiment classification, soft-prompt sanity checks, AutoPrompt/P-Tuning baseline
- License: see GLUE upstream terms and SST-2 source terms.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("nyu-mll/glue", "sst2")
dataset.save_to_disk("datasets/glue_sst2")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/glue_sst2")
```

### Sample Data

See `datasets/glue_sst2/samples/examples.json`.

### Validation Summary

Validated with:

```python
from datasets import load_from_disk
for name in ["wikitext2_raw", "lambada_openai", "glue_sst2"]:
    ds = load_from_disk(f"datasets/{name}")
    print(name, {k: len(v) for k, v in ds.items()})
```

Local sizes are small: WikiText-2 raw ~7.8 MB, SST-2 ~3.3 MB, LAMBADA ~1.2 MB.

