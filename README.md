# The Culture Within: Towards Mechanistic Understanding of Cultural Reasoning in LLMs

Samee Arif¹, Furkan Danisman², Angana Borah¹, Honglak Lee¹˒³, Dehan Kong²,
Lajanugen Logeswaran³, Moontae Lee³, Zhijing Jin², Rada Mihalcea¹

¹ University of Michigan, ² University of Toronto, ³ LG AI Research

- **Hugging Face collection:** [CultureLens](https://huggingface.co/collections/sameearif/culturelens-6ace88112ea328538a2636aa), which hosts the benchmark and all released adapters.

## Overview

This repository contains the code, data and results for a study of how large
language models use cultural information. It covers the CultureLens benchmark,
response generation for three models under eight evidence conditions,
LLM-as-a-Judge evaluation, a mechanistic analysis with probes and causal
interventions, and LoRA fine-tuning that teaches a model to resist a
contradictory norm.

CultureLens has 9,261 culturally grounded scenarios across 75 countries. Each
record pairs a documented norm with a scenario, two response options, and
evidence that can be shown or withheld independently: the country, a value, the
true norm, and a contradictory norm.

## Released artifacts

| Artifact | Hugging Face | Description |
|---|---|---|
| Benchmark | [sameearif/CultureLens](https://huggingface.co/datasets/sameearif/CultureLens) | 9,261 records, the train, validation and test splits, and the fine-tuning targets |
| Llama-3.1-8B SFT | [sameearif/CultureLens-Llama-3.1-8B-SFT](https://huggingface.co/sameearif/CultureLens-Llama-3.1-8B-SFT) | Trained on the country-only prompt |
| Llama-3.1-8B SFT + CF | [sameearif/CultureLens-Llama-3.1-8B-SFT-CF](https://huggingface.co/sameearif/CultureLens-Llama-3.1-8B-SFT-CF) | Adds value, true-norm and contradictory-norm prompts |
| Llama-3.1-8B SFT + CF + Request | [sameearif/CultureLens-Llama-3.1-8B-SFT-CF-Request](https://huggingface.co/sameearif/CultureLens-Llama-3.1-8B-SFT-CF-Request) | Adds genuine-request prompts, so the model resists false norms and still follows a real request |
| Gemma-3-12B SFT | [sameearif/CultureLens-Gemma-3-12B-SFT](https://huggingface.co/sameearif/CultureLens-Gemma-3-12B-SFT) | Trained on the country-only prompt |
| Gemma-3-12B SFT + CF | [sameearif/CultureLens-Gemma-3-12B-SFT-CF](https://huggingface.co/sameearif/CultureLens-Gemma-3-12B-SFT-CF) | Adds value, true-norm and contradictory-norm prompts |
| Qwen-3.5-27B SFT | [sameearif/CultureLens-Qwen-3.5-27B-SFT](https://huggingface.co/sameearif/CultureLens-Qwen-3.5-27B-SFT) | Trained on the country-only prompt |
| Qwen-3.5-27B SFT + CF | [sameearif/CultureLens-Qwen-3.5-27B-SFT-CF](https://huggingface.co/sameearif/CultureLens-Qwen-3.5-27B-SFT-CF) | Adds value, true-norm and contradictory-norm prompts |

All models are LoRA adapters of rank 16 for the instruction-tuned base model.

```python
from datasets import load_dataset

benchmark = load_dataset("sameearif/CultureLens")   # train, validation, test_seen, test_unseen
```

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "sameearif/CultureLens-Llama-3.1-8B-SFT-CF")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

## What the benchmark measures

The paper asks three things:

1. **Behavior.** How do country, value and norm evidence change what a model
   answers, especially when they conflict?
2. **Mechanism.** When a contradictory norm overrides a correct country-based
   answer, where in the model does this happen, and is the model's knowledge of
   the country lost or overridden?
3. **Mitigation.** Can fine-tuning make a model resist a contradictory norm
   without losing what it gains from a true one?

### Evidence conditions

Every record is evaluated under eight conditions, in a two-option choice
(constrained) and in open-ended generation (unconstrained).

| Condition | Name in files | What the prompt adds to the scenario |
|---|---|---|
| No-evidence | `base` | Nothing |
| Country-only | `country_only` | The country |
| Value-only | `value_only` | A value |
| Norm-only | `norm_only` | The true norm |
| Country-value | `country_value` | The country and a value |
| Country-norm | `country_norm` | The country and the true norm |
| Country-value-norm | `country_value_norm` | All three |
| Country-contradictory-norm | `country_opposite_norm` | The country and a contradictory norm |

A ninth, diagnostic condition, `opposite_norm_only`, states the contradictory
norm without the country.

### Models

| Role | Models |
|---|---|
| Evaluated | Llama-3.1-8B, Gemma-3-12B, Qwen-3.5-27B |
| Judges | Gemma-4-31B, Qwen-3.6-27B |
| Benchmark construction and teacher | DeepSeek-V4 |

### Headline findings

- **A contradictory norm overrides what the model knows.** Stating one lowers
  aligned answers by 32 to 47 percentage points, even when the model answers
  correctly from the country alone.
- **The default is Anglophone.** With no country given, a model already behaves
  as if it were given the United States.
- **The knowledge is overridden, not lost.** Blocking the norm inside the model
  brings the correct answer back in most cases.
- **Presentation decides compliance.** Models accept a stated norm according to
  how it is presented and not whether it is true. Attributed to a guide, a false
  norm is resisted in under 8% of answers by every model.
- **Resistance can be trained.** Fine-tuning with contradictory-norm examples
  raises resistance to 85 to 93%, including on countries never seen in training.

## Repository layout

```text
data/processed/culturelens/    the benchmark, splits, teacher targets, request prompts
data/processed/references/     country-level World Values Survey distances and geography
prompts/                       every system prompt: construction, judging, coding, teacher
src/benchmark/                 benchmark construction and splitting
src/behavioral_evaluation/     prompt construction, generation, judges, stance coding
src/mechanistic_analysis/      state extraction, probes, interventions, SAE features, figures
src/finetune/                  LoRA fine-tuning and the fine-tuning tables
src/inference/                 the served-model backend shared by generative stages
scripts/                       Slurm and VESSL job scripts for each stage
output/                        model answers, judge labels and result files
human_evaluation/              human ratings, with annotator identities removed
tests/                         unit tests
```

## Setup

Python 3.11 or later.

```bash
git clone https://github.com/sameearif/culturelens.git
cd culturelens
python -m pip install -e '.[figures,dev]'      # tables, figures and tests
export PYTHONPATH=src
bash scripts/unpack_outputs.sh                 # restores six gzipped answer files
```

Add the extras a stage needs: `generation` (vLLM, for answers, judging and
benchmark construction), `mechanistic` (state extraction and interventions),
`finetune` (LoRA training), `grouping` (semantic grouping during construction).

Every generative stage resumes from its own output, so an interrupted run is
restarted by reissuing the same command.

## Workflow

The first two commands below run on a laptop from the committed outputs. The
stages that produce those outputs need GPUs. The job scripts take the repository
root in `WORKSPACE` and a storage directory in `CULTURELENS_STORAGE_ROOT`, and
expect the Slurm account and partition on the command line:

```bash
WORKSPACE=$PWD CULTURELENS_STORAGE_ROOT=/path/to/storage \
  sbatch --account=<account> --partition=<gpu-partition> scripts/behavioral_evaluation/generate.sbatch
```

### Stage 1: Benchmark construction

`benchmark.generate` extracts atomic norms from each Cultural Atlas profile,
writes a scenario with country, true-norm and contradictory-norm evidence,
writes two response options, and assigns refined Schwartz values.
`benchmark.split` makes the splits. See `python -m benchmark.generate --help`.

### Stage 2: Response generation

`behavioral_evaluation.generate` answers every record under every condition in
both settings, with deterministic decoding. Scripts: `scripts/behavioral_evaluation/`.

### Stage 3: Judge evaluation

`behavioral_evaluation.judge` labels each open-ended answer as aligned,
partially aligned or misaligned with the documented norm. The judge never sees
the evidence condition or the contradictory norm. `behavioral_evaluation.stance`
codes what an answer does with a contradictory norm.

### Stage 4: Mechanistic analysis

| Analysis | Module |
|---|---|
| Hidden states at the answer start token | `mechanistic_analysis.activations.extract` |
| Country shift, distances, World Values Survey agreement | `mechanistic_analysis.country_space` |
| Probe for resistance and its baselines | `mechanistic_analysis.readout` |
| The conflict signal | `mechanistic_analysis.interaction`, `mechanistic_analysis.conflict` |
| Attention knockout of the norm | `mechanistic_analysis.interventions.localize` |
| Norm erasure by depth | `mechanistic_analysis.interventions.erasure` |
| Head-level knockout | `mechanistic_analysis.interventions.heads` |
| Sparse autoencoder features and steering | `mechanistic_analysis.sae` |
| Predictors, presentation and stance | `mechanistic_analysis.drivers` |

```bash
python -m mechanistic_analysis.report                 # probe, interaction and intervention tables
python -m mechanistic_analysis.drivers --draws 2000   # predictors, presentation, stance
```

Each figure has its own module, for example:

```bash
M=output/mechanistic_analysis
python -m mechanistic_analysis.knockout_figure --localize-root $M/localize --output figures
python -m mechanistic_analysis.erasure_figure  --erasure-root  $M/erasure  --output figures
python -m mechanistic_analysis.heads_figure    --heads-root    $M/heads    --output figures
python -m mechanistic_analysis.sae_figure      --analysis-root $M/sae/analysis/unconstrained --output figures
python -m mechanistic_analysis.wvs_figure      --geometry-root $M/country_space --references data/processed/references --output figures
python -m mechanistic_analysis.predictors_figure   --drivers $M/report/drivers.json --output figures
python -m mechanistic_analysis.presentation_figure --drivers $M/report/drivers.json --output figures
python -m mechanistic_analysis.stance_figure       --drivers $M/report/drivers.json --output figures
python -m finetune.route_figure --root output/finetune/mechanism --output figures
```

### Stage 5: Fine-tuning

`finetune.sft` trains the adapters. SFT trains on the country-only prompt, and
SFT + CF adds the country-value, country-norm and country-contradictory-norm
prompts. SFT + CF + Request also adds prompts where a person in the scenario
makes a genuine request. Scripts: `scripts/finetuning/`. Evaluation of the fine-tuned models and
the analysis of their internal changes run through
`scripts/mechanistic_analysis/finetune_shift.sbatch`.

| Setting | Value |
|---|---|
| Method | LoRA on the attention and feed-forward projections of every layer |
| Rank, scaling, dropout | 16, 32, 0.05 |
| Optimizer | AdamW, learning rate 1e-4 |
| Schedule | Linear warm-up over 5% of steps, cosine decay |
| Epochs, batch size | 3, 16 |
| Checkpoint | Lowest validation loss |
| Training records | 5,984 from 67 countries |
| Held-out countries | Argentina, Egypt, Ethiopia, Germany, India, Indonesia, Japan, Russia |

```bash
python -m finetune.tables               # main fine-tuning table and the request table
python -m finetune.tables --by-judge    # the same, one judge at a time
```

## Data

| File | Records | Description |
|---|---|---|
| `data/processed/culturelens/records.json` | 9,261 | Every record |
| `splits/train.json` | 5,984 | Training split, 67 countries |
| `splits/validation.json` | 480 | Validation split |
| `splits/test_seen.json` | 1,675 | Test records from the training countries |
| `splits/test_unseen.json` | 1,122 | Test records from the 8 held-out countries |
| `gold/train.country_only.jsonl` | 5,984 | Teacher answers, country alone |
| `gold/train.country_opposite_norm.jsonl` | 5,982 | Teacher answers that correct a contradictory norm |
| `requests/train.json`, `gold/train.request.jsonl` | 4,491 | Request prompts and their teacher answers |

A record looks like this:

```json
{
  "norm_id": "JPN__etiquette__001__012",
  "norm": "the documented norm",
  "values": ["refined Schwartz values, zero to three"],
  "scenario": {
    "situation": "...", "question": "...",
    "country_context": "...", "norm_evidence": "...",
    "opposite_norm_text": "...", "opposite_norm_evidence": "...",
    "value_evidence": "..."
  },
  "options": {"A": "...", "B": "..."}
}
```

The first three letters of `norm_id` are the ISO code of the country.

## Results

| Path | Contents |
|---|---|
| `output/behavioral_evaluation/{constrained,unconstrained}/responses/` | Model answers under every condition |
| `output/behavioral_evaluation/unconstrained/judgments/` | Judge labels, one file per judge |
| `output/mechanistic_analysis/` | Probe scores, country geometry, interventions, SAE analysis, the report tables |
| `output/finetune/mechanism/<run>/` | Answers, judge labels and internal analyses of each fine-tuned model |
| `human_evaluation/` | Ratings of benchmark quality and of the judges |

Result directories use internal run names. They map to the paper and to the
released adapters as follows.

| Directory under `output/finetune/mechanism/` | Name in the paper | Hugging Face |
|---|---|---|
| `base`, `base-gemma-3-12b`, `base-qwen-3.5-27b` | Untuned | the base models |
| `collab-sft-country` | Llama SFT | [CultureLens-Llama-3.1-8B-SFT](https://huggingface.co/sameearif/CultureLens-Llama-3.1-8B-SFT) |
| `collab-sft-cf` | Llama SFT + CF | [CultureLens-Llama-3.1-8B-SFT-CF](https://huggingface.co/sameearif/CultureLens-Llama-3.1-8B-SFT-CF) |
| `collab-gemma-sft-country` | Gemma SFT | [CultureLens-Gemma-3-12B-SFT](https://huggingface.co/sameearif/CultureLens-Gemma-3-12B-SFT) |
| `collab-gemma-sft-cf` | Gemma SFT + CF | [CultureLens-Gemma-3-12B-SFT-CF](https://huggingface.co/sameearif/CultureLens-Gemma-3-12B-SFT-CF) |
| `collab-qwen-sft-country` | Qwen SFT | [CultureLens-Qwen-3.5-27B-SFT](https://huggingface.co/sameearif/CultureLens-Qwen-3.5-27B-SFT) |
| `collab-qwen-sft-cf` | Qwen SFT + CF | [CultureLens-Qwen-3.5-27B-SFT-CF](https://huggingface.co/sameearif/CultureLens-Qwen-3.5-27B-SFT-CF) |
| `sft-cf-request` | Llama SFT + CF + Request | [CultureLens-Llama-3.1-8B-SFT-CF-Request](https://huggingface.co/sameearif/CultureLens-Llama-3.1-8B-SFT-CF-Request) |

## What is not in the repository

| Not included | Why | How to get it |
|---|---|---|
| Hidden states, about 45 GB | Size | Regenerate with `mechanistic_analysis.activations.extract` |
| Adapter weights | Size | The Hugging Face collection above |
| Cultural Atlas source pages | They belong to the Cultural Atlas | https://culturalatlas.sbs.com.au, placed under `data/raw/atlas/` to rebuild the benchmark |
| World Values Survey microdata | Distributed by the WVS Association | Wave 7 cross-national file, placed at `data/raw/wvs/cross_national_wave_7.csv` |

## Ethical considerations

- **Norms are common expectations, not rules.** CultureLens describes norms at
  the level of a country. A documented norm does not hold for every person, and
  it can be read as a stereotype if it is taken as a rule.
- **Contradictory norms are false by design.** They are released so the failure
  can be measured and reduced. Do not use them as cultural guidance, and do not
  train a model to state them as facts.
- **The models are research artifacts.** Their answers about a culture are
  fallible. A model trained to resist a stated norm can also dismiss a person
  who describes a real custom of their own community, or decline a genuine
  request.
- **Scope.** One source, the Cultural Atlas, and English only. The judges are
  language models, validated against human annotators.
- **No personal data.** Annotator identities are removed from the human
  evaluation files.

## License

Code is released under the MIT licence (`LICENSE`). The benchmark, annotations
and outputs are released under CC BY-NC 4.0 for non-commercial research use,
subject to the terms of the Cultural Atlas, with the notices in
`LICENSE-DATA.md`. Each adapter follows the licence of its base model: the
Llama 3.1 Community License, the Gemma Terms of Use, and Apache 2.0 for Qwen.

## Citation

```bibtex
@misc{culturelens2026,
  title  = {The Culture Within: Towards Mechanistic Understanding of Cultural Reasoning in LLMs},
  author = {Arif, Samee and Danisman, Furkan and Borah, Angana and Lee, Honglak and Kong, Dehan and Logeswaran, Lajanugen and Lee, Moontae and Jin, Zhijing and Mihalcea, Rada},
  year   = {2026}
}
```

## Acknowledgments

We thank the annotators who took part in the human evaluations.
