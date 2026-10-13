# CultureLens

Code, benchmark and outputs for **The Culture Within: Towards Mechanistic
Understanding of Cultural Reasoning in LLMs**.

Samee Arif¹, Furkan Danisman², Angana Borah¹, Honglak Lee¹˒³, Dehan Kong²,
Lajanugen Logeswaran³, Moontae Lee³, Zhijing Jin², Rada Mihalcea¹

¹ University of Michigan, ² University of Toronto, ³ LG AI Research

The benchmark is also on Hugging Face:
[sameearif/CultureLens](https://huggingface.co/datasets/sameearif/CultureLens).

CultureLens is a benchmark of 9,261 culturally grounded scenarios across 75
countries. Each record pairs a documented norm with a scenario, two response
options, and independently controllable evidence: the country, a value, the
true norm, and a contradictory norm. The paper uses it to study three things in
Llama-3.1-8B, Gemma-3-12B and Qwen-3.5-27B:

1. **Behavior.** How country, value and norm evidence change what a model
   answers, in a two-option choice and in open-ended generation.
2. **Mechanism.** Where a stated norm enters the answer, and whether the model
   loses or keeps its knowledge of the country when it follows a contradictory
   norm.
3. **Mitigation.** Whether fine-tuning can make a model resist a contradictory
   norm without losing what it gains from a true one.

## Repository layout

| Path | Contents |
|---|---|
| `data/processed/culturelens/` | The benchmark: `records.json`, the train, validation and test splits, the teacher's reference answers, and the request prompts |
| `data/processed/references/` | Country-level World Values Survey distances and geography used in the analysis |
| `src/benchmark/` | Benchmark construction and splitting |
| `src/behavioral_evaluation/` | Prompt construction, answer generation, the two judges, stance coding |
| `src/mechanistic_analysis/` | State extraction, probes, the conflict signal, attention knockout, norm erasure, head-level knockout, sparse autoencoder features, and the figure scripts |
| `src/finetune/` | LoRA fine-tuning (SFT, SFT + CF, and the request variant) and the fine-tuning tables |
| `prompts/` | Every system prompt used for construction, judging, coding and the teacher |
| `scripts/` | Slurm and VESSL job scripts that run each stage |
| `output/` | Model answers, judge labels and analysis results behind every table and figure |
| `human_evaluation/` | Human ratings of benchmark quality and of the judges, with annotator identities removed |
| `tests/` | Unit tests |

## Installation

Python 3.11 or later.

```bash
python -m pip install -e '.[figures,dev]'      # tables, figures and tests
export PYTHONPATH=src
```

Add the extras a stage needs: `generation` (vLLM, for answers, judging and
benchmark construction), `mechanistic` (state extraction and interventions),
`finetune` (LoRA training), `grouping` (semantic grouping during construction).

## Reproducing the tables and figures

Everything below runs on a laptop from the committed outputs. No GPU and no
model download is needed.

```bash
bash scripts/unpack_outputs.sh          # restores six gzipped answer files
```

```bash
python -m finetune.tables               # main fine-tuning table and the request table
python -m finetune.tables --by-judge    # the same, one judge at a time
python -m mechanistic_analysis.report   # probe, interaction and intervention tables
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

The docstring at the top of every module gives its exact command, and
`python -m <module> --help` lists its options.

## Running the pipeline from scratch

These stages need GPUs. The job scripts under `scripts/` show the settings each
stage was run with. They take the repository root in `WORKSPACE` and a storage
directory for models and caches in `CULTURELENS_STORAGE_ROOT`, and expect the
Slurm account and partition on the command line:

```bash
WORKSPACE=$PWD CULTURELENS_STORAGE_ROOT=/path/to/storage \
  sbatch --account=<account> --partition=<gpu-partition> scripts/behavioral_evaluation/generate.sbatch
```

| Stage | Module | Job scripts |
|---|---|---|
| Build the benchmark | `benchmark.generate`, `benchmark.split` | none, see `python -m benchmark.generate --help` |
| Generate and judge answers | `behavioral_evaluation.generate`, `.judge`, `.stance` | `scripts/behavioral_evaluation/` |
| Extract hidden states | `mechanistic_analysis.activations.extract` | `scripts/mechanistic_analysis/activations/` |
| Probes and the conflict signal | `mechanistic_analysis.readout`, `.interaction`, `.conflict`, `.country_space` | `scripts/mechanistic_analysis/` |
| Interventions | `mechanistic_analysis.interventions.localize`, `.erasure`, `.heads`, `.generate` | `scripts/mechanistic_analysis/` |
| Sparse autoencoder features | `mechanistic_analysis.sae.extract`, `.analyze`, `.describe` | `scripts/mechanistic_analysis/sae/` |
| Fine-tuning | `finetune.sft` | `scripts/finetuning/` |
| Fine-tuned models, evaluation and internal changes | `mechanistic_analysis.interventions.finetune_shift` | `scripts/mechanistic_analysis/finetune_shift.sbatch` |

Models used: `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-3-12b-it` and
`Qwen/Qwen3.5-27B` are evaluated; `google/gemma-4-31B-it` and `Qwen/Qwen3.6-27B`
are the judges. All generation and judging uses deterministic decoding.

## What is not in the repository

| Not included | Why | How to get it |
|---|---|---|
| Hidden states, about 45 GB | Size | Regenerate with `mechanistic_analysis.activations.extract` |
| Fine-tuned adapter weights | Size | Retrain with `finetune.sft` |
| Cultural Atlas source pages | They belong to the Cultural Atlas | https://culturalatlas.sbs.com.au, placed under `data/raw/atlas/` to rebuild the benchmark |
| World Values Survey microdata | Distributed by the WVS Association | Wave 7 cross-national file, placed at `data/raw/wvs/cross_national_wave_7.csv` |

Only rebuilding the benchmark from scratch needs the two source datasets. Using
the benchmark does not, since the finished records are in
`data/processed/culturelens/`.

## Benchmark format

`data/processed/culturelens/records.json` is a list of records:

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

The first three letters of `norm_id` are the ISO code of the country. In file
and condition names the contradictory norm is called `opposite_norm`, and the
no-evidence condition is called `base`.

## Licence

Code is released under the MIT licence (`LICENSE`). The benchmark, annotations
and outputs are released under CC BY-NC 4.0 for non-commercial research use,
with the notices in `LICENSE-DATA.md`. The contradictory norms are false by
design and must not be used as cultural guidance.

## Citation

```bibtex
@misc{culturelens2026,
  title  = {The Culture Within: Towards Mechanistic Understanding of Cultural Reasoning in LLMs},
  author = {Arif, Samee and Danisman, Furkan and Borah, Angana and Lee, Honglak and Kong, Dehan and Logeswaran, Lajanugen and Lee, Moontae and Jin, Zhijing and Mihalcea, Rada},
  year   = {2026}
}
```
