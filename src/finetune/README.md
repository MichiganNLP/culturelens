# finetune — GRPO and Critique-GRPO

```bash
python -m finetune.grpo --reward constrained \
  --train data/processed/culturelens/splits/train.json \
  --validation data/processed/culturelens/splits/validation.json \
  [--counterfactual]
```

## The grid

| `--reward` | method | policy task | reward |
|---|---|---|---|
| `constrained` | GRPO | A/B with options | exact match on the keyed letter |
| `critique` | Critique-GRPO | open answer, no options | the model's own critique of its answer, given the documented norm and values |
| `critique+constrained` | Critique-GRPO | half each, every step | each prompt's group scored by its own task |

× `--counterfactual` off / on:

| switch | views trained |
|---|---|
| off | `country_only` |
| on | `country_only`, `country_value`, `country_norm`, `country_opposite_norm` |

`country_norm` and `country_opposite_norm` are drawn from the same records in
lockstep, so every false-norm prompt has its true-norm twin in the same step.
Without the twin, "always answer against the stated norm" would earn full reward
on the counterfactual prompts. `--prompts-per-step` must be a multiple of
tasks × views (4 or 8 with the switch on) for the pairing to hold; the pool
refuses anything else.

`base`, `value_only`, `norm_only` and `country_value_norm` are never trained on.

## The critic

No external judge. The critic is the model being trained (`--critic policy`,
the default) or its frozen initialisation (`--critic reference`), and it applies
**the evaluation judge's own rubric**, sent exactly as the judge sends it:
`prompts/behavioral_evaluation/judge.txt` as the system prompt,
`behavioral_evaluation.judge.judge_payload()` as the user message. Training
therefore optimises the same definition of "aligned" the benchmark measures;
the only difference is who applies it — the policy itself during training, an
independent model (Gemma-4-31B) at evaluation. The payload carries the country,
the documented norm and its value labels — never the false norm a
counterfactual prompt asserted, never which view the policy saw.

Verdicts score aligned 1.0, partially aligned 0.5, misaligned 0.0. An unreadable
verdict scores 0 and is logged as `verdict_parse_rate`. The judge's
`improvement` field — one concrete change — is the note a rewrite receives.

Critique-GRPO then rewrites the `--refine-count` weakest open rollouts per group
with their critique in context. Rewrites are scored the same way, join their
prompt's group, and are trained as answers to the plain prompt — the one they
will face at test time. `--refine-count 0` gives GRPO with a self-critic reward
and no rewrites.

**The live critic can go easy on itself.** Reward rising while constrained
accuracy is flat is the sign. `--critic reference` removes that possibility at
no memory cost (the reference adapter is loaded for the KL term anyway), and
`critique+constrained` anchors the run with a reward no model graded.

## What gets logged

Every step (`metrics.jsonl`, `"event": "train"`): reward overall, by task and by
view; constrained accuracy and `chose_a` (a letter collapse is invisible in
accuracy when the order is shuffled); `verdict_parse_rate`; how many groups
carried a gradient; refinement reward; KL and clip fraction.

Every `--eval-every` steps (`"event": "validation"`), on a fixed validation
subset:

- **constrained accuracy on all four monitor views, whatever the reward** — a
  judge-free number for every run, using the benchmark's own fixed A/B order;
- **`country_norm_change`** — accuracy on true-norm prompts relative to step 0.
  A counterfactual run that starts rejecting stated norms shows up here first;
- under a critique reward, the critic's score on open answers, judged by the
  frozen reference by default (`--validation-critic`) so the yardstick does not
  drift with the policy.

`best/` is the adapter with the highest mean constrained validation accuracy
over the trained views (step 0 included); `step-N/` every `--save-every`;
`final/` at the end.

## Layout

| module | holds | needs a GPU |
|---|---|---|
| `__init__.py` | recipes, reward modes, method names | no |
| `data.py` | loading, the lockstep prompt pool, validation subsets | no |
| `prompts.py` | policy, critic and refinement prompts; verdict parsing | no |
| `rewards.py` | both rewards, ragged group advantages, weakest-k | no |
| `tokenization.py` | chat encoding shared by sampling and scoring | no |
| `losses.py` | per-token log-probs and the clipped GRPO loss | torch |
| `grpo.py` | the loop | torch, transformers, peft |

`tests/finetune/` covers every module that needs no GPU.

## SFT on gold answers

```bash
python -m finetune.sft --train data/processed/culturelens/splits/train.json \
  --validation data/processed/culturelens/splits/validation.json \
  --gold-dir data/processed/culturelens/gold [--counterfactual]
```

The same two recipes as GRPO: `country_only` alone, or with `--counterfactual`
also `country_value`, `country_norm` and `country_opposite_norm`. Targets are
the teacher's gold answers (`benchmark.generate gold` and `gold_cf`): the
country-only answer for every truthful view, and the correcting answer for the
false-norm view. Loss is on the answer tokens only. `best/` is the adapter with
the lowest held-out answer loss, on validation gold when it exists and
otherwise on a held-out 3% of the training records. Job scripts:
`scripts/finetuning/sft.sbatch` and `sft_cf.sbatch`.
