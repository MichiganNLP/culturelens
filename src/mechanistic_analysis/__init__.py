"""Mechanistic analysis of how CultureLens models use a country cue and a stated norm.

The study explains one behaviour measured on open-ended answers: a model that
answers correctly from the country alone mostly gives that answer up when the
prompt states a norm that is false for the country ("resisting" is keeping it).

    activations/     assistant-start residual states, cached per prompt condition
    outcomes.py      what each model did, as judged by each judge
    readout.py       is the outcome readable from the state before the answer
    interaction.py   how country and claim combine in the state
    interventions/   cutting the answer's attention to parts of the prompt:
                     where the claim is read, and what the model then writes
    report.py        tables and the figure, from the saved outputs
"""

# Layer at which single-layer statistics are reported for each model. Fixed
# from an earlier exploratory run (the first layer past the depth at which the
# decision becomes readable), before this study's outcomes were computed.
REPORT_LAYER = {"llama-3.1-8b": 16, "gemma-3-12b": 26, "qwen-3.5-27b": 36}
# Layer windows for attention knockout, fixed from the same exploratory run on
# a different sample of records: where blocking the claim mattered most, and a
# later window of the same width as the control.
CLAIM_WINDOW = {"llama-3.1-8b": (12, 16), "gemma-3-12b": (18, 24)}
CONTROL_WINDOW = {"llama-3.1-8b": (24, 28), "gemma-3-12b": (36, 42)}
# Layers after which the claim's own positions are erased (interventions/erasure.py):
# the first block, a block before the band in which the claim is read, and one
# after it. From the same exploratory run (bands: Llama 12-16, Gemma 18-24,
# Qwen 28-32). This intervention needs no attention mask, so it also runs on
# models whose layers cannot be masked.
ERASE_LAYERS = {"llama-3.1-8b": (0, 8, 16), "gemma-3-12b": (0, 12, 24), "qwen-3.5-27b": (0, 24, 36)}
# Sample seeds. Anything selected from data (attention heads) is selected on the
# development sample and tested on the confirmation sample.
DEVELOPMENT_SEED, CONFIRMATION_SEED = 42, 2026
JUDGES = ("gemma-4-31b", "qwen-3.6-27b")
RESIST = "country_opposite_norm"
