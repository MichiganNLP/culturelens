# Data licence

The MIT licence in `LICENSE` covers the code in this repository (`src/`,
`scripts/`, `tests/`). It does not cover the data.

The CultureLens benchmark and everything derived from it are released under the
Creative Commons Attribution-NonCommercial 4.0 International licence
(CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). This applies to

- `data/processed/culturelens/` (records, splits, reference answers, requests),
- `human_evaluation/` (annotations),
- `output/` (model answers, judge labels and analysis results).

Conditions and notices:

- **Source.** The true norms are derived from the Cultural Atlas
  (https://culturalatlas.sbs.com.au). The Atlas pages themselves are not
  redistributed here, and use of this data remains subject to the terms of the
  Cultural Atlas.
- **Contradictory norms are false by design.** They are not descriptions of any
  culture. Do not use them as cultural guidance or to train a model to state
  them as facts.
- **Norms are common expectations, not rules.** A documented norm does not
  describe the behaviour of every person in a country.
- **World Values Survey.** `data/processed/references/wvs/` holds country-level
  aggregates computed from WVS Wave 7. The survey microdata is not included and
  is available from the World Values Survey Association under its own terms.
- **Fine-tuned adapters** follow the licence of their base model (Llama 3.1
  Community License, Gemma Terms of Use, Apache 2.0 for Qwen), with the same
  research-use notice.
