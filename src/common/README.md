# Common Utilities

`common` contains dependency-light helpers shared across CultureLens packages.
It has no benchmark, model, or experiment-specific logic.

## Contents

- `io.py`: validated JSONL reading, JSON/YAML/text loading, prompt loading, and
  deterministic atomic JSON/JSONL writes.

Install the repository in editable mode and import these helpers as
`common.io`. Domain-specific helpers belong in their owning package rather
than here.

Atomic writers create a temporary file in the destination directory and then
replace the destination. This prevents an interrupted construction or analysis
run from leaving a partially written canonical artifact.

