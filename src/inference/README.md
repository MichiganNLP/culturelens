# Inference Backend

`inference` provides the model-agnostic local generation backend shared by
benchmark construction and later experiments.

## Backend

`vllm.py` wraps vLLM's in-process Python API and supports:

- batched chat generation;
- finite-choice constrained generation;
- JSON-schema-constrained generation with Pydantic validation;
- deterministic greedy decoding;
- optional model-specific chat-template arguments, including disabling Qwen
  thinking mode;
- optional LoRA requests.

The module intentionally contains no cultural categories, countries, prompts,
or dataset paths. vLLM is imported lazily, so CPU-only data inspection and CLI
validation do not require the GPU dependency.

Install the generation dependencies on a CUDA machine with:

```bash
python -m pip install -e '.[generation]'
```

Benchmark commands configure this backend through their command-line flags;
users normally do not invoke `inference.vllm` directly.

