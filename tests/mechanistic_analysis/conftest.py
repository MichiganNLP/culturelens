"""A tiny random Llama with a real fast tokenizer, and toy records, for the mechanistic tests."""

from __future__ import annotations

import pytest


def toy_records(per_country: int = 6) -> list[dict]:
    rows = []
    for code, name in (("JPN", "Japan"), ("BRA", "Brazil"), ("KEN", "Kenya")):
        for index in range(per_country):
            rows.append({
                "norm_id": f"{code}__etiquette__{index:03d}__001", "norm": f"norm {index}",
                "values": [],
                "options": {"A": f"They should greet warmly {index}",
                            "B": f"They should leave quietly now {index}"},
                "scenario": {
                    "country_context": f"This interaction takes place in {name}.",
                    "situation": f"A guest arrives at home number {index} in the evening.",
                    "question": "What should they do?",
                    "norm_evidence": f"Guests are greeted warmly {index}.",
                    "opposite_norm_evidence": f"Guests are left alone {index}.",
                    "opposite_norm_text": "x", "value_evidence": None,
                },
            })
    return rows


@pytest.fixture(scope="session")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from mechanistic_analysis.activations.backend import LoadedModel

    words = sorted({word for row in toy_records(12) for text in (
        *row["options"].values(), row["scenario"]["country_context"], row["scenario"]["situation"],
        row["scenario"]["question"], row["scenario"]["norm_evidence"],
        row["scenario"]["opposite_norm_evidence"], "Situation: Question:") for word in text.split()})
    vocab = {token: index for index, token in
             enumerate(["<pad>", "<unk>", "<user>", "<assistant>", "<end>"] + words)}
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>", eos_token="<end>")
    tokenizer.chat_template = (
        "{% for m in messages %}<user> {{ m['content'] }} {% endfor %}<assistant>")
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        pad_token_id=0, eos_token_id=4)).eval()
    return LoadedModel(model, tokenizer, model.model.layers, list(range(4)), 32)
