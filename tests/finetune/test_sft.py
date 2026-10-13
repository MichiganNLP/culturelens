"""SFT: which examples are built, how the run is ordered, and one tiny real run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from finetune import COUNTERFACTUAL_VIEWS, FACTUAL_VIEWS
from finetune.sft import (
    build_examples, constrained_target, epoch_order, hold_out, learning_rate_scale, load_gold,
    method_name, parse_args, resolve,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/processed/culturelens"


def record(index: int, *, value: bool = False, iso3: str = "JPN") -> dict:
    return {
        "norm_id": f"{iso3}__etiquette__001__{index:03d}", "norm": f"norm {index}",
        "values": ["tradition"] if value else [],
        "scenario": {
            "country_context": "This interaction takes place in Japan.",
            "situation": f"A guest arrives at house {index}.", "question": "What should the guest do?",
            "norm_evidence": f"Guests remove their shoes {index}.",
            "opposite_norm_evidence": f"Guests keep their shoes on {index}.",
            "opposite_norm_text": "x",
            "value_evidence": "Keeping customs is important." if value else None,
        },
        "options": {"A": "Remove the shoes.", "B": "Keep the shoes on."},
    }


def test_names_and_views():
    assert method_name(False) == "sft-country" and method_name(True) == "sft-cf"
    args = parse_args(["--train", "t.json", "--gold-dir", "g"])
    assert args.views == FACTUAL_VIEWS and args.method == "sft-country"
    args = parse_args(["--train", "t.json", "--gold-dir", "g", "--counterfactual"])
    assert args.views == COUNTERFACTUAL_VIEWS and str(args.output_dir).endswith("sft-cf")


def test_truthful_views_share_one_answer_and_the_false_norm_view_has_its_own():
    rows = [record(1, value=True), record(2)]
    gold = {rows[0]["norm_id"]: "gold one", rows[1]["norm_id"]: "gold two"}
    gold_cf = {rows[0]["norm_id"]: "correction one"}        # none for the second record
    examples, missing = build_examples(rows, COUNTERFACTUAL_VIEWS, gold, gold_cf)
    by = {(e["norm_id"], e["view"]): e for e in examples}
    first = rows[0]["norm_id"]
    for view in ("country_only", "country_value", "country_norm"):
        assert by[(first, view)]["response"] == "gold one"
    assert by[(first, "country_opposite_norm")]["response"] == "correction one"
    # The prompt is the evaluation's own open-ended prompt for that view.
    assert by[(first, "country_norm")]["prompt"].endswith("Guests remove their shoes 1.")
    assert by[(first, "country_opposite_norm")]["prompt"].endswith("Guests keep their shoes on 1.")
    assert "important" in by[(first, "country_value")]["prompt"]
    assert by[(first, "country_only")]["prompt"].endswith("takes place in Japan.")
    # No value evidence -> no value view; no false-norm gold -> counted, not invented.
    second = rows[1]["norm_id"]
    assert (second, "country_value") not in by and (second, "country_opposite_norm") not in by
    assert missing == {"country_only": 0, "country_value": 0, "country_norm": 0,
                       "country_opposite_norm": 1}
    country, _ = build_examples(rows, FACTUAL_VIEWS, gold, gold_cf)
    assert [e["view"] for e in country] == ["country_only", "country_only"]


def test_held_out_records_never_reach_training():
    rows = [record(i) for i in range(100)]
    train, dev = hold_out(rows, 0.1, seed=1)
    assert len(dev) == 10 and len(train) == 90
    assert not {r["norm_id"] for r in train} & {r["norm_id"] for r in dev}
    assert [r["norm_id"] for r in hold_out(rows, 0.1, seed=1)[1]] == [r["norm_id"] for r in dev]


def test_epoch_order_and_schedule():
    assert sorted(epoch_order(50, 0, 42)) == list(range(50))
    assert epoch_order(50, 0, 42) == epoch_order(50, 0, 42) != epoch_order(50, 1, 42)
    assert learning_rate_scale(1, 100, 10) == pytest.approx(0.1)
    assert learning_rate_scale(10, 100, 10) == pytest.approx(1.0)
    assert learning_rate_scale(55, 100, 10) == pytest.approx(0.5)
    assert learning_rate_scale(100, 100, 10) == pytest.approx(0.0, abs=1e-9)


def test_gold_file_reading(tmp_path):
    path = tmp_path / "train.country_only.jsonl"
    path.write_text(json.dumps({"norm_id": "a", "response": " answer "}) + "\n\n"
                    + json.dumps({"norm_id": "b", "response": ""}) + "\n")
    assert load_gold(path) == {"a": "answer"}
    assert load_gold(tmp_path / "absent.jsonl") == {}


@pytest.mark.skipif(not (DATA / "gold/train.country_only.jsonl").is_file(), reason="no gold files")
def test_real_gold_covers_the_train_split():
    from finetune.data import load_records
    rows = load_records(DATA / "splits/train.json")
    gold = load_gold(DATA / "gold/train.country_only.jsonl")
    gold_cf = load_gold(DATA / "gold/train.country_opposite_norm.jsonl")
    examples, missing = build_examples(rows, COUNTERFACTUAL_VIEWS, gold, gold_cf)
    assert missing["country_only"] == 0 and missing["country_opposite_norm"] <= 10
    assert len(examples) > 3 * len(rows)


@pytest.mark.parametrize("both_tasks", [False, True])
def test_a_tiny_run_trains_selects_and_resumes(tmp_path, both_tasks):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from finetune import sft

    rows = [record(i, value=i % 2 == 0) for i in range(24)]
    splits, gold_dir = tmp_path / "splits", tmp_path / "gold"
    splits.mkdir(); gold_dir.mkdir()
    (splits / "train.json").write_text(json.dumps(rows))
    for name, text in (("country_only", "The guest should remove the shoes at the door"),
                       ("country_opposite_norm", "Guests do not keep shoes on so remove the shoes")):
        (gold_dir / f"train.{name}.jsonl").write_text("\n".join(
            json.dumps({"norm_id": r["norm_id"], "response": f"{text} {i}"})
            for i, r in enumerate(rows)))

    # A word-level tokenizer with a minimal chat template, and a tiny random Llama.
    words = sorted({w for r in rows for t in (r["scenario"]["situation"], r["scenario"]["question"],
                                              r["scenario"]["country_context"], r["scenario"]["norm_evidence"],
                                              r["scenario"]["opposite_norm_evidence"], "Situation: Question:",
                                              "Keeping customs is important.",
                                              "The guest should remove the shoes at the door",
                                              "Guests do not keep shoes on so remove the shoes",
                                              " ".join(str(i) for i in range(24)))
                    for w in t.split()})
    vocab = {token: index for index, token in enumerate(["<pad>", "<unk>", "<user>", "<assistant>", "<end>"] + words)}
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>", eos_token="<end>")
    tokenizer.chat_template = (
        "{% for m in messages %}{% if m['role'] == 'user' %}<user> {{ m['content'] }} <assistant> "
        "{% else %}{{ m['content'] }} <end>{% endif %}{% endfor %}")
    model_dir = tmp_path / "model"
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        pad_token_id=0, eos_token_id=4)).save_pretrained(model_dir)

    out = tmp_path / "run"
    common = ["--model", str(model_dir), "--train", str(splits / "train.json"),
              "--gold-dir", str(gold_dir), "--counterfactual", "--output-dir", str(out),
              "--device", "cpu", "--dtype", "float32", "--batch-size", "8", "--micro-batch-size", "4",
              "--learning-rate", "5e-3", "--eval-every", "5", "--state-every", "5",
              "--dev-fraction", "0.2", "--lora-dropout", "0.0"] + (["--constrained"] if both_tasks else [])
    assert sft.main(common + ["--epochs", "2"]) == 0
    events = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    plan = events[0]
    assert plan["views"] == list(COUNTERFACTUAL_VIEWS) and "held out" in plan["development"]
    assert plan["examples_by_view"]["country_value"] < plan["examples_by_view"]["country_only"]
    assert plan["constrained"] is both_tasks
    if both_tasks:
        assert plan["examples_by_view"]["country_only:constrained"] == plan["examples_by_view"]["country_only"]
        assert json.loads((out / "manifest.json").read_text())["method"] == "sft-all-cf"
    validation = [e for e in events if e["event"] == "validation"]
    assert validation[-1]["selection_score"] < validation[0]["selection_score"]   # it learned
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["best"]["step"] > 0 and math.isfinite(manifest["best"]["score"])
    assert (out / "best/adapter_config.json").is_file() and (out / "final/adapter_config.json").is_file()
    # A finished run refuses to be overwritten, and resumes without redoing steps.
    with pytest.raises(FileExistsError):
        sft.main([a for a in common if a != "--resume"] + ["--epochs", "2"])
    before = len((out / "metrics.jsonl").read_text().splitlines())
    assert sft.main(common + ["--epochs", "2", "--resume"]) == 0
    after = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()][before:]
    assert after[0]["event"] == "resume" and not [e for e in after if e["event"] == "train"]


def test_both_tasks_teach_one_answer_in_two_formats():
    assert method_name(False, True) == "sft-all" and method_name(True, True) == "sft-all-cf"
    args = parse_args(["--train", "t.json", "--gold-dir", "g", "--constrained", "--counterfactual"])
    assert args.method == "sft-all-cf" and args.views == COUNTERFACTUAL_VIEWS
    rows = [record(1), record(2)]
    gold = {row["norm_id"]: f"gold {i}" for i, row in enumerate(rows)}
    examples, _ = build_examples(rows, FACTUAL_VIEWS, gold, {}, constrained=True)
    assert [e["kind"] for e in examples] == ["open", "constrained"] * 2
    open_example, choice = examples[0], examples[1]
    assert resolve(open_example, 42, 0) is open_example and "system" not in open_example
    seen = set()
    for epoch in range(40):
        drawn = resolve(choice, 42, epoch)
        assert drawn == resolve(choice, 42, epoch)                     # same when resumed
        assert drawn["system"].startswith("You will be given a multiple-choice")
        assert drawn["prompt"].endswith("Choose A or B.") and "takes place in Japan." in drawn["prompt"]
        first_is_documented = "A: Remove the shoes." in drawn["prompt"]
        letter = "A" if first_is_documented else "B"
        assert drawn["response"] == f"<reasoning>gold 0</reasoning>\n<answer>{letter}</answer>"
        seen.add(letter)
    assert seen == {"A", "B"}                                          # the letter is not fixed
    fixed = resolve(choice, 42, None)
    assert fixed == resolve(choice, 7, None)                           # evaluation order ignores the seed
    assert constrained_target(" x ", True) == "<reasoning>x</reasoning>\n<answer>B</answer>"


def test_heads_are_parsed_and_masked():
    from finetune.sft import head_mask, parse_heads

    assert parse_heads(["13.4", "13.22", "12.4"]) == {12: [4], 13: [4, 22]}
    with pytest.raises(ValueError):
        parse_heads(["13.4", "13.4"])
    with pytest.raises(ValueError):
        parse_heads(["13"])
    heads = {1: [0, 3]}
    query = head_mask("base_model.model.model.layers.1.self_attn.q_proj.lora_B.default.weight", (32, 4), heads, 4)
    assert query == [True] * 8 + [False] * 16 + [True] * 8          # rows of heads 0 and 3
    out = head_mask("base_model.model.model.layers.1.self_attn.o_proj.lora_A.default.weight", (4, 32), heads, 4)
    assert out == query                                              # the same heads, as columns
    # The other half of each adapter, other modules and other layers are not sliced by head.
    assert head_mask("base_model.model.model.layers.1.self_attn.q_proj.lora_A.default.weight", (4, 32), heads, 4) is None
    assert head_mask("base_model.model.model.layers.1.mlp.up_proj.lora_B.default.weight", (64, 4), heads, 4) is None
    assert not any(head_mask("base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight", (32, 4), heads, 4))
    assert parse_args(["--train", "t.json", "--gold-dir", "g"]).lora_layers is None


@pytest.mark.parametrize("arm", ["layers", "heads", "conflict", "framings", "distill", "gate", "mirror", "mirror-v2", "value", "memory"])
def test_tiny_runs_of_the_mechanistically_placed_arms(tmp_path, arm):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from finetune import sft
    from peft.utils import load_peft_weights

    rows = [record(i, value=i % 2 == 0) for i in range(24)]
    splits, gold_dir = tmp_path / "splits", tmp_path / "gold"
    splits.mkdir(); gold_dir.mkdir()
    (splits / "train.json").write_text(json.dumps(rows))
    for name, text in (("country_only", "The guest should remove the shoes at the door"),
                       ("country_opposite_norm", "Guests do not keep shoes on so remove the shoes")):
        (gold_dir / f"train.{name}.jsonl").write_text("\n".join(
            json.dumps({"norm_id": r["norm_id"], "response": f"{text} {i}"})
            for i, r in enumerate(rows)))
    words = sorted({w for r in rows for t in (r["scenario"]["situation"], r["scenario"]["question"],
                                              r["scenario"]["country_context"], r["scenario"]["norm_evidence"],
                                              r["scenario"]["opposite_norm_evidence"], "Situation: Question:",
                                              "Keeping customs is important.",
                                              "The guest should remove the shoes at the door",
                                              "Guests do not keep shoes on so remove the shoes",
                                              "Someone told me this: According to a guide to local customs: "
                                              "I am not sure this is right. <mem>",
                                              " ".join(str(i) for i in range(24)))
                    for w in t.split()})
    vocab = {token: index for index, token in enumerate(["<pad>", "<unk>", "<user>", "<assistant>", "<end>"] + words)}
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>", eos_token="<end>")
    tokenizer.chat_template = (
        "{% for m in messages %}{% if m['role'] == 'user' %}<user> {{ m['content'] }} <assistant> "
        "{% else %}{{ m['content'] }} <end>{% endif %}{% endfor %}")
    model_dir = tmp_path / "model"
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        pad_token_id=0, eos_token_id=4)).save_pretrained(model_dir)

    extra = {"layers": ["--lora-layers", "1"], "heads": ["--lora-heads", "1.0", "1.3", "2.2"],
             "conflict": ["--conflict-weight", "0.5", "--conflict-layer", "1"],
             "framings": ["--framings"], "mirror": ["--mirror", "--task-tokens", "8"],
             "mirror-v2": ["--mirror", "--propose", "--task-tokens", "8"],
             "value": ["--value-first", "--task-tokens", "8"],
             "memory": ["--country-memory", "2", "--memory-token", " <mem>"],
             "distill": ["--distill-weight", "0.5", "--distill-activation-weight", "0.5",
                         "--distill-layer", "1", "--aux-records", "2"],
             "gate": ["--gate-weight", "1.0", "--gate-heads", "1.0", "1.3", "2.2",
                      "--aux-records", "2"]}[arm]
    out = tmp_path / "run"
    common = ["--model", str(model_dir), "--train", str(splits / "train.json"),
              "--gold-dir", str(gold_dir), "--counterfactual", "--output-dir", str(out),
              "--method", f"sft-cf-{arm}", "--device", "cpu", "--dtype", "float32", "--batch-size", "8",
              "--micro-batch-size", "4", "--learning-rate", "5e-3", "--eval-every", "5",
              "--state-every", "5", "--dev-fraction", "0.2", "--lora-dropout", "0.0"] + extra
    assert sft.main(common + ["--epochs", "2"]) == 0
    events = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    validation = [e for e in events if e["event"] == "validation"]
    assert validation[-1]["selection_score"] < validation[0]["selection_score"]
    weights = load_peft_weights(str(out / "final"), device="cpu")
    layers = {int(name.split("layers.")[1].split(".")[0]) for name in weights}
    if arm == "layers":
        assert layers == {1}
    elif arm == "heads":
        assert layers == {1, 2} and all("q_proj" in name or "o_proj" in name for name in weights)
        for name, value in weights.items():
            chosen = {1: [0, 3], 2: [2]}[int(name.split("layers.")[1].split(".")[0])]
            inside = torch.zeros(32, dtype=torch.bool)
            for head in chosen:
                inside[head * 8:(head + 1) * 8] = True
            if "q_proj.lora_B" in name:
                assert value[~inside].abs().sum() == 0 and value[inside].abs().sum() > 0
            if "o_proj.lora_A" in name:
                assert value[:, ~inside].abs().sum() == 0 and value[:, inside].abs().sum() > 0
    elif arm == "memory":
        import numpy as np
        from finetune import country_memory

        saved = torch.load(out / "best" / country_memory.MEMORY_FILE, weights_only=False)
        assert saved["countries"] == ["JPN"] and tuple(saved["table"].shape) == (1, 2, 32)
        start = country_memory.initial_vectors(torch, transformers.AutoModelForCausalLM.from_pretrained(model_dir),
                                               tokenizer, 2)
        assert not torch.allclose(saved["table"][0], start)                 # the vectors were trained
        assert layers == {0, 1, 2} and [e for e in events if e["event"] == "memory"][0]["slots"] == 2
        # Generation: a country that was trained on, and one that borrows from its neighbour.
        (splits / "other.json").write_text(json.dumps([record(i, iso3="KOR") for i in range(3)] + rows[:3]))
        np.savez(tmp_path / "geometry.npz", countries=np.array(["JPN", "KOR"]), layers=np.array([16]),
                 rdm_shift=np.array([[[0.0, 0.4], [0.4, 0.0]]]))
        answers = tmp_path / "answers.jsonl"
        call = ["--model", str(model_dir), "--lora", str(out / "best"), "--input", str(splits / "other.json"),
                "--output", str(answers), "--geometry", str(tmp_path / "geometry.npz"), "--device", "cpu",
                "--dtype", "float32", "--max-tokens", "4", "--batch-size", "4",
                "--conditions", "country_only", "country_norm"]
        try:
            assert country_memory.main(call) == 0
        except RuntimeError as error:                 # a tiny random model may write nothing for a prompt
            assert "incomplete" in str(error)
        assert json.loads(answers.with_suffix(".neighbours.json").read_text()) == {"KOR": {"JPN": 1.0}}
        written = [json.loads(line) for line in answers.read_text().splitlines() if line.strip()]
        assert {row["memory"] for row in written} <= {"trained", "neighbours"}
        assert all("<mem>" not in row["user_message"] for row in written)
    elif arm == "value":
        by_view = events[0]["examples_by_view"]
        assert layers == {0, 1, 2} and by_view["value"] == by_view["country_value"]
        assert by_view["country_opposite_norm"] == by_view["country_only"]     # sft-cf is intact
        assert "value" in validation[-1]["loss"]
    elif arm in ("framings", "distill", "gate"):
        assert layers == {0, 1, 2}
        train = [e for e in events if e["event"] == "train"]
        if arm == "distill":
            assert math.isfinite(train[-1]["distill_kl"]) and 0 <= train[-1]["distill_state"] <= 2
        if arm == "gate":
            assert 0 <= train[-1]["attention_false"] <= 1 and 0 <= train[-1]["attention_true"] <= 1
            assert train[-1]["attention_false"] < train[0]["attention_false"]   # the gate closes
    elif arm in ("mirror", "mirror-v2"):
        by_view = events[0]["examples_by_view"]
        assert layers == {0, 1, 2} and "country_opposite_norm" not in by_view
        assert by_view["mirror"] == by_view["compare"] == 2 * by_view["country_norm"]
        assert {"mirror", "compare"} <= set(validation[-1]["loss"])
        assert ("propose" in by_view) is (arm == "mirror-v2")
        if arm == "mirror-v2":
            assert by_view["propose"] == by_view["country_only"] + by_view["country_value"]
    else:
        assert layers == {0, 1, 2} and (out / "best/conflict_probe.pt").is_file()
        assert validation[-1]["conflict"]["records"] > 0
        train = [e for e in events if e["event"] == "train"]
        assert "conflict_loss" in train[-1] and math.isfinite(train[-1]["conflict_loss"])
    # The run resumes without redoing steps.
    before = len((out / "metrics.jsonl").read_text().splitlines())
    assert sft.main(common + ["--epochs", "2", "--resume"]) == 0
    after = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()][before:]
    assert after[0]["event"] == "resume" and not [e for e in after if e["event"] == "train"]


def test_framings_match_the_evaluation_and_never_reveal_the_norm():
    from behavioral_evaluation.prompts import build_user_message
    from finetune.prompts import FRAMINGS, constrained_message, open_message

    row = record(3, value=True)
    # A framed contradictory norm is exactly the evaluation's diagnostic prompt.
    for framing, condition in (("hearsay", "country_opposite_norm_hearsay"), ("guide", "country_opposite_norm_guide"),
                               ("doubt", "country_opposite_norm_doubt"), ("first", "opposite_norm_country")):
        assert open_message(row, "country_opposite_norm", framing) == build_user_message(row, condition, False)[0]
    assert open_message(row, "country_opposite_norm", "plain") == build_user_message(row, "country_opposite_norm", False)[0]
    # The true norm gets the same wording around it, and other views are never framed.
    true = open_message(row, "country_norm", "guide")
    assert "According to a guide to local customs: Guests remove their shoes 3." in true
    assert open_message(row, "country_only", "guide") == open_message(row, "country_only")
    assert constrained_message(row, "country_norm", shuffled=False, framing="doubt").count("I am not sure this is right.") == 1
    # Drawn per epoch, fixed for a given epoch, plain on development examples.
    example = {"norm_id": row["norm_id"], "view": "country_opposite_norm", "kind": "open", "record": row,
               "prompt": open_message(row, "country_opposite_norm"), "response": "x"}
    assert resolve(example, 1, None, True)["prompt"] == example["prompt"]
    assert resolve(example, 1, 4, True) == resolve(example, 1, 4, True)
    assert len({resolve(example, 1, epoch, True)["prompt"] for epoch in range(40)}) == len(FRAMINGS)
    assert resolve(example, 1, 4)["prompt"] == example["prompt"]


def test_mirror_verification_examples_and_decision():
    from finetune.mirror import (
        COMPARE_SYSTEM, answer_message, clean_mirror, compare_message, decide, summarize,
    )
    from finetune.sft import MIRROR_VIEWS

    rows = [record(i, value=i % 2 == 0) for i in range(6)]
    gold = {r["norm_id"]: f"answer {i}" for i, r in enumerate(rows)}
    examples, _ = build_examples(rows, MIRROR_VIEWS, gold, {}, mirror=True)
    assert not [e for e in examples if e["view"] == "country_opposite_norm"]
    mirrors = [e for e in examples if e["kind"] == "mirror" and e["norm_id"] == rows[0]["norm_id"]]
    scenario = rows[0]["scenario"]
    assert {(e["prompt"], e["response"]) for e in mirrors} == {
        (f"Statement: {scenario['norm_evidence']}", scenario["opposite_norm_evidence"]),
        (f"Statement: {scenario['opposite_norm_evidence']}", scenario["norm_evidence"])}
    assert "Japan" not in mirrors[0]["prompt"]                    # wording only, no country
    # The two copies of a record show the two orders, in training and in development.
    for epoch in (None, 0, 1, 2):
        pair = [resolve(e, 42, epoch) for e in examples
                if e["kind"] == "compare" and e["norm_id"] == rows[0]["norm_id"]]
        assert {p["response"] for p in pair} == {"1", "2"} and pair[0]["system"] == COMPARE_SYSTEM
        for p in pair:
            line = f"Statement {p['response']}: {scenario['norm_evidence']}"
            assert line in p["prompt"] and scenario["country_context"] in p["prompt"]
    # The control drops the country line and nothing else.
    with_country = compare_message(rows[0], "a", "b")
    assert compare_message(rows[0], "a", "b", country=False) == with_country.replace(
        scenario["country_context"] + "\n\n", "")
    # The answer prompt is the evaluation's country_norm prompt with the chosen norm.
    from behavioral_evaluation.prompts import build_user_message
    assert answer_message(rows[0], scenario["norm_evidence"]) == build_user_message(
        rows[0], "country_norm", False)[0]
    assert answer_message(rows[0], scenario["opposite_norm_evidence"]) == build_user_message(
        rows[0], "country_opposite_norm", False)[0]
    # Both orders must pick the opposite for the stated norm to be replaced.
    assert decide("2", "1") == ("mirror", True) and decide("1", "2") == ("stated", True)
    assert decide("1", "1") == ("stated", False) and decide("2", "2") == ("stated", False)
    assert clean_mirror("Statement: Guests keep shoes on.\nextra") == "Guests keep shoes on."
    steps = summarize([
        {"condition": "country_norm", "chosen": "stated", "orders_agree": True,
         "mirror_matches_dataset": True, "control_chosen": "mirror", "control_orders_agree": True},
        {"condition": "country_opposite_norm", "chosen": "stated", "orders_agree": False,
         "mirror_matches_dataset": False, "control_chosen": "mirror", "control_orders_agree": True}])
    assert steps["country_norm"]["chose_documented"] == 1.0
    assert steps["country_opposite_norm"]["chose_documented"] == 0.0
    assert steps["country_opposite_norm"]["control_chose_documented"] == 1.0


def test_mirror_v2_proposes_a_norm_where_none_is_stated():
    from behavioral_evaluation.prompts import build_user_message
    from finetune.mirror import PROPOSE_SYSTEM, propose_message, summarize
    from finetune.sft import MIRROR_VIEWS

    rows = [record(i, value=i % 2 == 0) for i in range(6)]
    gold = {r["norm_id"]: f"answer {i}" for i, r in enumerate(rows)}
    plain, _ = build_examples(rows, MIRROR_VIEWS, gold, {}, mirror=True)
    examples, _ = build_examples(rows, MIRROR_VIEWS, gold, {}, mirror=True, propose=True)
    proposals = [e for e in examples if e["kind"] == "propose"]
    assert [e for e in examples if e["kind"] != "propose"] == plain       # v1 is unchanged
    assert len(proposals) == 6 + 3                                        # one more where there is a value
    for example in proposals:
        scenario = example["record"]["scenario"]
        assert example["response"] == scenario["norm_evidence"] and example["system"] == PROPOSE_SYSTEM
        assert scenario["norm_evidence"] not in example["prompt"]
        assert scenario["opposite_norm_evidence"] not in example["prompt"]
        assert resolve(example, 42, 0) is example
    assert propose_message(rows[0]).startswith(build_user_message(rows[0], "country_only", False)[0])
    assert rows[0]["scenario"]["value_evidence"] in propose_message(rows[0], "country_value")
    with pytest.raises(ValueError):
        propose_message(rows[0], "country_norm")
    steps = summarize([{"condition": "country_only", "proposed_norm": "x", "chosen": "mirror",
                        "orders_agree": True, "proposal_is_documented": False,
                        "proposal_is_contradictory": True, "control_chosen": "stated"}])
    assert steps["country_only"]["replaced_by_opposite"] == 1.0
    assert steps["country_only"]["proposal_is_contradictory_norm"] == 1.0


def test_value_first_trains_the_value_and_answers_through_it():
    from behavioral_evaluation.prompts import build_user_message
    from finetune.value_first import VALUE_SYSTEM, answer_message, summarize, value_message

    rows = [record(i, value=i % 2 == 0) for i in range(6)]
    gold = {r["norm_id"]: f"answer {i}" for i, r in enumerate(rows)}
    plain, _ = build_examples(rows, COUNTERFACTUAL_VIEWS, gold, gold)
    examples, _ = build_examples(rows, COUNTERFACTUAL_VIEWS, gold, gold, value_first=True)
    values = [e for e in examples if e["kind"] == "value"]
    assert [e for e in examples if e["kind"] != "value"] == plain          # sft-cf is unchanged
    assert len(values) == 3                                                # only records with a value
    for example in values:
        scenario = example["record"]["scenario"]
        assert example["response"] == scenario["value_evidence"] and example["system"] == VALUE_SYSTEM
        assert scenario["value_evidence"] not in example["prompt"]
        assert scenario["norm_evidence"] not in example["prompt"]
        assert example["prompt"].startswith(build_user_message(example["record"], "country_only", False)[0])
        assert resolve(example, 42, 0) is example
    # The answer prompt is the evaluation's country_value prompt with the value written in.
    assert answer_message(rows[0], rows[0]["scenario"]["value_evidence"]) == build_user_message(
        rows[0], "country_value", False)[0]
    assert value_message(rows[1]).startswith("Situation:")                 # works without a value too
    steps = summarize([{"predicted_value": "x", "dataset_value": "x", "value_matches_dataset": True},
                       {"predicted_value": "", "dataset_value": "", "value_matches_dataset": False}])
    assert steps == {"prompts": 2, "without_a_prediction": 1, "with_annotated_value": 1,
                     "prediction_is_annotated_value": 1.0}


def test_country_memory_marks_the_norm_position_and_blends_neighbours():
    from behavioral_evaluation.prompts import build_user_message
    from finetune.country_memory import neighbour_weights, with_memory

    row = record(0)
    line = row["scenario"]["country_context"]
    for view in ("country_only", "country_norm", "country_opposite_norm"):
        prompt = build_user_message(row, view, False)[0]
        marked = with_memory(prompt, row, 3, "<m>")
        assert marked == prompt.replace(line, line + "<m><m><m>") and marked.count("<m>") == 3
    # The memory sits where a stated norm goes: directly after the country, before the norm.
    marked = with_memory(build_user_message(row, "country_norm", False)[0], row, 2, "<m>")
    assert marked.index(line) < marked.index("<m>") < marked.index(row["scenario"]["norm_evidence"])
    bare = build_user_message(row, "base", False)[0]
    assert with_memory(bare, row, 3, "<m>") == bare and with_memory(marked, row, 0, "<m>") == marked
    distance = [[0.0, 0.2, 0.9, 0.5], [0.2, 0.0, 0.8, 0.6], [0.9, 0.8, 0.0, 0.3], [0.5, 0.6, 0.3, 0.0]]
    weights = neighbour_weights(["A", "B", "C", "D"], distance, ["A", "B", "C"], "D", nearest=2)
    assert set(weights) == {"C", "A"} and weights["C"] > weights["A"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert neighbour_weights(["A", "B"], [[0, 1], [1, 0]], ["A"], "Z") == {}
