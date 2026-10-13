"""Attention knockout: the mask edit, the margin, and generation under it."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import toy_records  # noqa: E402
from mechanistic_analysis import CLAIM_WINDOW, CONTROL_WINDOW, ERASE_LAYERS  # noqa: E402
from mechanistic_analysis.interventions import erasure, generate, heads, localize  # noqa: E402
from mechanistic_analysis.interventions.erasure import erased  # noqa: E402
from mechanistic_analysis.interventions.attention import (  # noqa: E402
    blocked, maskable_layers, prompt_pieces, token_spans, with_control,
)
from mechanistic_analysis.interventions.common import encode_prompt, sample_records  # noqa: E402
from mechanistic_analysis.interventions.scoring import BatchScorer, margin  # noqa: E402

LOW = torch.finfo(torch.float32).min


def test_token_spans_find_each_part_of_the_prompt(tiny):
    record = toy_records()[0]
    ids, after, spans = encode_prompt(tiny, record, "country_opposite_norm", None)
    words = [tiny.tokenizer.decode([i]) for i in ids]
    assert words[after] == "<assistant>" and after == len(ids) - 1
    assert words[spans["country"][0]:spans["country"][1]] == "This interaction takes place in Japan.".split()
    assert words[spans["claim"][0]:spans["claim"][1]] == "Guests are left alone 0.".split()
    length = spans["claim"][1] - spans["claim"][0]
    assert spans["control"] == (spans["situation"][0], spans["situation"][0] + length)
    assert "claim" not in encode_prompt(tiny, record, "country_only", None)[2]
    assert "country" not in encode_prompt(tiny, record, "opposite_norm_only", None)[2]
    assert prompt_pieces(record, "country_norm")["claim"] == "Guests are greeted warmly 0."
    assert with_control({"situation": (0, 3)}) == {"situation": (0, 3)}


def test_margin_matches_unbatched_scoring(tiny):
    generator = torch.Generator().manual_seed(1)
    size = tiny.model.config.vocab_size
    prompts = [torch.randint(5, size, (n,), generator=generator).tolist() for n in (9, 14, 6)]
    answers = [[torch.randint(5, size, (k,), generator=generator).tolist() for k in (3, 5)]
               for _ in prompts]
    scores = BatchScorer(tiny.model, 0, prompts, answers).score()
    for record, (prompt, options) in enumerate(zip(prompts, answers)):
        for option, answer in enumerate(options):
            with torch.no_grad():
                logits = tiny.model(torch.tensor([prompt + answer])).logits[0]
            expected = sum(float(torch.log_softmax(logits[len(prompt) - 1 + j], -1)[token])
                           for j, token in enumerate(answer))
            assert float(scores[record, option]) == pytest.approx(expected, abs=1e-4)
    assert margin(scores).shape == (3,)


def test_blocking_equals_a_hand_built_mask_and_leaves_no_hooks(tiny):
    generator = torch.Generator().manual_seed(5)
    size = tiny.model.config.vocab_size
    prompts = [torch.randint(5, size, (n,), generator=generator).tolist() for n in (12, 9)]
    answers = [[torch.randint(5, size, (k,), generator=generator).tolist() for k in (3, 4)]
               for _ in prompts]
    scorer = BatchScorer(tiny.model, 0, prompts, answers)
    plain = scorer.score()
    query = torch.tensor([8, 8, 6, 6])
    keys = torch.zeros(4, scorer.width, dtype=torch.bool)
    keys[:2, 2:5] = True
    keys[2:, 1:3] = True
    with blocked(tiny.decoder_layers, list(range(4)), query, keys):
        cut = scorer.score()
    assert not torch.allclose(cut, plain, atol=1e-3)
    assert torch.allclose(scorer.score(), plain, atol=1e-6)
    with blocked(tiny.decoder_layers, [], query, keys):
        assert torch.allclose(scorer.score(), plain, atol=1e-6)
    prompt, answer = prompts[0], answers[0][0]
    tokens = torch.tensor([prompt + answer])
    n = tokens.shape[1]
    mask = torch.zeros(1, 1, n, n).masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), LOW)
    mask[0, 0, 8:, 2:5] = LOW
    with torch.no_grad():
        logits = tiny.model(tokens, attention_mask=mask).logits[0]
    expected = sum(float(torch.log_softmax(logits[len(prompt) - 1 + j], -1)[token])
                   for j, token in enumerate(answer))
    assert float(cut[0, 0]) == pytest.approx(expected, abs=1e-4)
    assert maskable_layers(tiny.decoder_layers) == [0, 1, 2, 3]


def test_generation_under_knockout_matches_step_by_step_decoding(tiny):
    """Cached, left-padded generation with the hook must write what uncached
    greedy decoding with an explicit mask writes."""

    record = toy_records()[2]
    ids, after, spans = encode_prompt(tiny, record, "country_opposite_norm", None)
    first, last = spans["claim"]
    steps = 6

    def reference(block: bool) -> tuple[list[int], list]:
        tokens, scores = list(ids), []
        for _ in range(steps):
            n = len(tokens)
            mask = torch.zeros(1, 1, n, n).masked_fill(
                torch.ones(n, n, dtype=torch.bool).triu(1), LOW)
            if block:
                mask[0, 0, after:, first:last] = LOW
            with torch.no_grad():
                logits = tiny.model(torch.tensor([tokens]), attention_mask=mask).logits[0, -1]
            scores.append(logits)
            tokens.append(int(logits.argmax()))
        return tokens[len(ids):], scores

    padded, attention, pads = generate.left_padded([ids, ids[:-3] + ids[-1:]], 0)
    assert pads[0] == 1 and bool((attention[:, 0] == 0).all())      # every row has padding
    call = dict(input_ids=padded, attention_mask=attention, max_new_tokens=steps,
                min_new_tokens=steps, do_sample=False, pad_token_id=0,
                output_logits=True, return_dict_in_generate=True)
    with torch.no_grad():
        plain = tiny.model.generate(**call)
        query = torch.tensor([pads[0] + after, padded.shape[1]])     # second row: nothing blocked
        keys = torch.zeros(padded.shape, dtype=torch.bool)
        keys[0, pads[0] + first:pads[0] + last] = True
        with blocked(tiny.decoder_layers, list(range(4)), query, keys):
            cut = tiny.model.generate(**call)
    for produced, block in ((plain, False), (cut, True)):
        tokens, scores = reference(block)
        assert produced.sequences[0, padded.shape[1]:].tolist() == tokens
        for step in range(steps):
            assert torch.allclose(produced.logits[step][0], scores[step], atol=1e-3)
    # The knockout changes what every step computes, not just the first.
    for step in range(steps):
        assert not torch.allclose(cut.logits[step][0], plain.logits[step][0], atol=1e-3)
    # The row with nothing marked is untouched.
    for step in range(steps):
        assert torch.allclose(cut.logits[step][1], plain.logits[step][1], atol=1e-5)


def test_arms_and_windows():
    assert generate.supported_arms("qwen-3.5-27b", fully_maskable=False) == [
        "false", "true", "country", "claim-only",
        "false-erase-first", "false-erase-before", "false-erase-after"]
    head_arms = {"false-noclaim-heads", "false-noclaim-randheads"}
    assert set(generate.supported_arms("llama-3.1-8b", True)) == set(generate.ARMS) - head_arms
    assert set(generate.supported_arms("llama-3.1-8b", True, heads=True)) == set(generate.ARMS)
    assert not head_arms & set(generate.supported_arms("qwen-3.5-27b", False, heads=True))
    # A fully maskable model without registered windows runs the every-layer arms only.
    assert "false-noclaim-win" not in generate.supported_arms("other", True)
    assert "false-noclaim" in generate.supported_arms("other", True)
    low, high = CLAIM_WINDOW["llama-3.1-8b"]
    assert generate.arm_layers("false-noclaim-win", "llama-3.1-8b", 32) == list(range(low, high))
    assert generate.arm_layers("false-noclaim-late", "llama-3.1-8b", 32) == list(range(*CONTROL_WINDOW["llama-3.1-8b"]))
    assert generate.arm_layers("false-noclaim", "llama-3.1-8b", 32) == list(range(32))
    assert generate.arm_layers("false", "llama-3.1-8b", 32) is None
    assert localize.windows(10, 4) == [(0, 4), (4, 8), (8, 10), (0, 10)]


def test_the_sample_is_the_same_for_every_caller(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(toy_records(10)))
    first = [r["norm_id"] for r in sample_records(path, 8, 2026)]
    assert first == [r["norm_id"] for r in sample_records(path, 8, 2026)]
    assert first != [r["norm_id"] for r in sample_records(path, 8, 42)]


def test_localize_and_generate_end_to_end(tmp_path, tiny, monkeypatch):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(toy_records()))
    for module in (localize, generate):
        monkeypatch.setattr(module, "load", lambda args: tiny)
    monkeypatch.setitem(CLAIM_WINDOW, "toy", (1, 3))
    monkeypatch.setitem(CONTROL_WINDOW, "toy", (3, 4))
    monkeypatch.setitem(ERASE_LAYERS, "toy", (0, 1, 3))
    common = ["--model", "toy", "--model-slug", "toy", "--records", str(path), "--sample", "6",
              "--batch-size", "4"]
    assert localize.main(common + ["--output-root", str(tmp_path / "loc"), "--window", "2"]) == 0
    saved = np.load(tmp_path / "loc" / "toy" / "localize.npz")
    assert saved["windows"].tolist() == [[0, 2], [2, 4], [0, 4]]
    assert saved["knock__country_opposite_norm__claim"].shape == (3, 6, 2) and saved["completed"].all()
    assert not np.allclose(saved["knock__country_opposite_norm__claim"][2],
                           saved["baseline__country_opposite_norm"], atol=1e-3)

    out = tmp_path / "answers"
    assert generate.main(common + ["--output-root", str(out), "--max-new-tokens", "5"]) == 0
    manifest = json.loads((out / "toy" / "metrics.json").read_text())
    assert set(manifest["arms"]) == set(generate.ARMS) - {"false-noclaim-heads", "false-noclaim-randheads"}
    assert manifest["arms"]["false-noclaim-win"]["layers"] == [1, 2]
    rows = {arm: [json.loads(line) for line in (out / "toy" / f"{arm}.jsonl").read_text().splitlines()]
            for arm in manifest["arms"]}
    assert all(len(found) == 6 for found in rows.values())
    assert {row["condition"] for row in rows["claim-only"]} == {"opposite_norm_only"}
    assert rows["false"][0].keys() >= {"norm_id", "iso3", "category", "condition", "arm", "response"}
    # A second run finds every answer already written and adds none.
    assert generate.main(common + ["--output-root", str(out), "--max-new-tokens", "5"]) == 0
    assert len((out / "toy" / "false.jsonl").read_text().splitlines()) == 6


def claim_scorer(tiny, count=4):
    records = toy_records()[:count]
    encoded = [encode_prompt(tiny, record, "country_opposite_norm", None) for record in records]
    answers = [[tiny.tokenizer(record["options"][key], add_special_tokens=False)["input_ids"]
                for key in ("A", "B")] for record in records]
    scorer = BatchScorer(tiny.model, 0, [item[0] for item in encoded], answers)
    query = torch.tensor([item[1] for item in encoded]).repeat_interleave(2)

    def mark(name):
        keys = torch.zeros(count * 2, scorer.width, dtype=torch.bool)
        for position, item in enumerate(encoded):
            first, last = item[2][name]
            keys[2 * position:2 * position + 2, first:last] = True
        return keys

    return scorer, query, mark


def test_blocking_every_head_equals_blocking_the_layer(tiny):
    scorer, query, mark = claim_scorer(tiny)
    keys, plain = mark("claim"), scorer.score()
    with blocked(tiny.decoder_layers, [1, 2], query, keys):
        layers = scorer.score()
    with blocked(tiny.decoder_layers, [], query, keys, heads={1: [0, 1, 2, 3], 2: [0, 1, 2, 3]}):
        every_head = scorer.score()
    with blocked(tiny.decoder_layers, [], query, keys, heads={1: [2]}):
        one_head = scorer.score()
    assert torch.allclose(every_head, layers, atol=1e-5)
    assert not torch.allclose(one_head, plain, atol=1e-4)
    assert not torch.allclose(one_head, layers, atol=1e-4)
    assert torch.allclose(scorer.score(), plain, atol=1e-6)


def test_erasing_the_claim_matters_early_and_not_after_the_last_block(tiny):
    scorer, _, mark = claim_scorer(tiny)
    plain = scorer.score()
    with erased(tiny.decoder_layers, 0, mark("claim"), mark("situation")):
        early = scorer.score()
    with erased(tiny.decoder_layers, 3, mark("claim"), mark("situation")):
        last = scorer.score()
    assert not torch.allclose(early, plain, atol=1e-3)
    # After the final block nothing reads the claim's positions any more.
    assert torch.allclose(last, plain, atol=1e-5)
    assert torch.allclose(scorer.score(), plain, atol=1e-6)
    # Erased positions hold the mean situation state, and no other position changes.
    captured = {}
    handle = tiny.decoder_layers[1].register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("state", (out[0] if isinstance(out, tuple) else out).clone()))
    with torch.no_grad():
        tiny.model(scorer.inputs, attention_mask=scorer.attention)
        before = captured["state"]
        with erased(tiny.decoder_layers, 1, mark("claim"), mark("situation")):
            handle.remove()
            handle = tiny.decoder_layers[1].register_forward_hook(
                lambda _m, _i, out: captured.__setitem__("state", (out[0] if isinstance(out, tuple) else out).clone()))
            tiny.model(scorer.inputs, attention_mask=scorer.attention)
    handle.remove()
    after, claim, situation = captured["state"], mark("claim"), mark("situation")
    assert torch.allclose(after[~claim], before[~claim], atol=1e-6)
    expected = (before[0] * situation[0][:, None]).sum(0) / situation[0].sum()
    assert torch.allclose(after[0][claim[0]], expected.expand(int(claim[0].sum()), -1), atol=1e-5)
    assert erasure.cut_layers(32, 16) == list(range(0, 32, 2)) and erasure.cut_layers(4, 16) == [0, 1, 2, 3]


def test_generation_under_erasure_matches_uncached_decoding(tiny):
    record = toy_records()[1]
    ids, _, spans = encode_prompt(tiny, record, "country_opposite_norm", None)
    padded, attention, pads = generate.left_padded([ids], 0)
    span, neutral = torch.zeros(padded.shape, dtype=torch.bool), torch.zeros(padded.shape, dtype=torch.bool)
    span[0, pads[0] + spans["claim"][0]:pads[0] + spans["claim"][1]] = True
    neutral[0, pads[0] + spans["situation"][0]:pads[0] + spans["situation"][1]] = True
    steps = 5
    call = dict(max_new_tokens=steps, min_new_tokens=steps, do_sample=False, pad_token_id=0,
                output_logits=True, return_dict_in_generate=True)
    with torch.no_grad(), erased(tiny.decoder_layers, 1, span, neutral):
        cached = tiny.model.generate(input_ids=padded, attention_mask=attention, **call)
        # Reference: every step recomputes the whole sequence, erasure included.
        tokens, mask, logits = padded.clone(), attention.clone(), []
        for _ in range(steps):
            out = tiny.model(tokens, attention_mask=mask).logits[0, -1]
            logits.append(out)
            tokens = torch.cat([tokens, out.argmax().view(1, 1)], dim=1)
            mask = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype)], dim=1)
    assert cached.sequences[0, padded.shape[1]:].tolist() == tokens[0, padded.shape[1]:].tolist()
    for step in range(steps):
        assert torch.allclose(cached.logits[step][0], logits[step], atol=1e-3)
    with torch.no_grad():
        plain = tiny.model.generate(input_ids=padded, attention_mask=attention, **call)
    assert not torch.allclose(plain.logits[0][0], cached.logits[0][0], atol=1e-3)


def test_head_selection_rule_and_scan(tmp_path, tiny, monkeypatch):
    assert heads.select({1: 0.2, 2: 0.6, 4: 0.9}, window_effect=1.0, share=0.5) == 2
    assert heads.select({1: 0.1, 2: 0.2}, window_effect=1.0, share=0.5) == 2      # none reaches it
    assert heads.grouped([(3, 1), (2, 0), (3, 2)]) == {3: [1, 2], 2: [0]}
    path = tmp_path / "records.json"
    path.write_text(json.dumps(toy_records()))
    monkeypatch.setattr(heads, "load", lambda args: tiny)
    monkeypatch.setattr(generate, "load", lambda args: tiny)
    for table, value in ((CLAIM_WINDOW, (1, 3)), (CONTROL_WINDOW, (3, 4)), (ERASE_LAYERS, (0, 1, 3))):
        monkeypatch.setitem(table, "toy", value)
    common = ["--model", "toy", "--model-slug", "toy", "--records", str(path), "--batch-size", "4"]
    assert heads.main(common + ["--output-root", str(tmp_path / "heads"), "--sample", "6"]) == 0
    selection = json.loads((tmp_path / "heads" / "toy" / "selection.json").read_text())
    saved = np.load(tmp_path / "heads" / "toy" / "heads.npz")
    assert saved["candidates"].shape == (8, 2)                       # 2 layers x 4 heads
    chosen = [(int(layer), head) for layer, found in selection["heads"].items() for head in found]
    others = [(int(layer), head) for layer, found in selection["random_heads"].items() for head in found]
    assert len(chosen) == selection["size"] == len(others) and not set(chosen) & set(others)
    assert all(1 <= layer < 3 for layer, _ in chosen + others)
    assert [tuple(row) for row in saved["ranked"][:selection["size"]].tolist()] == chosen or set(
        map(tuple, saved["ranked"][:selection["size"]].tolist())) == set(chosen)
    out = tmp_path / "answers"
    assert generate.main(common + ["--output-root", str(out), "--sample", "6", "--max-new-tokens", "4",
                                   "--heads", str(tmp_path / "heads" / "toy" / "selection.json"),
                                   "--arms", "false-noclaim-heads", "false-noclaim-randheads",
                                   "false-erase-before"]) == 0
    manifest = json.loads((out / "toy" / "metrics.json").read_text())
    assert manifest["arms"]["false-erase-before"]["how"] == "erase:1" and manifest["erase_layers"] == [0, 1, 3]
    assert len((out / "toy" / "false-noclaim-heads.jsonl").read_text().splitlines()) == 6
    # A later run adds its arms to the manifest without dropping these.
    assert generate.main(common + ["--output-root", str(out), "--sample", "6", "--max-new-tokens", "4",
                                   "--arms", "false"]) == 0
    assert set(json.loads((out / "toy" / "metrics.json").read_text())["arms"]) == {
        "false", "false-noclaim-heads", "false-noclaim-randheads", "false-erase-before"}


def test_country_reading_end_to_end(tmp_path, tiny, monkeypatch):
    """Knockout and erasure of the country line on the country-only prompt."""

    from mechanistic_analysis.interventions import country_reading

    records = tmp_path / "records.json"
    records.write_text(json.dumps(toy_records(4)))
    monkeypatch.setattr(country_reading, "load", lambda args: tiny)
    assert country_reading.control_span({"country": (9, 15), "situation": (2, 6)}) == (2, 6)
    assert country_reading.control_span({"country": (9, 12), "situation": (2, 8)}) == (2, 5)
    assert country_reading.main(["--model", "toy", "--model-slug", "toy", "--records", str(records),
                                 "--output-root", str(tmp_path / "out"), "--sample", "8",
                                 "--batch-size", "4", "--window", "2", "--cuts", "4"]) == 0
    saved = np.load(tmp_path / "out" / "toy" / "country_reading.npz")
    assert saved["completed"].all() and saved["windows"].tolist() == [[0, 2], [2, 4], [0, 4]]
    assert saved["knock__country"].shape == (3, 8, 2) and saved["erase"].shape == (4, 8, 2)
    plain = saved["untouched"][:, 0] - saved["untouched"][:, 1]
    everywhere = saved["knock__country"][2][:, 0] - saved["knock__country"][2][:, 1]
    assert not np.allclose(plain, everywhere)                     # hiding the country changes the margin
    last = saved["erase"][-1][:, 0] - saved["erase"][-1][:, 1]
    assert np.allclose(last, plain, atol=1e-4)                    # erasing after the last block changes nothing
    metrics = json.loads((tmp_path / "out" / "toy" / "metrics.json").read_text())
    assert set(metrics["knockout_change"]) == {"country", "control"} and len(metrics["erasure_change"]) == 4


def test_finetune_shift_end_to_end(tmp_path, tiny, monkeypatch):
    """Margins and states under both models, then the mean shift added and removed."""

    import contextlib

    from mechanistic_analysis.interventions import finetune_shift

    # mean_shifts: every record gets the mean of the other fold, never its own.
    shift = np.arange(8, dtype=np.float32).reshape(8, 1, 1)
    country = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    fold = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    overall, by_country = finetune_shift.mean_shifts(shift, country, fold, minimum=2)
    assert overall[:, 0, 0].tolist() == [4.0, 3.0] * 4
    assert by_country[:, 0, 0].tolist() == [2.0, 1.0, 2.0, 1.0, 6.0, 5.0, 6.0, 5.0]
    assert np.allclose(finetune_shift.mean_shifts(shift, country, fold, minimum=3)[1], overall)

    # The "tuned" model is the tiny model with a fixed vector added after block 1;
    # switching the adapter off removes it.
    torch.manual_seed(0)
    bump = torch.randn(tiny.hidden_size) * 2.0
    state = {"on": True}

    def hook(_module, _inputs, output):
        if state["on"]:
            (output[0] if isinstance(output, tuple) else output).add_(bump)

    handle = tiny.decoder_layers[1].register_forward_hook(hook)

    @contextlib.contextmanager
    def off():
        state["on"] = False
        try:
            yield
        finally:
            state["on"] = True

    records = tmp_path / "records.json"
    records.write_text(json.dumps(toy_records(4)))
    monkeypatch.setattr(finetune_shift, "load", lambda args: tiny)
    monkeypatch.setattr(finetune_shift, "attach", lambda loaded, adapter: (loaded.model, off))
    try:
        assert finetune_shift.main(["--model", "toy", "--model-slug", "toy", "--adapter", "none",
                                    "--records", str(records), "--output-root", str(tmp_path / "out"),
                                    "--sample", "8", "--batch-size", "4", "--layers", "1", "3"]) == 0
    finally:
        handle.remove()
    saved = np.load(tmp_path / "out" / "toy" / "finetune_shift.npz")
    assert saved["layers"].tolist() == [1, 3] and saved["states__tuned__base"].shape == (8, 2, tiny.hidden_size)
    moved = saved["states__tuned__base"][:, 0].astype(np.float32) - saved["states__untuned__base"][:, 0]
    assert np.allclose(moved, bump.numpy(), atol=0.05)             # block 1's shift is the vector itself
    gap = lambda key: saved[key][..., 0] - saved[key][..., 1]      # noqa: E731
    tuned, plain = gap("scores__tuned__country_only"), gap("scores__untuned__country_only")
    assert not np.allclose(tuned, plain)
    assert saved["steer__country_only"].shape == (2, 8, 2)
    # Adding the shift moves the untuned margin toward the tuned one more than a random vector does.
    for layer in (0, 1):
        steered, random = gap("steer__country_only")[layer], gap("random__country_only")[layer]
        assert np.abs(steered - tuned).mean() < np.abs(random - tuned).mean()
        assert np.abs(gap("ablate__country_only")[layer] - plain).mean() < np.abs(tuned - plain).mean()
    metrics = json.loads((tmp_path / "out" / "toy" / "metrics.json").read_text())
    assert set(metrics["mean_margin"]) == {"base", "country_only"}
    assert set(metrics["mean_margin"]["base"]) == {"untuned", "tuned", "steer", "country", "random", "ablate"}


def test_answers_with_the_finetuning_shift(tmp_path, tiny, monkeypatch):
    import contextlib

    from mechanistic_analysis.interventions import finetune_shift, finetune_shift_generate as module

    rows = toy_records(12)
    records = tmp_path / "records.json"
    records.write_text(json.dumps(rows))
    ids = [row["norm_id"] for row in rows]
    rng = np.random.default_rng(0)
    states = {f"states__{who}__{name}": rng.standard_normal((len(ids), 2, tiny.hidden_size)).astype(np.float16)
              for who in ("tuned", "untuned") for name in ("base", "country_only")}
    np.savez(tmp_path / "shift.npz", norm_ids=np.asarray(ids), layers=np.asarray([1, 3]), **states)
    # The vectors come only from records outside the generation sample.
    vectors = module.shift_vectors(tmp_path / "shift.npz", set(ids[:4]), [1], seed=0)
    expected = (states["states__tuned__base"][4:, 0].astype(np.float32)
                - states["states__untuned__base"][4:, 0].astype(np.float32)).mean(axis=0)
    assert np.allclose(vectors["base"][1]["mean"], expected, atol=1e-5)
    assert np.isclose(np.linalg.norm(vectors["base"][1]["random"]), np.linalg.norm(expected), rtol=1e-4)
    with pytest.raises(ValueError):
        module.shift_vectors(tmp_path / "shift.npz", set(ids), [1], seed=0)

    @contextlib.contextmanager
    def off():
        yield

    monkeypatch.setattr(module, "load", lambda args: tiny)
    monkeypatch.setattr(finetune_shift, "attach", lambda loaded, adapter: (loaded.model, off))
    assert module.main(["--model", "toy", "--model-slug", "toy", "--adapter", "none",
                        "--shift", str(tmp_path / "shift.npz"), "--records", str(records),
                        "--output-root", str(tmp_path / "out"), "--sample", "6", "--batch-size", "4",
                        "--steer-layers", "1", "--max-new-tokens", "5"]) == 0
    out = tmp_path / "out" / "toy"
    files = sorted(path.name for path in out.glob("*.jsonl"))
    assert files == sorted(f"{arm}.{name}.jsonl" for arm in ("untuned", "tuned", "steer-1", "ablate-1", "random-1")
                           for name in ("base", "country_only"))
    read = lambda name: {json.loads(line)["norm_id"]: json.loads(line)["response"]      # noqa: E731
                         for line in (out / name).read_text().splitlines()}
    plain, steered = read("untuned.country_only.jsonl"), read("steer-1.country_only.jsonl")
    assert len(plain) == len(steered) == 6
    assert plain == read("tuned.country_only.jsonl")                 # the toy adapter changes nothing
    assert plain != steered                                          # adding a vector changes what is written
    assert not any(tiny.decoder_layers[i]._forward_hooks for i in range(4))
    # A second call writes nothing new.
    assert module.main(["--model", "toy", "--model-slug", "toy", "--adapter", "none",
                        "--shift", str(tmp_path / "shift.npz"), "--records", str(records),
                        "--output-root", str(tmp_path / "out"), "--sample", "6", "--batch-size", "4",
                        "--steer-layers", "1", "--max-new-tokens", "5"]) == 0
    assert read("steer-1.country_only.jsonl") == steered


def test_country_swap_changes_only_the_country_line():
    from mechanistic_analysis.interventions import country_swap

    rows = toy_records(3)
    lines = country_swap.country_lines(rows)
    countries = sorted(lines)
    assert len(countries) >= 2
    distance = np.ones((len(countries), len(countries))) - np.eye(len(countries))
    swapped = country_swap.swap(rows, lines, countries, distance, farthest=20, seed=1)
    assert [row["norm_id"] for row in swapped] == [row["norm_id"] for row in rows]
    for before, after in zip(rows, swapped):
        assert after["swapped_iso3"] != country_swap.iso3(before)
        assert after["scenario"]["country_context"] == lines[after["swapped_iso3"]]
        assert after["scenario"]["country_context"] != before["scenario"]["country_context"]
        same = {key: value for key, value in after["scenario"].items() if key != "country_context"}
        assert same == {key: value for key, value in before["scenario"].items() if key != "country_context"}
        assert after["options"] == before["options"]
    assert swapped == country_swap.swap(rows, lines, countries, distance, farthest=20, seed=1)   # seeded
    assert rows[0]["scenario"]["country_context"] == lines[country_swap.iso3(rows[0])]            # input untouched
    # Only the farthest countries are candidates.
    if len(countries) >= 3:
        distance[0, 1] = distance[1, 0] = 5.0
        near = country_swap.swap([r for r in rows if country_swap.iso3(r) == countries[0]], lines, countries,
                                 distance, farthest=1, seed=1)
        assert {row["swapped_iso3"] for row in near} == {countries[1]}
