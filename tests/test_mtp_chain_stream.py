# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for chained multi-token MTP speculative decoding.

Uses a tiny synthetic model that implements the injected-MTP interface
(``__call__(return_hidden=)``, ``mtp_forward``, ``make_mtp_cache``) with
fully controlled logits, plus real ``mlx_lm`` cache classes (``KVCache`` and
``ArraysCache``) so accept/reject cache restoration is exercised for both
kinds:

- KVCache records the exact token ids fed at each position (offset + content
  assertions catch trim/replay arithmetic errors).
- ArraysCache carries an order-sensitive rolling hash of every token fed
  (any rejected token leaking into the recurrent state changes the hash).

Target rule: next token = (tok + 1) % vocab.
Draft rule:  identical, except when the chain input token equals
``wrong_token`` the draft is (tok + 2) % vocab — off by one — so the
acceptance point is precisely controllable.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

# All tensor work in this file must stay on CPU (benchmarks may be running on
# the GPU). Set the default device before mlx_lm.generate is imported so its
# module-level generation_stream is created on CPU as well.
mx.set_default_device(mx.cpu)

import vllm_mlx  # noqa: E402
from vllm_mlx.mtp_chain_stream import (  # noqa: E402
    cache_supports_chained_mtp,
    mtp_chain_stream_generate,
)

VOCAB = 64
HIDDEN = 8
HASH_MOD = 1000003


def test_imports_worktree_vllm_mlx():
    """The tests must exercise this worktree's code, not the editable install."""
    print(f"vllm_mlx.__file__ = {vllm_mlx.__file__}")
    assert "vllm-mlx-mtp-depth" in vllm_mlx.__file__


@pytest.fixture(autouse=True)
def _cpu_and_no_wired_limit(monkeypatch):
    """Keep Metal untouched: CPU device and a no-op wired_limit."""
    mx.set_default_device(mx.cpu)
    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(
        mlx_generate, "wired_limit", lambda *a, **k: contextlib.nullcontext()
    )
    yield


class StubDetokenizer:
    def __init__(self):
        self.tokens = []
        self.last_segment = ""

    def add_token(self, tok):
        self.tokens.append(int(tok))
        self.last_segment = f"<{int(tok)}>"

    def finalize(self):
        self.last_segment = ""


class StubTokenizer:
    bos_token = None

    def __init__(self, eos_id=9999):
        self.eos_token_ids = {eos_id}
        self.detokenizer = StubDetokenizer()

    def encode(self, s, add_special_tokens=True):
        return [int(x) for x in s.split()]


def _one_hot_logits(next_tokens):
    """(B, T) int -> (B, T, V) logits peaking at next_tokens."""
    vocab_ids = mx.arange(VOCAB)[None, None, :]
    return mx.where(vocab_ids == next_tokens[..., None], 5.0, 0.0)


def _rolling_hash(tokens, start=0):
    s = start
    for t in tokens:
        s = (s * 31 + int(t) + 1) % HASH_MOD
    return s


class StubMTPModel:
    """Tiny deterministic target+MTP pair over real mlx_lm cache classes."""

    def __init__(self, wrong_token=None):
        from mlx_lm.models.cache import ArraysCache, KVCache

        self._KVCache = KVCache
        self._ArraysCache = ArraysCache
        self.mtp = object()  # non-None marker, mirroring the injected patch
        self.wrong_token = wrong_token
        # Telemetry for assertions
        self.mtp_calls = []  # (input_token, mtp_cache_offset_at_call)
        self.forward_lens = []  # T of every cached target forward
        self.fed_tokens = []  # every token fed with a cache (incl. replays)

    # mlx_lm's wired_limit walks parameters; keep it trivial.
    def parameters(self):
        return {}

    def make_cache(self):
        return [self._KVCache(), self._ArraysCache(size=1)]

    def make_mtp_cache(self):
        return [self._KVCache()]

    def _draft_rule(self, tok):
        if self.wrong_token is not None and tok == self.wrong_token:
            return (tok + 2) % VOCAB
        return (tok + 1) % VOCAB

    def __call__(self, inputs, cache=None, return_hidden=False):
        toks = inputs  # (1, T) int
        seq = toks.shape[1]
        if cache is not None:
            self.forward_lens.append(seq)
            kv, arr = cache
            k = toks.astype(mx.float32)[:, None, :, None]  # (1, 1, T, 1)
            kv.update_and_fetch(k, k)
            token_list = [int(t) for t in toks[0].tolist()]
            self.fed_tokens.extend(token_list)
            state = arr[0]
            start = int(state[0].item()) if state is not None else 0
            # Rebind (not mutate) the state slot, like GatedDeltaNet does.
            arr[0] = mx.array([_rolling_hash(token_list, start)], dtype=mx.int32)
        logits = _one_hot_logits((toks + 1) % VOCAB)
        if return_hidden:
            hidden = mx.broadcast_to(
                toks[..., None].astype(mx.float32), (1, seq, HIDDEN)
            )
            return logits, hidden
        return logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        cache=None,
        mtp_cache=None,
        return_hidden=False,
    ):
        assert mtp_cache is not None, "generator must pass the drafter cache"
        self.seen_mtp_cache = mtp_cache
        seq = next_token_ids.shape[1]
        tok = int(next_token_ids[0, -1].item())
        self.mtp_calls.append((tok, mtp_cache[0].offset, seq))
        # Exercise the persistent drafter KV cache like a real attention layer.
        k = next_token_ids.astype(mx.float32)[:, None, :, None]
        mtp_cache[0].update_and_fetch(k, k)
        draft = self._draft_rule(tok)
        logits = _one_hot_logits(mx.array([[draft]]))
        if return_hidden:
            hidden = mx.full((1, 1, HIDDEN), float(tok))
            return logits, hidden
        return logits


def _run(model, prompt, *, max_tokens, num_draft_tokens, eos_id=9999, stats=None):
    tokenizer = StubTokenizer(eos_id=eos_id)
    responses = list(
        mtp_chain_stream_generate(
            model,
            tokenizer,
            mx.array(prompt),
            max_tokens=max_tokens,
            num_draft_tokens=num_draft_tokens,
            stats=stats,
        )
    )
    return responses, tokenizer.detokenizer


class TestChainedDraftAndAcceptance:
    def test_full_acceptance_emits_n_drafts_per_round(self):
        model = StubMTPModel(wrong_token=None)
        stats = {}
        responses, detok = _run(
            model, [0], max_tokens=10, num_draft_tokens=3, stats=stats
        )

        # Output must equal the pure autoregressive sequence 1..10.
        assert detok.tokens == list(range(1, 11))
        assert responses[-1].finish_reason == "length"
        # Rounds: 3+3 drafts fully accepted, then a 0-draft closing step.
        assert stats["rounds"] == 3
        assert stats["drafted"] == 6
        assert stats["accepted"] == 6
        # Drafter-call schedule with the persistent seeded cache:
        # bootstrap pair (y0=1) at offset 0; round 1 chains d2,d3 from seed=2
        # then advance-feeds the 4 committed pairs (last token = bonus 5);
        # round 2 likewise; the closing 0-draft round only advance-feeds the
        # bonus. Tuples are (last_token, cache_offset_at_call, batch_T).
        assert model.mtp_calls == [
            (1, 0, 1),  # bootstrap
            (2, 1, 1),  # r1 chain -> d2
            (3, 2, 1),  # r1 chain -> d3
            (5, 1, 4),  # r1 advance: pairs for d1,d2,d3,bonus
            (6, 5, 1),  # r2 chain
            (7, 6, 1),  # r2 chain
            (9, 5, 4),  # r2 advance
            (10, 9, 1),  # r3 (0-draft) advance: bonus pair only
        ]
        # Persistent drafter cache holds exactly one pair per committed
        # position at the end (positions 0..9 -> pairs 0..9).
        assert model.seen_mtp_cache[0].offset == 10
        # No rejected round -> no replay forwards: prefill(1), verify(4),
        # verify(4), closing verify(1).
        assert model.forward_lens == [1, 4, 4, 1]
        # Cumulative per-request counters ride on every response.
        assert responses[-1].mtp_drafts == 6
        assert responses[-1].mtp_accepted == 6

    def test_partial_acceptance_accepts_exact_prefix(self):
        model = StubMTPModel(wrong_token=3)
        stats = {}
        responses, detok = _run(
            model, [0], max_tokens=8, num_draft_tokens=4, stats=stats
        )

        # Despite the draft diverging at chain input 3, the emitted sequence
        # must be exactly the autoregressive one.
        assert detok.tokens == list(range(1, 9))
        assert responses[-1].finish_reason == "length"
        # Round 1: drafts [2,3,5,6] vs targets [2,3,4,...] -> 2 accepted.
        # Round 2: drafts [5,6,7] all accepted.
        assert stats["rounds"] == 2
        assert stats["drafted"] == 7
        assert stats["accepted"] == 5
        # Rejected round triggered one replay of the accepted prefix
        # [y, d1, d2] (3 tokens): prefill(1), verify(5), replay(3), verify(4).
        assert model.forward_lens == [1, 5, 3, 4]

    def test_full_rejection_still_advances_one_token(self):
        model = StubMTPModel(wrong_token=1)
        stats = {}
        responses, detok = _run(
            model, [0], max_tokens=4, num_draft_tokens=2, stats=stats
        )

        assert detok.tokens == [1, 2, 3, 4]
        # Round 1: drafts [3, 4] vs targets [2, ...] -> 0 accepted, bonus 2.
        assert stats["drafted"] == 3  # 2 + 1 (second round capped by budget)
        assert stats["accepted"] == 1
        # prefill(1), verify(3), replay(1), verify(2)
        assert model.forward_lens == [1, 3, 1, 2]

    def test_eos_inside_accepted_block_stops_without_emitting_eos(self):
        model = StubMTPModel(wrong_token=None)
        responses, detok = _run(model, [0], max_tokens=20, num_draft_tokens=3, eos_id=4)

        assert detok.tokens == [1, 2, 3]  # 4 is EOS: never detokenized
        assert responses[-1].finish_reason == "stop"
        assert responses[-1].token == 4

    def test_depth_one_chain_matches_plain_greedy(self):
        model = StubMTPModel(wrong_token=None)
        responses, detok = _run(model, [0], max_tokens=6, num_draft_tokens=1)
        assert detok.tokens == list(range(1, 7))
        assert responses[-1].finish_reason == "length"

    def test_prompt_prefill_feeds_drafter_pairs(self):
        """Chunked prompt prefill must feed the drafter one (hidden, next
        token) pair per prompt position, in order, before the bootstrap."""
        model = StubMTPModel(wrong_token=None)
        tokenizer = StubTokenizer()
        list(
            mtp_chain_stream_generate(
                model,
                tokenizer,
                mx.array([10, 11, 12, 13, 14, 15]),
                max_tokens=4,
                num_draft_tokens=2,
                prefill_step_size=2,
            )
        )
        # prompt[:-1] chunks of 2: pairs for tokens [11,12], [13,14], [15];
        # then the bootstrap pair for y0=16 at offset 5.
        assert model.mtp_calls[:4] == [
            (12, 0, 2),
            (14, 2, 2),
            (15, 4, 1),
            (16, 5, 1),  # bootstrap
        ]
        # Output must still be the plain autoregressive continuation.
        assert tokenizer.detokenizer.tokens == [16, 17, 18, 19]


class TestCacheConsistency:
    def test_cache_matches_committed_sequence_after_rejections(self):
        """After accept/reject rounds, both cache kinds must hold exactly the
        state produced by feeding only the committed tokens, in order."""
        from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

        model = StubMTPModel(wrong_token=3)
        tokenizer = StubTokenizer()
        cache = make_prompt_cache(model)
        assert isinstance(cache[0], KVCache) and isinstance(cache[1], ArraysCache)

        list(
            mtp_chain_stream_generate(
                model,
                tokenizer,
                mx.array([0]),
                max_tokens=8,
                num_draft_tokens=4,
                prompt_cache=cache,
            )
        )

        # Committed fed tokens: prompt [0], then every emitted token except
        # the final bonus (which is sampled but never fed): 0..7.
        committed = list(range(0, 8))
        kv, arr = cache
        assert kv.offset == len(committed)
        fed = [int(v) for v in kv.keys[0, 0, : kv.offset, 0].tolist()]
        assert fed == committed
        expected_hash = _rolling_hash(committed)
        assert int(arr[0][0].item()) == expected_hash

    def test_cache_matches_committed_sequence_full_acceptance(self):
        from mlx_lm.models.cache import make_prompt_cache

        model = StubMTPModel(wrong_token=None)
        tokenizer = StubTokenizer()
        cache = make_prompt_cache(model)

        list(
            mtp_chain_stream_generate(
                model,
                tokenizer,
                mx.array([0]),
                max_tokens=10,
                num_draft_tokens=3,
                prompt_cache=cache,
            )
        )

        committed = list(range(0, 10))
        kv, arr = cache
        assert kv.offset == len(committed)
        fed = [int(v) for v in kv.keys[0, 0, : kv.offset, 0].tolist()]
        assert fed == committed
        assert int(arr[0][0].item()) == _rolling_hash(committed)


class TestCapabilityGates:
    def test_cache_supports_chained_mtp(self):
        from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

        assert cache_supports_chained_mtp([KVCache(), ArraysCache(size=1)])
        assert cache_supports_chained_mtp([KVCache()])
        assert not cache_supports_chained_mtp([])
        assert not cache_supports_chained_mtp(None)
        assert not cache_supports_chained_mtp([KVCache(), RotatingKVCache(max_size=8)])
        assert not cache_supports_chained_mtp([object()])

    def test_generator_refuses_model_without_mtp_interface(self):
        class NoMTP:
            mtp = None

        with pytest.raises(ValueError, match="MTP interface"):
            list(
                mtp_chain_stream_generate(
                    NoMTP(), StubTokenizer(), mx.array([0]), num_draft_tokens=2
                )
            )

    def test_generator_refuses_unsupported_cache_entries(self):
        model = StubMTPModel()
        with pytest.raises(ValueError, match="unsupported"):
            list(
                mtp_chain_stream_generate(
                    model,
                    StubTokenizer(),
                    mx.array([0]),
                    num_draft_tokens=2,
                    prompt_cache=[object()],
                )
            )


class TestRealTinyQwen35:
    """End-to-end checks against a real (tiny, random) mlx_lm qwen3_5 hybrid
    TextModel with the actual injected MTP patch: 3 GatedDeltaNet layers with
    ArraysCache + 1 full-attention layer with KVCache.

    The gold invariant: greedy chained-MTP decoding must produce EXACTLY the
    same tokens as plain greedy decoding, for any acceptance pattern. With a
    random MTP head most drafts are rejected, so this exercises the
    trim + ArraysCache-snapshot + replay path on the real layers.
    """

    CFG = dict(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        rms_norm_eps=1e-6,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
        mtp_num_hidden_layers=1,
    )

    def _build(self, seed):
        import os
        import tempfile

        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        from vllm_mlx.patches.qwen3_5_mtp import inject_mtp_support

        mx.random.seed(seed)
        model = TextModel(TextModelArgs.from_dict(self.CFG))
        # Keep training=True: the GatedDeltaNet CPU (non-kernel) path.
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "mtp"))
            fc = mx.random.normal((32, 64)) * 0.05
            mx.save_safetensors(
                os.path.join(td, "mtp", "weights.safetensors"),
                {"mtp.fc.weight": fc},
            )
            assert inject_mtp_support(model, td, self.CFG)
        return model

    def _greedy_reference(self, model, prompt, n):
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(model)
        logits = model(prompt[None], cache=cache)
        out = [int(mx.argmax(logits[:, -1, :]).item())]
        for _ in range(n - 1):
            logits = model(mx.array([[out[-1]]]), cache=cache)
            out.append(int(mx.argmax(logits[:, -1, :]).item()))
        return out

    def test_injected_mtp_forward_return_hidden(self):
        model = self._build(seed=0)
        h = mx.random.normal((1, 1, 32))
        mtp_cache = model.make_mtp_cache()

        logits, hidden = model.mtp_forward(
            h, mx.array([[5]]), mtp_cache=mtp_cache, return_hidden=True
        )
        assert logits.shape == (1, 1, 64)
        assert hidden.shape == (1, 1, 32)
        assert mtp_cache[0].offset == 1

        # Chain step 2 consumes step 1's hidden; the MTP KV cache grows.
        logits2, hidden2 = model.mtp_forward(
            hidden, mx.array([[7]]), mtp_cache=mtp_cache, return_hidden=True
        )
        assert logits2.shape == (1, 1, 64)
        assert hidden2.shape == (1, 1, 32)
        assert mtp_cache[0].offset == 2

        # Depth-1 legacy call shape is unchanged (no return_hidden).
        legacy = model.mtp_forward(h, mx.array([[5]]), mtp_cache=None)
        assert legacy.shape == (1, 1, 64)

    @pytest.mark.parametrize("seed", [2, 7])
    def test_chained_output_matches_plain_greedy(self, seed):
        model = self._build(seed=seed)
        prompt = mx.array([3, 14, 15, 9, 2, 6])
        reference = self._greedy_reference(model, prompt, 12)

        for num_draft in (1, 3, 4):
            tokenizer = StubTokenizer()
            stats = {}
            list(
                mtp_chain_stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=12,
                    num_draft_tokens=num_draft,
                    stats=stats,
                )
            )
            assert tokenizer.detokenizer.tokens == reference, (
                f"seed={seed} N={num_draft}: speculative output diverged "
                f"from plain greedy"
            )
            assert stats["drafted"] > 0


class TestEngineRouting:
    """Gating in SimpleEngine._stream_generate_text: chained MTP only for
    num_draft_tokens > 1, temperature == 0, chain-capable engines; everything
    else must take the exact pre-existing mlx_lm.stream_generate path."""

    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    def _engine(self, num_draft_tokens, chain_capable=True):
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "user: hi assistant:"
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 99

        text_model = MagicMock()
        text_model.mtp = object()

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=num_draft_tokens,
        )
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer
        engine._supports_system_kv_cache = False
        engine._mtp_chain_capable = chain_capable
        engine._mtp_effective_draft_tokens = (
            num_draft_tokens if (num_draft_tokens > 1 and chain_capable) else 1
        )
        return engine

    async def _collect(self, engine, temperature):
        chunks = []
        async for c in engine._stream_generate_text(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4,
            temperature=temperature,
            top_p=1.0,
        ):
            chunks.append(c)
        return chunks

    def _vanilla_response(self):
        return SimpleNamespace(
            text="ok",
            finish_reason="stop",
            token=7,
        )

    @pytest.mark.anyio
    async def test_greedy_multi_draft_routes_through_chained_generator(self):
        engine = self._engine(num_draft_tokens=4)

        chained_calls = []

        def fake_chained(model, tokenizer, prompt, **kw):
            chained_calls.append(kw)
            yield SimpleNamespace(
                text="ok",
                finish_reason="stop",
                token=7,
                mtp_drafts=8,
                mtp_accepted=5,
            )

        def fake_vanilla(*a, **kw):
            raise AssertionError("vanilla stream_generate must not run")
            yield  # pragma: no cover

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch("mlx_lm.sample_utils.make_logits_processors", return_value=[]),
            patch("mlx_lm.stream_generate", side_effect=fake_vanilla),
            patch(
                "vllm_mlx.mtp_chain_stream.mtp_chain_stream_generate",
                side_effect=fake_chained,
            ),
        ):
            chunks = await self._collect(engine, temperature=0.0)

        assert chained_calls, "chained MTP generator was not invoked"
        assert chained_calls[0]["num_draft_tokens"] == 4
        assert chained_calls[0]["stats"] is engine._mtp_text_stats
        assert chunks and chunks[0].text == "ok"
        # Per-request counters must surface on the outputs.
        assert chunks[0].mtp_drafts == 8
        assert chunks[0].mtp_accepted == 5

    @pytest.mark.anyio
    async def test_depth_one_keeps_vanilla_path(self):
        engine = self._engine(num_draft_tokens=1)

        vanilla_kwargs = []

        def fake_vanilla(*a, **kw):
            vanilla_kwargs.append(kw)
            yield self._vanilla_response()

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch("mlx_lm.sample_utils.make_logits_processors", return_value=[]),
            patch("mlx_lm.stream_generate", side_effect=fake_vanilla),
            patch(
                "vllm_mlx.mtp_chain_stream.mtp_chain_stream_generate",
                side_effect=AssertionError("chained path must not run for N=1"),
            ),
        ):
            chunks = await self._collect(engine, temperature=0.0)

        assert vanilla_kwargs, "vanilla stream_generate was not invoked"
        # Exact pre-existing kwargs: num_draft_tokens still forwarded, no mtp=.
        assert vanilla_kwargs[0]["num_draft_tokens"] == 1
        assert "mtp" not in vanilla_kwargs[0]
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_sampling_request_falls_back_to_vanilla(self):
        engine = self._engine(num_draft_tokens=4)

        vanilla_kwargs = []

        def fake_vanilla(*a, **kw):
            vanilla_kwargs.append(kw)
            yield self._vanilla_response()

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch("mlx_lm.sample_utils.make_logits_processors", return_value=[]),
            patch("mlx_lm.stream_generate", side_effect=fake_vanilla),
            patch(
                "vllm_mlx.mtp_chain_stream.mtp_chain_stream_generate",
                side_effect=AssertionError(
                    "chained path must not run for temperature>0"
                ),
            ),
        ):
            chunks = await self._collect(engine, temperature=0.7)

        assert vanilla_kwargs, "vanilla stream_generate was not invoked"
        assert vanilla_kwargs[0]["num_draft_tokens"] == 4
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_incapable_cache_falls_back_to_vanilla(self):
        engine = self._engine(num_draft_tokens=4, chain_capable=False)

        vanilla_kwargs = []

        def fake_vanilla(*a, **kw):
            vanilla_kwargs.append(kw)
            yield self._vanilla_response()

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch("mlx_lm.sample_utils.make_logits_processors", return_value=[]),
            patch("mlx_lm.stream_generate", side_effect=fake_vanilla),
            patch(
                "vllm_mlx.mtp_chain_stream.mtp_chain_stream_generate",
                side_effect=AssertionError(
                    "chained path must not run when not chain-capable"
                ),
            ),
        ):
            chunks = await self._collect(engine, temperature=0.0)

        assert vanilla_kwargs, "vanilla stream_generate was not invoked"
        assert chunks and chunks[0].text == "ok"

    def test_get_stats_exposes_mtp_block(self):
        engine = self._engine(num_draft_tokens=4)
        engine._mtp_text_stats.update(
            {"requests": 2, "rounds": 10, "drafted": 40, "accepted": 30}
        )
        stats = engine.get_stats()
        assert stats["mtp"]["enabled"] is True
        assert stats["mtp"]["configured_draft_tokens"] == 4
        assert stats["mtp"]["effective_draft_tokens"] == 4
        assert stats["mtp"]["text_route"]["acceptance_rate"] == 0.75
