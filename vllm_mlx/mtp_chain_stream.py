# SPDX-License-Identifier: Apache-2.0
"""Chained multi-token MTP speculative decoding for the SimpleEngine text route.

vllm-mlx injects a one-layer MTP head into Qwen3.5/3.6-family models
(``vllm_mlx/patches/qwen3_5_mtp.py``). The stock ``mlx_lm.stream_generate``
path silently drops ``num_draft_tokens`` when no ``draft_model`` is given, so
the text route never actually speculated. This module implements the missing
loop, DeepSeek-V3 style: the single MTP layer is applied *recursively* to
draft N tokens per round, all N drafts are verified in one target forward,
and the longest exactly-matching prefix is accepted.

The drafter keeps a persistent KV cache over the whole request (upstream
mlx-vlm's Qwen MTP drafter informed this design). MTP cache position ``p``
holds the pair ``(target_hidden(p), token(p+1))``; the head's output at
pair ``p`` predicts token ``p+2``. During prompt prefill every prompt
position's pair is fed (one cheap single-layer pass per chunk), so drafts
are conditioned on the full context instead of a per-round scratch cache.
After each verify, the round's speculative chain pairs are trimmed and the
committed positions are re-fed with *target* hidden states, so the drafter
cache only ever contains full-context, target-hidden pairs. The last output
of that advance feed is the next round's first draft for free (the "seed").

Per round (greedy decoding only):

1. Chain-draft: ``d_1`` is the seed from the previous advance feed; each
   further step feeds the previous step's MTP hidden state and drafted token
   back in (DeepSeek-V3-style recursion over the single MTP layer).
2. Verify: run the target once on ``[y, d_1, ..., d_N]`` with
   ``return_hidden=True``. ``argmax`` of the target logits at position ``i``
   is the target's own choice for the token after ``d_i``.
3. Accept the longest prefix of drafts that matches the target argmax; the
   target's token at the first mismatch (or after the last accepted draft) is
   emitted as the bonus token and becomes the next round's ``y``.
4. Cache restore: the verify forward advanced the target cache by ``N + 1``
   positions, ``N - k`` of which are rejected. Trimmable entries (``KVCache``)
   are trimmed back to the pre-verify offset; ``ArraysCache`` entries
   (Mamba-style recurrent state of the hybrid linear-attention layers, which
   cannot be trimmed token-by-token) are restored from an O(1) pre-verify
   snapshot. The accepted prefix ``[y, d_1..d_k]`` is then re-run through the
   target to advance every cache entry by exactly the committed tokens.
   Fully-accepted rounds skip all of this. This trades one extra short
   forward per partially-rejected round for never having to roll recurrent
   state back token-by-token — correctness over speed.
5. Drafter advance: trim the chain pairs appended in step 1, then feed the
   committed pairs ``(Hv[i], token(i+1))`` for the accepted drafts and the
   bonus token in one batched single-layer forward; its last logits row is
   the next round's ``d_1``.

The ``ArraysCache`` snapshot is safe because ``GatedDeltaNet`` writes its
state back by *rebinding* the cache slots (``cache[0] = ...``) with freshly
computed arrays; the arrays captured by the snapshot are never mutated in
place. Only ``mlx_lm.models.cache.ArraysCache`` is accepted for the snapshot
path — unknown cache classes make the round refuse to run rather than risk
silently corrupting state (see :func:`cache_supports_chained_mtp`).

Sampling: this generator is greedy-only (draft and verify both use argmax).
Callers must route ``temperature > 0`` requests to the non-speculative path.

Numerics: on Metal the batched verify forward (T = N+1) does not produce
bit-identical logits to a single-token forward — different matmul/kernel
paths accumulate in a different order, and near-tied top-2 candidates can
flip. Measured on Qwen3.8-27B-4bit this flips roughly one argmax per
50-150 generated tokens, so speculative output is greedy-equivalent but not
guaranteed byte-identical to non-speculative decoding. This is inherent to
batched verification (mlx_lm's own draft-model speculative path has the
same property), not an acceptance-logic defect: in observed divergences the
emitted token always equals the batched-verify argmax.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator, List, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def _is_snapshot_entry(entry: Any) -> bool:
    """True for cache entries restored via state snapshot (ArraysCache only)."""
    from mlx_lm.models.cache import ArraysCache

    return isinstance(entry, ArraysCache)


def _is_trim_entry(entry: Any) -> bool:
    """True for cache entries restored via ``trim``.

    ``RotatingKVCache`` is excluded even though it reports trimmable while
    below capacity: once the ring buffer wraps mid-generation, trim can no
    longer restore the pre-verify state and speculation would silently
    corrupt the cache.
    """
    from mlx_lm.models.cache import RotatingKVCache

    if isinstance(entry, RotatingKVCache):
        return False
    is_trimmable = getattr(entry, "is_trimmable", None)
    return callable(is_trimmable) and entry.is_trimmable()


def cache_supports_chained_mtp(cache: Optional[List[Any]]) -> bool:
    """Whether every cache entry can be restored to the acceptance point."""
    if not cache:
        return False
    return all(_is_trim_entry(c) or _is_snapshot_entry(c) for c in cache)


def model_supports_chained_mtp(model: Any) -> bool:
    """Whether the model exposes the injected MTP interface we need."""
    return (
        getattr(model, "mtp", None) is not None
        and callable(getattr(model, "mtp_forward", None))
        and callable(getattr(model, "make_mtp_cache", None))
    )


def mtp_chain_stream_generate(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    *,
    max_tokens: int = 256,
    num_draft_tokens: int = 2,
    prompt_cache: Optional[List[Any]] = None,
    prefill_step_size: int = 2048,
    stats: Optional[dict] = None,
    round_log: Optional[list] = None,
) -> Generator[Any, None, None]:
    """Greedy chained-MTP speculative stream mirroring ``mlx_lm.stream_generate``.

    Yields ``mlx_lm.generate.GenerationResponse`` objects with the same
    text/EOS/max_tokens semantics as ``mlx_lm.stream_generate``. Each response
    additionally carries cumulative ``mtp_drafts`` / ``mtp_accepted``
    attributes for this request.

    Args:
        model: Target model with injected MTP support (``mtp_forward`` with
          ``return_hidden``, ``make_mtp_cache``, ``__call__(return_hidden=)``).
        tokenizer: Tokenizer (wrapped in ``TokenizerWrapper`` if needed).
        prompt: Prompt string, token list, or ``mx.array`` of token ids.
        max_tokens: Maximum number of generated tokens.
        num_draft_tokens: Draft chain depth N per speculative round.
        prompt_cache: Optional pre-populated *backbone* cache (no MTP entries).
          Updated in place. Created via ``make_prompt_cache(model)`` if None.
        prefill_step_size: Chunk size for prompt prefill.
        stats: Optional dict of cumulative counters (``requests``, ``rounds``,
          ``drafted``, ``accepted``) updated in place across requests.
        round_log: Optional list; when given, a per-round diagnostics dict
          (drafts, verify targets, accepted count) is appended for each
          speculative round. Debugging aid — off in production.
    """
    from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if num_draft_tokens < 1:
        raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
    if not model_supports_chained_mtp(model):
        raise ValueError(
            "Model does not expose the injected MTP interface "
            "(mtp / mtp_forward / make_mtp_cache); refusing to run chained "
            "MTP speculation."
        )

    if not (hasattr(tokenizer, "detokenizer") and hasattr(tokenizer, "eos_token_ids")):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    if prompt.size == 0:
        raise ValueError("Chained MTP generation requires a non-empty prompt.")

    detokenizer = tokenizer.detokenizer
    eos_token_ids = tokenizer.eos_token_ids

    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    trim_entries = [c for c in cache if _is_trim_entry(c)]
    snapshot_entries = [c for c in cache if not _is_trim_entry(c)]
    unsupported = [c for c in snapshot_entries if not _is_snapshot_entry(c)]
    if unsupported:
        names = sorted({type(c).__name__ for c in unsupported})
        raise ValueError(
            f"Chained MTP requires trimmable or ArraysCache cache entries; "
            f"got unsupported entries: {names}. Refusing to run rather than "
            f"risk silent cache corruption."
        )

    mtp_cache = model.make_mtp_cache()
    if not mtp_cache or not all(
        callable(getattr(c, "is_trimmable", None)) and c.is_trimmable()
        for c in mtp_cache
    ):
        raise ValueError(
            "Chained MTP requires a trimmable MTP drafter cache "
            "(make_mtp_cache must return KVCache-like entries)."
        )

    if stats is not None:
        stats["requests"] = stats.get("requests", 0) + 1
    request_drafted = 0
    request_accepted = 0

    total_prompt_tokens = int(prompt.size)

    def _round(
        y_tok: mx.array,
        h: mx.array,
        seed_tok: mx.array,
        seed_h: mx.array,
        n_draft: int,
    ):
        """One speculative round.

        Args:
            y_tok: shape (1,) — last emitted token, not yet fed to the target.
            h: shape (1, 1, H) — target hidden state at the last fed position.
            seed_tok: shape (1, 1) — the drafter's prediction of the token
              after ``y_tok`` (from the previous advance feed); used as d_1.
            seed_h: shape (1, 1, H) — MTP hidden state paired with seed_tok.
            n_draft: chain depth for this round (may be < num_draft_tokens
              near the max_tokens budget; 0 falls back to a plain decode step).

        Returns:
            (committed, new_y, new_h, new_seed_tok, new_seed_h, accepted,
            drafted) where committed is a list of
            (token_id, logprobs_row, from_draft) in emission order — the
            accepted drafts followed by the target bonus token.
        """
        drafts = None
        chain_appended = 0
        if n_draft > 0:
            chain: List[mx.array] = [seed_tok]
            d_h = seed_h
            d_tok = seed_tok
            for _ in range(n_draft - 1):
                d_logits, d_h = model.mtp_forward(
                    d_h, d_tok, mtp_cache=mtp_cache, return_hidden=True
                )
                chain_appended += 1
                d_tok = mx.argmax(d_logits[:, -1:, :], axis=-1)  # (1, 1)
                chain.append(d_tok)
            drafts = mx.concatenate(chain, axis=1)  # (1, n_draft)
            verify_in = mx.concatenate([y_tok[None], drafts], axis=1)
        else:
            verify_in = y_tok[None]

        # O(1) pre-verify snapshot of the recurrent-state entries. GatedDeltaNet
        # rebinds cache slots with new arrays, so holding references restores
        # the exact pre-verify state.
        saved_states = (
            [list(c.state) for c in snapshot_entries] if n_draft > 0 else None
        )

        v_logits, v_hidden = model(verify_in, cache=cache, return_hidden=True)
        v_targets = mx.argmax(v_logits, axis=-1)  # (1, 1 + n_draft)
        targets_list = v_targets[0].tolist()
        drafts_list = drafts[0].tolist() if drafts is not None else []

        accepted = 0
        while accepted < n_draft and targets_list[accepted] == drafts_list[accepted]:
            accepted += 1

        if round_log is not None:
            round_log.append(
                {
                    "verify_in": [int(t) for t in verify_in[0].tolist()],
                    "drafts": list(drafts_list),
                    "targets": list(targets_list),
                    "accepted": accepted,
                }
            )

        rejected = n_draft - accepted
        if rejected > 0:
            # Roll every entry back to the pre-verify state, then advance by
            # exactly the committed tokens. KVCache: trim the whole verify
            # block; ArraysCache: restore the snapshot. The replay forward
            # re-feeds [y, d_1..d_k] so both cache kinds land at the same
            # position with state derived only from committed tokens.
            for c in trim_entries:
                c.trim(1 + n_draft)
            for c, s in zip(snapshot_entries, saved_states):
                c.state = s
            model(verify_in[:, : 1 + accepted], cache=cache)

        # Drafter advance: drop this round's speculative chain pairs, then
        # feed the committed pairs (Hv[i], token(i+1)) — accepted drafts plus
        # the bonus — with target hidden states. The last output row predicts
        # the token after the bonus: the next round's d_1, for free.
        for c in mtp_cache:
            c.trim(chain_appended)
        adv_hidden = v_hidden[:, : accepted + 1, :]
        adv_tokens = mx.concatenate(
            [verify_in[:, 1 : 1 + accepted], v_targets[:, accepted : accepted + 1]],
            axis=1,
        )
        s_logits, s_hidden = model.mtp_forward(
            adv_hidden, adv_tokens, mtp_cache=mtp_cache, return_hidden=True
        )
        new_seed_tok = mx.argmax(s_logits[:, -1:, :], axis=-1)  # (1, 1)
        new_seed_h = s_hidden[:, -1:, :]

        lp = v_logits[:, : accepted + 1, :]
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        committed = []
        for i in range(accepted):
            committed.append((drafts_list[i], lp[0, i], True))
        committed.append((targets_list[accepted], lp[0, accepted], False))

        new_y = v_targets[:, accepted]  # (1,)
        new_h = v_hidden[:, accepted : accepted + 1, :]
        return committed, new_y, new_h, new_seed_tok, new_seed_h, accepted, n_draft

    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        with mx.stream(generation_stream):
            # Chunked prefill of all but the last prompt token, feeding the
            # drafter the pair (hidden(p), token(p+1)) for every position so
            # drafts are conditioned on the full context.
            remaining = prompt
            fed = 0
            while remaining.size > 1:
                n_to_process = min(prefill_step_size, remaining.size - 1)
                _, chunk_hidden = model(
                    remaining[:n_to_process][None], cache=cache, return_hidden=True
                )
                pair_tokens = prompt[fed + 1 : fed + n_to_process + 1]
                model.mtp_forward(chunk_hidden, pair_tokens[None], mtp_cache=mtp_cache)
                mx.eval([c.state for c in cache])
                mx.eval([c.state for c in mtp_cache])
                fed += n_to_process
                remaining = remaining[n_to_process:]
                mx.clear_cache()

            logits, hidden = model(remaining[None], cache=cache, return_hidden=True)
            last_logits = logits[:, -1, :]
            y_tok = mx.argmax(last_logits, axis=-1)  # (1,)
            h = hidden[:, -1:, :]
            first_lp = (last_logits - mx.logsumexp(last_logits, keepdims=True))[0]
            # Bootstrap the drafter: feed the final prompt pair (h, y0); its
            # output is round 1's first draft.
            seed_logits, seed_hidden = model.mtp_forward(
                h, y_tok[None], mtp_cache=mtp_cache, return_hidden=True
            )
            seed_tok = mx.argmax(seed_logits[:, -1:, :], axis=-1)  # (1, 1)
            seed_h = seed_hidden[:, -1:, :]
            mx.eval(y_tok)

        prompt_time = time.perf_counter() - tic
        prompt_tps = total_prompt_tokens / max(prompt_time, 1e-9)
        tic = time.perf_counter()

        n = 0
        token = int(y_tok.item())
        logprobs = first_lp
        finish_reason: Optional[str] = None
        from_draft = False
        pending = [(token, first_lp, False)]

        def _response(fin: Optional[str]) -> Any:
            resp = GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=total_prompt_tokens,
                prompt_tps=prompt_tps,
                generation_tokens=n,
                generation_tps=n / max(time.perf_counter() - tic, 1e-9),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=fin,
            )
            resp.mtp_drafts = request_drafted
            resp.mtp_accepted = request_accepted
            return resp

        while finish_reason is None:
            for tok, lp_row, was_draft in pending:
                token = tok
                logprobs = lp_row
                from_draft = was_draft
                if tok in eos_token_ids:
                    finish_reason = "stop"
                    break
                detokenizer.add_token(tok)
                n += 1
                if n == max_tokens:
                    finish_reason = "length"
                    break
                yield _response(None)
                if n % 256 == 0:
                    mx.clear_cache()
            if finish_reason is not None:
                break

            n_draft = min(num_draft_tokens, max_tokens - n - 1)
            with mx.stream(generation_stream):
                (
                    pending,
                    y_tok,
                    h,
                    seed_tok,
                    seed_h,
                    accepted,
                    drafted,
                ) = _round(y_tok, h, seed_tok, seed_h, n_draft)
            request_drafted += drafted
            request_accepted += accepted
            if stats is not None:
                stats["rounds"] = stats.get("rounds", 0) + 1
                stats["drafted"] = stats.get("drafted", 0) + drafted
                stats["accepted"] = stats.get("accepted", 0) + accepted

        detokenizer.finalize()
        yield _response(finish_reason)
