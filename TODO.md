# TODO

## SpecPrefill for Multimodal (MLLM) Models

Current SpecPrefill implementation is text-only. Extending it to support vision/video tasks could significantly reduce TTFT for long multimodal inputs (thousands of vision tokens from video frames).

### Idea

Use a small MLLM (e.g., Qwen3.5-3B) as the draft model instead of a text-only LLM. Same-family models share the vision encoder and tokenizer, so:

- Vision encoder runs once, output embeddings shared between draft and target
- Draft MLLM scores token importance via attention (including vision tokens)
- Target MLLM does sparse prefill on only the important tokens

### Why this should work

- Same-family small/large models share vision encoder + tokenizer (e.g., Qwen3.5-3B and Qwen3.5-27B)
- Cross-family SpecPrefill paper (arxiv 2603.02631) shows attention patterns correlate even across different model families; same-family correlation should be higher
- Video inputs produce thousands of vision tokens with O(n^2) attention cost; sparse prefill on a 27B model after scoring on a 3B model would be a large win
- Related work (SparseVILA, MMInference) confirms ~60-80% of visual tokens are redundant and can be pruned with VLM-aware methods

### Implementation needed

1. Pass vision embeddings from vision encoder to both draft and target MLLM
2. Add vision token handling in `specprefill.py` importance scoring
3. Ensure vision encoder output is computed once and reused (not re-encoded per model)
4. Handle video-specific temporal token structure (Grid attention pattern) during importance scoring
5. Benchmark quality vs speedup on video understanding tasks

### References

- SpecPrefill: https://arxiv.org/abs/2502.02789
- Cross-Family Speculative Prefill: https://arxiv.org/abs/2603.02631
- SparseVILA (video-aware visual token sparsity): https://arxiv.org/abs/2510.17777
- MMInference (long-context MLLM acceleration): https://arxiv.org/abs/2504.16083
- Visual token redundancy analysis: https://arxiv.org/abs/2603.00510
