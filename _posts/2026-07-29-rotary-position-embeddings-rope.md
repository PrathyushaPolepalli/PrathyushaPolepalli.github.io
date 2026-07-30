---
title: "Rotary Position Embeddings (RoPE): From Rotation to Long Context"
description: "A geometric, mathematical, and practical guide to RoPE, including frequency design, relative-position attention, KV caching, implementation conventions, and context extension."
date: 2026-07-29 16:03:00 -0700
series: Transformer Mechanics
tags:
  - transformers
  - rope
  - positional-encoding
  - attention
  - long-context
  - llm
---

Rotary Position Embeddings, usually shortened to **RoPE**, are one of the small mechanisms carrying a surprising amount of responsibility inside modern language models.

RoPE does not add a position vector to each token embedding. Instead, it rotates pairs of query and key coordinates by position-dependent angles. The rotation gives every token an absolute phase, while the query-key dot product depends on the relative distance between their positions.

That sentence is compact. Implementing it correctly is not.

A production model also commits to:

- a rotary dimension;
- a frequency base;
- a coordinate-pairing convention;
- a position-ID convention;
- a long-context scaling recipe;
- a numerical precision policy; and
- a KV-cache interpretation.

Change one of those without changing the checkpoint consistently, and the model may still run while producing much worse answers.

This article builds RoPE from a single 2D rotation, derives its relative-position property, implements it in PyTorch, and then follows it into autoregressive decoding and long-context extensions.

<figure>
  <img src="{{ '/assets/images/rope/relative-rotation.svg' | relative_url }}" alt="Two query and key vectors are rotated by angles determined by positions m and n, leaving a relative angular difference based on n minus m.">
  <figcaption>Figure 1. Original diagram by the author, based on the RoFormer rotation formulation. Conceptual and not to scale.</figcaption>
</figure>

## Summary

1. **RoPE rotates queries and keys, not values.** The rotation happens after Q/K projection and head reshaping in a standard decoder stack.
2. **Each pair of coordinates is a 2D plane.** Position changes the vector's angle in that plane without changing its norm.
3. **The attention score becomes relative.** For query position `m` and key position `n`, the positional part of their dot product depends on `n - m`.
4. **RoPE is a bank of frequencies.** Some coordinate pairs rotate quickly and others slowly, allowing attention to represent relative offsets at several scales.
5. **The implementation is part of the checkpoint.** Adjacent-pair and split-half layouts are mathematically valid but not interchangeable after training.
6. **KV-cache offsets must be absolute and consistent.** Rotate a new query and key at the next cache position; do not rotate cached keys again.
7. **RoPE has a formula for any position, but that is not a long-context guarantee.** Extrapolation beyond the training distribution can fail.
8. **Context-extension methods change the position-frequency mapping.** Position Interpolation, dynamic/NTK-aware recipes, YaRN, LongRoPE, and XPos make different trade-offs.
9. **Precision matters.** Compute positions, inverse frequencies, and trigonometric functions in FP32 before casting to the activation dtype.

## Why attention needs a position signal

Content-only self-attention is permutation equivariant: if the input tokens are reordered, the outputs reorder in the same way. A decoder's causal mask restricts which tokens are visible, but it does not provide the model with a rich representation of distance or order.

The original Transformer addresses this by adding sinusoidal position vectors to token embeddings before the attention layers ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). Position information then flows through the query, key, and value projections along with token content.

RoPE takes a different route:

1. compute content-dependent query and key vectors;
2. divide part or all of each attention head into coordinate pairs;
3. rotate each pair by an angle determined by token position; and
4. compute attention with the rotated queries and keys.

The values remain unchanged in standard RoPE.

The resulting attention operation is conceptually:

<pre class="math-block" aria-label="Attention using rotary queries and keys">
Attention(Q, K, V)
    = softmax((RoPE(Q) RoPE(K)ᵀ) / √dₕ + mask) V
</pre>

RoPE and the causal mask solve different problems. RoPE supplies a position-dependent geometry; the mask determines which positions may interact.

## Start with one 2D plane

Take two coordinates from a query or key vector:

<pre class="math-block" aria-label="A two-dimensional vector">
x = [x₀, x₁]ᵀ
</pre>

At position `p`, rotate that pair by angle `pθ`:

<pre class="math-block" aria-label="Two-dimensional rotary position matrix">
          [ cos(pθ)  -sin(pθ) ]
R(pθ)  = [                     ]
          [ sin(pθ)   cos(pθ) ]

R(pθ)x = [
  x₀ cos(pθ) - x₁ sin(pθ),
  x₀ sin(pθ) + x₁ cos(pθ)
]ᵀ
</pre>

The token position controls phase. The content vector controls the radius and starting direction.

## Why the attention score depends on relative position

Let an unrotated query content vector be `q` and an unrotated key content vector be `k`. Rotate the query at position `m` and the key at position `n`:

<pre class="math-block" aria-label="Rotated query and key">
qₘ = R(mθ)q
kₙ = R(nθ)k
</pre>

Their dot product is:

<pre class="math-block" aria-label="Derivation of RoPE relative-position property">
qₘᵀkₙ
  = (R(mθ)q)ᵀ(R(nθ)k)
  = qᵀ R(mθ)ᵀ R(nθ) k
  = qᵀ R((n - m)θ) k
</pre>

The absolute rotations combine into one rotation based on the displacement `n - m`. This is the central algebraic property introduced by [Su et al. in RoFormer](https://arxiv.org/abs/2104.09864).

For one coordinate pair and `Δ = n - m`, the score expands to:

<pre class="math-block" aria-label="Expanded one-pair RoPE attention score">
qₘᵀkₙ
  = (q₀k₀ + q₁k₁) cos(Δθ)
  + (q₁k₀ - q₀k₁) sin(Δθ)
</pre>

The model learns query and key features that interact with both the cosine and sine of relative displacement.

### A concrete three-token example

Take the sequence:

```text
I love pizza
```

| Token | Position | Query rotation | Key rotation |
|---|---:|---:|---:|
| `I` | 0 | `0θⱼ` | `0θⱼ` |
| `love` | 1 | `1θⱼ` | `1θⱼ` |
| `pizza` | 2 | `2θⱼ` | `2θⱼ` |

The notation `θⱼ` matters: every coordinate pair has its own frequency.

Suppose the query for `pizza` at position `m = 2` attends to the key for `love` at position `n = 1`. Under the matrix convention used here:

<pre class="math-block" aria-label="Concrete one-token RoPE displacement">
q₂ᵀk₁ = qᵀ R((1 - 2)θⱼ) k
       = qᵀ R(-θⱼ) k
</pre>

Move the same relationship to positions `1002` and `1001`:

<pre class="math-block" aria-label="Shifted RoPE positions preserve displacement">
q₁₀₀₂ᵀk₁₀₀₁
  = qᵀ R((1001 - 1002)θⱼ) k
  = qᵀ R(-θⱼ) k
</pre>

The absolute phases are different, but the positional part of the score represents the same one-token displacement.

### What “relative” does not mean

The derivation does **not** prove that:

- the entire network is free of absolute-position effects;
- all relative distances are equally easy to learn;
- the model will extrapolate to arbitrary context lengths;
- attention must decay monotonically with distance; or
- RoPE replaces the need for a causal or padding mask.

It proves a narrower and valuable statement: the positional transform inside a query-key score is translation invariant.

## RoPE is a bank of frequencies

An attention head has more than two coordinates, so RoPE divides the rotary subspace into pairs. For rotary dimension `d`, pair index `j = 0, …, d/2 - 1`, and frequency base `b`:

<pre class="math-block" aria-label="RoPE inverse-frequency schedule">
θⱼ = b⁻²ʲ⁄ᵈ

angle(position p, pair j) = p θⱼ
</pre>

The common illustrative base is `10,000`, but the base is checkpoint configuration, not a universal constant.

For `d = 8` and `b = 10,000`:

| Pair `j` | `θⱼ` radians/token | Approximate period |
|---:|---:|---:|
| 0 | 1 | 6.28 tokens |
| 1 | 0.1 | 62.8 tokens |
| 2 | 0.01 | 628 tokens |
| 3 | 0.001 | 6,283 tokens |

One pair changes phase quickly. Another changes slowly. A real head uses many pairs, giving attention access to a spectrum of relative-position features.

<figure>
  <img src="{{ '/assets/images/rope/frequency-bank.svg' | relative_url }}" alt="Four rotary coordinate pairs shown as clocks with progressively slower angular frequencies.">
  <figcaption>Figure 2. Original diagram by the author. The four clocks illustrate the standard geometric frequency schedule for an eight-dimensional rotary subspace.</figcaption>
</figure>

Each individual pair is periodic, so no single pair uniquely identifies every distance. The combined phase pattern across many frequencies carries the useful signal.

### Rotary dimension is not always head dimension

Some architectures rotate the entire head. Others rotate only a prefix and leave a suffix unchanged. GPT-NeoX, for example, exposes a partial rotary factor in its configuration and implementation ([EleutherAI GPT-NeoX source](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/positional_embeddings.py)).

The frequency equation must use the configured **rotary dimension**, not the model hidden size and not automatically the full head size.

## Where RoPE sits in a decoder

A typical decoder attention block:

1. projects hidden states into Q, K, and V;
2. reshapes them into attention heads;
3. applies RoPE to Q and K at the actual token positions;
4. stores the rotated K and ordinary V in the KV cache;
5. computes attention; and
6. merges head outputs.

<figure>
  <img src="{{ '/assets/images/rope/attention-cache-flow.svg' | relative_url }}" alt="Decoder attention flow showing Q and K projections passing through RoPE while V bypasses it, followed by rotated key and unrotated value caching.">
  <figcaption>Figure 3. Original diagram by the author, informed by RoFormer and current decoder implementations. RoPE transforms Q and K; standard values remain unrotated.</figcaption>
</figure>

The [Hugging Face Qwen2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modular_qwen2.py) provides a concrete example: Q, K, and V can have different head counts under grouped-query attention, but RoPE is applied to Q and K before cache update and attention.

### Grouped-query and multi-query attention

GQA and MQA reduce the number of key/value heads relative to query heads. They do not change RoPE's geometry:

- rotate every query head at the token position;
- rotate every key head at the same token position;
- leave value heads unrotated; and
- repeat or share rotated keys as required by the attention implementation.

Rotate K before key-head repetition. Repeating does not require a new positional transform.

## KV caching: positions must continue, not restart

During autoregressive generation, prefill might process positions `0 … S - 1`. Those keys are rotated at their positions and stored.

At the next decode step:

1. the new token has position `S`;
2. rotate its query and key with position `S`;
3. append the rotated key and ordinary value to the cache; and
4. attend the new query against all cached rotated keys.

At the following step, use position `S + 1`.

<pre class="math-block" aria-label="Decode-time RoPE position IDs">
position_ids = cache_position  # absolute positions of the newly supplied tokens

# For an append-only cache with no eviction:
cache_position = past_seen_tokens + arange(number_of_new_tokens)
</pre>

Do not:

- start decode positions again at zero;
- apply the offset to Q but not the new K; or
- rotate old cached keys a second time.

### Sliding windows and eviction

If a serving system evicts old KV blocks, the retained keys still represent their original positions. Keep absolute positions, or rebase every retained key and every future query and key through one mathematically consistent transform.

Renumbering only the incoming query changes its relative phase against every cached key.

### Padding and packed sequences

Padding positions must be masked. For left-padded batches, many model stacks derive valid-token positions from the attention mask so that the first real token receives the expected position.

Packed training needs two aligned controls:

1. reset or continue position IDs according to the training design; and
2. prevent attention across document boundaries.

Resetting positions without a block-diagonal attention mask allows unrelated documents to interact. With a correct block-diagonal mask, resetting versus continuing positions gives each document a uniform position offset; in pure RoPE Q-K scores that offset cancels within the document. Choose the convention that matches training, batching, and cache handling.

In ideal pure-RoPE attention, shifting every valid token in one isolated sequence by the same constant cancels from query-key scores. Real batching, cache, masking, and checkpoint conventions still demand consistent position IDs.

## Two valid layouts that are not checkpoint-compatible

RoPE needs to decide which coordinates form each 2D plane.

| Layout | Coordinate pairs | Rotation helper shape |
|---|---|---|
| Adjacent-pair | `(x₀, x₁)`, `(x₂, x₃)`, … | `[-x₁, x₀, -x₃, x₂, …]` |
| Split-half | `(x₀, x_d/2)`, `(x₁, x_d/2+1)`, … | `concat(-x[d/2:], x[:d/2])` |

Both implement banks of 2D rotations. They differ by a coordinate permutation.

Before training, either basis is valid. After training, the Q/K projection weights have learned features in that basis. Running a split-half checkpoint with an adjacent-pair helper does not merely change style; it pairs different learned coordinates.

A correct conversion must permute the relevant Q/K weight axes and all associated rotary state consistently. The split-half helper in [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/positional_embeddings.py) illustrates one convention.

## A minimal PyTorch implementation

The following implementation is original code derived from the equations above. It uses adjacent pairs and supports Q and K with different head counts.

```python
import torch


def build_rope_cache(max_positions, rotary_dim, base=10_000.0, device=None):
    """Return FP32 cosine and sine tables shaped [position, pair]."""
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")

    pair_index = torch.arange(
        rotary_dim // 2,
        device=device,
        dtype=torch.float32,
    )
    inv_freq = base ** (-2.0 * pair_index / rotary_dim)
    positions = torch.arange(
        max_positions,
        device=device,
        dtype=torch.float32,
    )
    angles = positions[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(q, k, cos_cache, sin_cache, position_ids, rotary_dim):
    """
    Rotate Q and K using adjacent coordinate pairs.

    q: [batch, query_heads, sequence, head_dim]
    k: [batch, key_heads, sequence, head_dim]
    position_ids: [batch, sequence]
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be rank-4 tensors")
    if q.shape[0] != k.shape[0] or q.shape[-2:] != k.shape[-2:]:
        raise ValueError("q and k must share batch, sequence, and head dimensions")
    if rotary_dim % 2 != 0 or rotary_dim > q.shape[-1]:
        raise ValueError("invalid rotary_dim")

    # [B, S, R/2] -> [B, 1, S, R/2] for head broadcasting.
    cos = cos_cache[position_ids].unsqueeze(1)
    sin = sin_cache[position_ids].unsqueeze(1)

    def rotate(x):
        rotary = x[..., :rotary_dim]
        tail = x[..., rotary_dim:]

        # Angle arithmetic stays in FP32 even when activations are BF16/FP16.
        even = rotary[..., 0::2].float()
        odd = rotary[..., 1::2].float()
        rotated = torch.stack(
            (
                even * cos - odd * sin,
                even * sin + odd * cos,
            ),
            dim=-1,
        ).flatten(-2)

        return torch.cat((rotated.to(x.dtype), tail), dim=-1)

    return rotate(q), rotate(k)
```

Use the function after Q/K projection and head reshape. Cache the returned K and the original V.

### Three tests every implementation should pass

**Position-zero identity**

```python
zeros = torch.zeros((batch, sequence), dtype=torch.long, device=q.device)
q0, k0 = apply_rope(q, k, cos, sin, zeros, rotary_dim)
torch.testing.assert_close(q0, q)
torch.testing.assert_close(k0, k)
```

**Norm preservation**

```python
q_rot, k_rot = apply_rope(q, k, cos, sin, position_ids, rotary_dim)
torch.testing.assert_close(
    q_rot.float().norm(dim=-1),
    q.float().norm(dim=-1),
    rtol=1e-5,
    atol=1e-5,
)
```

Use a looser tolerance for low-precision activations.

**Full-sequence versus cached decoding**

Compute logits once for a complete sequence, then recompute the last-token logits by caching the prefix and rotating the final Q/K at the absolute final position. They should match within numerical tolerance.

This test catches offset mistakes and accidental re-rotation better than checking shapes.

## Sin/cos caches, precision, and kernels

### Cache by actual position

Precomputing cosine and sine avoids repeating trigonometric work. A cache is valid only for its:

- frequency base;
- rotary dimension;
- scaling recipe;
- maximum position;
- device; and
- position convention.

Gather rows by `position_ids`, not merely by local tensor index.

### Compute phase in FP32

At long positions, BF16 or FP16 cannot represent every consecutive integer. If positions are converted to a low-precision floating type before multiplication, adjacent tokens can receive indistinguishable or inaccurate phases.

Keep:

- position values;
- inverse frequencies;
- angle multiplication; and
- cosine/sine evaluation

in FP32. Cast the resulting phase tensors or rotated output only when required by the activation path. GPT-NeoX similarly constructs rotary phase values in FP32 before conversion ([GPT-NeoX source](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/positional_embeddings.py)).

### Fusion changes cost, not semantics

RoPE is elementwise and small compared with quadratic attention, but it reads and writes Q/K tensors and phase data. At high throughput, those memory operations matter.

Production kernels may fuse RoPE with:

- Q/K projection output handling;
- KV-cache writes; or
- an attention kernel's input preparation.

A fused implementation must still match an eager FP32 reference at:

- nonzero cache offsets;
- partial rotary dimensions;
- both supported pairing conventions;
- GQA/MQA head counts; and
- configured context scaling.

An accidental transpose, materialized key-head repeat, or cache copy can cost more than the rotations themselves.

## Why ordinary RoPE does not guarantee long context

The rotation formula accepts any integer position. A trained model has still only optimized its weights over a finite distribution of positions and relative offsets.

Beyond the training context:

- the model encounters phase combinations and distances not constrained by training;
- high-frequency and low-frequency bands can fail differently;
- attention patterns calibrated for the training range may shift; and
- low-precision phase calculation can add numerical error.

The [YaRN paper](https://arxiv.org/abs/2309.00071) explicitly treats context extension as a model-adaptation problem rather than assuming the unmodified formula will extrapolate reliably.

<figure>
  <img src="{{ '/assets/images/rope/context-scaling.svg' | relative_url }}" alt="Comparison of direct RoPE extrapolation, linear position interpolation, frequency rescaling, and frequency-band-specific scaling for extending context.">
  <figcaption>Figure 4. Original diagram by the author. It compares conceptual position-to-phase mappings; exact scaling formulas and parameters are checkpoint-specific.</figcaption>
</figure>

## Context-extension methods

### Direct extrapolation

Use positions beyond the training maximum with the original base and frequencies.

This is simple and may retain some capability, but it is out-of-distribution. The existence of a valid cosine and sine value is not evidence of model quality.

### Position Interpolation

[Chen et al.](https://arxiv.org/abs/2306.15595) map a longer target range back into the pretrained position range. For an extension factor `s`:

<pre class="math-block" aria-label="Linear position interpolation">
effective_position = original_position / s
</pre>

The method avoids feeding positions beyond the original numerical range, but it compresses phase differences between neighboring tokens. The paper evaluates the method with additional fine-tuning; it should not be interpreted as proof that arbitrary zero-shot interpolation is safe.

### NTK-aware and dynamic scaling

“NTK-aware” RoPE scaling emerged as a family of community and implementation techniques that change the base or frequency schedule so different bands scale non-uniformly.

There is no single universal NTK-aware formula that can be applied to every checkpoint. Hugging Face exposes a `"dynamic"` RoPE type and several parameterized variants in its [RoPE utilities documentation](https://huggingface.co/docs/transformers/main/en/internal/rope_utils). Treat the exact type and parameters as model configuration.

`NTK` stands for **Neural Tangent Kernel**, but an inference engine using dynamic or NTK-aware RoPE is not running an NTK training algorithm. Operationally, the label usually refers to a family of base- or frequency-rescaling rules intended to preserve useful attention behavior over a longer range. The name does not make the formula universal; implementations and released checkpoints can differ.

### YaRN

[YaRN](https://arxiv.org/abs/2309.00071) treats frequency bands differently rather than uniformly compressing every dimension. It combines interpolation and extrapolation behavior across the spectrum and includes attention calibration for efficient context extension.

Its thresholds and scaling parameters are empirical and model-specific. “YaRN” is not a flag with one correct value independent of the released checkpoint.

### LongRoPE

[LongRoPE](https://arxiv.org/abs/2402.13753) searches non-uniform rescaling factors across dimensions and positions and uses progressive extension. The paper reports very long contexts in its own model and training setting.

Those reported lengths are experimental results, not a generic promise for any RoPE checkpoint.

### XPos

[XPos](https://arxiv.org/abs/2212.10554) augments rotary phase with reciprocal, position-dependent scaling on queries and keys to improve length extrapolation behavior.

Because it scales magnitudes as well as angles, XPos is not pure norm-preserving RoPE. It is a related positional mechanism with a different inductive bias.

### RoPE types you will see in model configs

Current [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/main/en/internal/rope_utils) exposes six primary `rope_type` values:

| Config type | Operational meaning | Important parameters |
|---|---|---|
| `"default"` | Original checkpoint RoPE with no context scaling | `rope_theta`, optional partial rotary factor |
| `"linear"` | Divide effective positions by a scale factor | `factor` |
| `"dynamic"` | NTK-aware dynamic base/frequency rescaling | `factor`; its reference length is `max_position_embeddings` from the model configuration |
| `"yarn"` | Frequency-band-aware scaling with attention calibration | `factor`, original context, attention and ramp parameters |
| `"longrope"` | Short- and long-context factors that can vary by rotary pair | `factor`, `short_factor`, `long_factor`, `original_max_position_embeddings`, optional `attention_factor` |
| `"llama3"` | Llama 3.1-style selective low/high-frequency scaling | `factor`, low/high-frequency factors, original context |

These are serving configurations, not interchangeable names for the same transform. A model released with `"llama3"` scaling should not be silently served as `"linear"` because both claim the same maximum length.

The current Transformers API documents a configuration shaped like:

```json
{
  "rope_parameters": {
    "rope_type": "linear",
    "rope_theta": 10000.0,
    "factor": 8.0
  }
}
```

Released checkpoints and other serving libraries may expose the same information under fields such as `rope_scaling`. Field names also evolve across library versions. Read the checkpoint configuration and the exact engine version rather than translating parameters from memory.

Some architectures use different RoPE settings for different layer types, such as full-attention and sliding-window layers. In that case, one model can legitimately contain more than one RoPE configuration.

### The scaling recipe is part of the checkpoint

A deployable model should specify:

- original training context;
- target context;
- rotary dimension;
- base;
- scaling type;
- scaling factor and any band thresholds;
- attention-temperature adjustment, if any; and
- whether additional training or fine-tuning was performed.

Changing only the inference server configuration can produce a model that executes successfully but no longer matches the position distribution used during training.

### A longer position range is not free serving capacity

RoPE scaling changes positional geometry. It does not remove the memory and compute cost of storing and attending to more tokens.

For one sequence, a useful first-order KV-cache estimate is:

<pre class="math-block" aria-label="KV-cache memory estimate">
KV bytes
  ≈ 2 × layers × cached tokens × KV heads
    × head dimension × bytes per element
</pre>

The leading `2` represents keys and values. Grouped-query or multi-query attention reduces `KV heads`, while quantization, paging, offload, and allocator overhead modify the practical result.

Extending an 8K context to 128K increases the token term by `16×`. Even when the model's RoPE configuration supports that range:

- KV-cache capacity per request grows approximately linearly with cached tokens;
- decode must read a larger cache for each new token unless the architecture limits attention;
- prefill processes a much longer prompt; and
- concurrency and token throughput can fall.

`max_model_len` is therefore both a model-quality setting and a serving-capacity decision.

## RoPE versus other position strategies

| Method | Where position enters | Relative-distance behavior | Main compatibility concern |
|---|---|---|---|
| Learned absolute table | Position vector added to hidden states | Learned indirectly | Fixed table size and checkpoint-specific entries |
| Transformer sinusoidal | Fixed vector added to embeddings | Can be learned from sinusoidal structure | Addition affects all later Q/K/V projections |
| RoPE | Rotation of Q and K coordinates | Explicitly relative inside Q-K score | Pairing, rotary dimension, base, positions, scaling |
| ALiBi | Head-specific distance bias added to logits | Explicit linear penalty by distance | Slope schedule and checkpoint training |

[ALiBi](https://arxiv.org/abs/2108.12409) leaves Q, K, and V unchanged and adds a distance-dependent bias to attention logits. Its reported length extrapolation is empirical, as are the extension results for RoPE variants.

No position strategy eliminates the need to evaluate:

- retrieval at long distance;
- local language quality;
- perplexity by position;
- passkey or needle tasks;
- multi-document interference;
- generation stability; and
- serving memory and latency.

## Common implementation bugs

1. **Rotating V.** Standard RoPE checkpoints rotate Q and K only.
2. **Using model hidden size in the frequency equation.** Use the configured rotary head dimension.
3. **Assuming full-head RoPE.** Some checkpoints rotate only a prefix.
4. **Mixing adjacent-pair and split-half layouts.**
5. **Using a different frequency base from the checkpoint.**
6. **Starting decode positions at zero after prefill.**
7. **Offsetting the new Q but not the new K.**
8. **Re-rotating cached keys on every step.**
9. **Building phase values from local decode indices rather than absolute cache positions.**
10. **Computing positions and trigonometry in BF16 or FP16.**
11. **Broadcasting phase over the head axis instead of the sequence/pair axes.**
12. **Forgetting padding or packed-document attention masks.**
13. **Resetting packed positions without resetting attention boundaries.**
14. **Applying a long-context scaling type to an incompatible checkpoint.**
15. **Assuming a successful long-context forward pass proves useful long-context behavior.**

## A practical review checklist

When implementing or adopting RoPE, verify:

1. What is the head dimension?
2. What is the rotary dimension?
3. Which coordinates form each pair?
4. What base and inverse-frequency formula were used in training?
5. Is there a released scaling configuration?
6. Are position IDs absolute across prefill and decode?
7. Are cached keys already rotated?
8. Is V left unchanged?
9. How are left padding and packed sequences handled?
10. Are angle calculations FP32?
11. Does the sin/cos cache include the largest absolute position?
12. Does GQA/MQA rotate K before head sharing or repetition?
13. Does an eager reference match the fused kernel?
14. Do full-sequence and cached-decoding logits match?
15. Has quality been evaluated across the full target context, not only near position zero?

## Final mental model

> RoPE does not mainly attach the label “position 100” to a token. It changes how that token's query and key are oriented so attention can represent relationships such as “this key is five positions behind this query.”

Each token's query and key receive absolute rotations. Their Q-K interaction depends on relative displacement.

## Quick knowledge check

Open each question to reveal an interview-ready answer.

<details class="knowledge-check">
  <summary>1. Why is positional encoding needed in Transformers?</summary>
  <div class="knowledge-check__answer">
    <p>Content-only self-attention has no inherent understanding of token order: reordering the inputs simply reorders the outputs. Without positional information, sequences containing the same tokens in different orders, such as “Dog bites man” and “Man bites dog,” provide no explicit signal that their structure is different. Positional encoding injects token-position information so the model can learn word order, token relationships, and sequence structure.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>2. Why does RoPE rotate only Query (Q) and Key (K), but not Value (V)?</summary>
  <div class="knowledge-check__answer">
    <p>Standard RoPE is applied to Q and K because position needs to influence the attention score, which is based on <code>QK<sup>T</sup></code>. Rotating Q and K makes that score position-aware by incorporating relative distance. V contains the semantic information aggregated after attention weights are computed, so rotating it would alter the content passed forward without being necessary for RoPE's relative-score property.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>3. If RoPE rotates Q and K using absolute positions, why does it encode relative positions?</summary>
  <div class="knowledge-check__answer">
    <p>RoPE rotates each query and key according to its absolute position. When a query at position <code>m</code> is compared with a key at position <code>n</code>, the rotation matrices combine as <code>R(m)<sup>T</sup>R(n) = R(n - m)</code>. The attention score therefore depends on the difference between the positions rather than on either absolute position alone. An alternate rotation convention may write the sign as <code>m - n</code>; the relative-distance property is the same.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>4. Why is RoPE better than learned positional embeddings for long-context inference?</summary>
  <div class="knowledge-check__answer">
    <p>Learned positional embeddings store a separate vector for each trained position. A model trained with an 8K table does not naturally have learned entries for positions near 128K. RoPE computes position information with deterministic rotations, so it can generate phases for unseen positions. Its relative-position property and multi-frequency design make it structurally better suited to extrapolation, but they do not guarantee quality beyond training; long contexts still require scaling, adaptation, and evaluation.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>5. How does RoPE interact with the KV cache during autoregressive decoding?</summary>
  <div class="knowledge-check__answer">
    <p>During prefill, the model computes Q, K, and V for all input tokens. RoPE is applied to Q and K, and only the rotated keys and values are stored in the KV cache. Queries are discarded after the current attention computation because they are not reused. During decoding, the new token's Q, K, and V are computed, RoPE is applied to the new Q and K at the absolute cache position, and the new query attends to the cached keys plus the current key. The old keys are already rotated at their original positions and do not need to be recomputed.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>6. What happens if RoPE is completely disabled?</summary>
  <div class="knowledge-check__answer">
    <p>Without RoPE or another position mechanism, content-only self-attention lacks an explicit representation of token order and relative distance. The model would struggle with word order, syntax, and relationships between nearby and distant tokens. Disabling RoPE in a checkpoint trained with it is even more damaging because the learned Q/K projections expect the rotary geometry and would receive incompatible attention scores.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>7. Why do people call RoPE a relative positional encoding even though it uses absolute positions?</summary>
  <div class="knowledge-check__answer">
    <p>RoPE uses absolute positions to choose the rotation applied to each query and key. After the attention dot product combines those rotations, the positional factor depends only on the difference between the token positions. Because the interaction captures relative displacement rather than either absolute index alone, RoPE is commonly described as a relative positional encoding.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>8. Why does RoPE use different rotation frequencies across dimensions?</summary>
  <div class="knowledge-check__answer">
    <p>Language contains relationships at several distance scales. Fast-changing frequency pairs provide fine-grained phase differences for nearby tokens, while slow-changing pairs retain useful variation across longer spans. The combined frequency bank lets the model learn both local syntax and long-range dependencies within the same attention head.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>9. Why does extending context length with RoPE sometimes fail even though RoPE can generate arbitrary positions?</summary>
  <div class="knowledge-check__answer">
    <p>The rotation formula can produce phases for any position, but every frequency is periodic and the model was trained on only a finite distribution of positions and relative distances. At long contexts, individual frequency bands can produce repeated or poorly calibrated phase patterns, and the model encounters relationships outside its training distribution. NTK-aware scaling, YaRN, and LongRoPE modify the position-frequency mapping to preserve more useful behavior over a larger range.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>10. Why not rotate the old Keys in the KV cache again during decoding?</summary>
  <div class="knowledge-check__answer">
    <p>Cached keys were already rotated according to their original token positions during prefill or an earlier decoding step. Applying RoPE again would rotate them a second time and corrupt their positional representation. Only the new query and new key need the rotation for the current absolute position.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>11. Why does RoPE help models understand that nearby tokens are related?</summary>
  <div class="knowledge-check__answer">
    <p>RoPE expresses token distance through the phase difference between rotated queries and keys. Nearby positions create small phase differences in the slower and medium-frequency bands, while the complete frequency bank gives the model several signals from which to learn distance-sensitive patterns. This supports local syntax and longer-range relationships, although attention is not forced to decay monotonically with distance.</p>
  </div>
</details>

<details class="knowledge-check">
  <summary>12. What is the main intuition behind RoPE?</summary>
  <div class="knowledge-check__answer">
    <p>RoPE converts position into rotations of query and key vectors. Instead of adding a separate position embedding, it changes the geometry of attention so that token similarity depends on relative position. This gives the model an efficient way to represent order, distance, and relationships across the sequence.</p>
  </div>
</details>

## Sources, attribution, and diagrams

This article paraphrases technical papers and implementation documentation for explanation. All four diagrams and the PyTorch reference implementation are original work by the author; no source figure, table, prose passage, or code block is reproduced.

The RoFormer paper is credited for the rotary formulation and relative-position derivation. Later methods are attributed to their respective papers. Open-source implementations are cited only to document production conventions such as partial rotary dimensions, split-half pairing, GQA, cache handling, and configured scaling types.

License note: a public paper or repository does not automatically grant permission to reproduce its figures or prose. Hugging Face Transformers code is Apache-2.0, and GPT-NeoX code is covered by its repository license; copying their code would require compliance with the applicable terms. This article instead provides an independently written implementation derived from the cited mathematics.

## References

1. **Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; Polosukhin, I.** [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). 2017.
2. **Su, J.; Lu, Y.; Pan, S.; Murtadha, A.; Wen, B.; Liu, Y.** [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864). 2021.
3. **Chen, S.; Wong, S.; Chen, L.; Tian, Y.** [*Extending Context Window of Large Language Models via Positional Interpolation*](https://arxiv.org/abs/2306.15595). 2023.
4. **Peng, B.; Quesnelle, J.; Fan, H.; Shippole, E.** [*YaRN: Efficient Context Window Extension of Large Language Models*](https://arxiv.org/abs/2309.00071). 2023.
5. **Ding, Y.; Zhang, L. L.; Zhang, C.; Xu, Y.; Shang, N.; Xu, J.; Yang, F.; Yang, M.** [*LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*](https://arxiv.org/abs/2402.13753). 2024.
6. **Press, O.; Smith, N. A.; Lewis, M.** [*Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*](https://arxiv.org/abs/2108.12409). 2021.
7. **Sun, Y.; Dong, L.; Patra, B.; Ma, S.; Huang, S.; Benhaim, A.; Chaudhary, V.; Song, X.; Wei, F.** [*A Length-Extrapolatable Transformer*](https://arxiv.org/abs/2212.10554). 2022.
8. **Hugging Face.** [*RoPE utilities documentation*](https://huggingface.co/docs/transformers/main/en/internal/rope_utils). Transformers documentation.
9. **Hugging Face.** [*Qwen2 modular attention implementation*](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modular_qwen2.py). Transformers source.
10. **EleutherAI.** [*GPT-NeoX rotary-position implementation*](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/positional_embeddings.py). GPT-NeoX source.
11. **Hugging Face.** [*Transformers license*](https://github.com/huggingface/transformers/blob/main/LICENSE).
12. **EleutherAI.** [*GPT-NeoX license*](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE).

All links were accessed on July 29, 2026.

## Changelog

- **2026-07-29:** Removed three advanced mathematical side sections for a faster read.
- **2026-07-29:** Added a concrete rotation example, serving configuration guide, and KV-cache capacity model.
- **2026-07-29:** Expanded the quick knowledge check to twelve interview questions.
- **2026-07-29:** Added a ten-question quick knowledge check.
- **2026-07-29:** Initial publication.
