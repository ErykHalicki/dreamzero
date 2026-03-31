# Model Feasibility Analysis: DreamZero-DROID on Single NVIDIA GB10

## Model Overview

- **Checkpoint:** DreamZero-DROID
- **Architecture:** Vision-Language-Action (VLA) model
  - Backbone: CausalWanModel (flow matching diffusion), 40 layers, dim=5120, ffn_dim=13,824
  - Components: diffusion backbone + image encoder (WanImageEncoder) + text encoder (WanTextEncoder) + VAE (WanVideoVAE) + VL self-attention
  - Action horizon: 24 frames, video sequence length: 880 frames
- **Weight size:** 45,848,344,232 bytes (~42.7 GB, bfloat16, split across 10 shards)
- **Estimated parameter count:** ~14B

## NVIDIA GB10 (DGX Spark) Specifications

| Spec | Value |
|---|---|
| Unified memory | 128 GB LPDDR5X |
| Memory bandwidth | ~273 GB/s |
| AI compute (FP8) | ~1 PFLOP |

## Memory Budget Estimate (Inference)

| Component | Estimated Size |
|---|---|
| Model weights (bfloat16) | ~42.7 GB |
| Activation memory (forward pass) | ~20–40 GB |
| OS + framework overhead | ~5–10 GB |
| **Total** | **~68–93 GB** |

The total fits within the 128 GB unified memory budget, leaving a narrow margin.

## Conclusion

**A single NVIDIA GB10 is sufficient for single-sample inference only.**

### Why it works
- 128 GB unified memory accommodates the ~42.7 GB weight footprint plus activation overhead (~70–93 GB total), which remains within the 128 GB limit.

### Why it is limited

1. **Memory bandwidth bottleneck:** LPDDR5X provides ~273 GB/s bandwidth, compared to 3.35 TB/s on an H100 SXM5 — approximately 12x slower. For a large diffusion model with an 880-frame video sequence, each denoising step will be significantly slower.

2. **No training feasibility:** Storing gradients and optimizer states requires 3–4x the weight size (~130–170 GB), which exceeds available memory. Fine-tuning or full training is not possible on a single GB10.

3. **Batch size restricted to 1:** Memory will be near capacity during inference, making batch sizes greater than 1 impractical.

### Recommendation

The GB10 is viable as a development or demo device for running single inference passes, but is not recommended for production serving or any training workload. A GPU with high-bandwidth memory (e.g., H100, A100 80GB) would provide substantially better throughput for this model.
