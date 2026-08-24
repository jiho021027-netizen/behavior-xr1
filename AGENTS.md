# Repository instructions

## Machine role

This repository is developed on a resource-constrained WSL2 laptop. Only code
editing, repository management, static analysis, configuration parsing, and
small CPU-only synthetic tests are allowed.

Never install or launch OmniGibson / Isaac Sim, download a BEHAVIOR dataset or
asset bundle, download Xiaomi-Robotics-1 or Qwen3-VL / XR-1 weights, install
CUDA / FlashAttention, initialize a large model, run training, or execute GPU
inference here.

## Upstreams

`../Xiaomi-Robotics-1` and `../BEHAVIOR-1K` are reference/vendor checkouts.
Do not modify them. Put all research code in this repository. BEHAVIOR
challenge evaluation is pinned to tag `v3.9.1`.

## Evidence discipline

When documenting upstream behavior, cite the exact repository revision, file,
and symbol. Mark dataset-dependent facts as unconfirmed until verified from
metadata on the A100/data host. Keep adapters explicit about ordering, units,
frames, normalization, padding, and masks.
