# behavior-xr1

Research scaffold for adapting Xiaomi Robotics XR-1 to the 2026 BEHAVIOR
Challenge. The upstream repositories are read-only references; all adaptation
code belongs here.

This checkout is intentionally lightweight. Do not install or launch
OmniGibson / Isaac Sim, download BEHAVIOR datasets or assets, download XR-1 or
Qwen checkpoints, install CUDA / FlashAttention, or train models on this
machine. Training, inference, and simulator evaluation belong on the later
Ubuntu A100 host.

The current milestone contains architecture analysis and interface contracts
only. See `docs/ARCHITECTURE.md` and `docs/IMPLEMENTATION_PLAN.md`.

## Lightweight checks

```bash
python3 -m compileall -q src tests
python3 -m unittest discover -s tests -v
```
