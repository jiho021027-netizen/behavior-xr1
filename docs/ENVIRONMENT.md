# Development environment

Audited on 2026-08-24 (Asia/Seoul).

## Observed inside WSL2

| Item | Observed value |
|---|---|
| Kernel | Linux 6.6.87.2-microsoft-standard-WSL2, x86_64 |
| Distribution | Ubuntu 22.04.5 LTS (Jammy) |
| Python | 3.10.12 |
| Git | 2.34.1 |
| Node.js | 22.23.2 |
| WSL-visible memory | 3.7 GiB total, 3.1 GiB available; 1.0 GiB swap |
| Workspace filesystem | `/dev/sdd`, 1007 GiB total, 954 GiB available |

The WSL-visible disk and memory values differ from the user-provided host
specification (8 GB physical RAM and approximately 55 GB host SSD free). WSL
mount/accounting may not represent the Windows host capacity. The conservative
user-provided limits govern this project.

## User-provided hardware

- Windows 11 laptop with WSL2 Ubuntu 22.04
- NVIDIA GeForce MX250, 2 GB VRAM
- 8 GB host RAM
- Approximately 55 GB host SSD free

## Hard safety boundary

This machine is for source development, repository management, static analysis,
configuration parsing, and lightweight CPU-only unit tests. Do not install or
run OmniGibson, Isaac Sim, CUDA, or FlashAttention. Do not download the full
BEHAVIOR dataset/assets, Xiaomi-Robotics-1 checkpoints, Qwen3-VL/XR-1 large
models, or other multi-GB assets. Do not initialize a large model, train, run
CUDA inference, or launch a simulator.

Training, inference, checkpoint conversion, and OmniGibson evaluation are
deferred to an Ubuntu A100 server.

## Source checkouts

| Repository | Revision | Role |
|---|---|---|
| XiaomiRobotics/Xiaomi-Robotics-1 | `556cca33963a2b36d835a40374c3b4c8eef68401` (`main`) | read-only reference |
| StanfordVL/BEHAVIOR-1K | `v3.9.1`, `26f2c7ef7b9cf96bd0414f81e1e751e493762779` | read-only challenge reference |

Both were cloned with partial-clone filtering and `GIT_LFS_SKIP_SMUDGE=1`.
No dataset, asset bundle, checkpoint, submodule, or simulator dependency was
downloaded. At audit time their working-tree sizes were approximately 4.5 MiB
and 1003 MiB respectively.
