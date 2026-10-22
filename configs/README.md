# Configuration entry points

| Path | Role |
|---|---|
| `data/collection/retrieval.py` | Current complete collection parameters |
| `hydrostate_retrieval.py` | Shared training config; imports the canonical data contract |
| `hydrostate_mtlora.py` | Shared TA-LoRA plus task-specific exits |
| `hydrostate_mtlora_shared.py` | Shared-LoRA control, `private_rank=0` |
| `hydrostate_mtlora_frozen_embed.py` | Frozen patch-embedding ablation |
| `_base_/runtime.py` | Common MMEngine runtime |
| `experiments/legacy/` | Historical architecture, data and host presets |

Machine paths are environment variables; do not duplicate scientific settings for each host.
Historical configs retain their original data definitions. The current reader rejects their
sample formats; use an explicit new current config for a controlled new-data comparison.
