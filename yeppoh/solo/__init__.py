"""Solo single-agent Yeppoh training — the Phylum I sculpture loop.

A minimal parallel codepath to `yeppoh.env` / `yeppoh.brain`. One body,
one policy, standard PPO. Vectorized across n_envs. Independent of the
MARL scaffolding — nothing here imports `yeppoh.env.multi_agent`,
`yeppoh.brain.policy`, or `yeppoh.training.algorithms`.

Use this for Phylum I curation runs. Use `yeppoh.env` + `yeppoh.brain`
for multi-agent research (currently inert in Phylum I).
"""

from .env import SoloYeppohEnv
from .policy import SoloPolicy
from .ppo import PPO
from .runner import run_solo

__all__ = ["SoloYeppohEnv", "SoloPolicy", "PPO", "run_solo"]
