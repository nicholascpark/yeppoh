"""Curriculum learning — progressive difficulty scheduling.

Starts creatures with easier tasks and gradually increases complexity:
- Phase 1: Fixed body, learn to pulse/contract
- Phase 2: Enable locomotion reward
- Phase 3: Enable growth (particle emission)
- Phase 4: Enable inter-creature interaction
- Phase 5: Full ecosystem

Curriculum stages can be time-based or performance-based.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumStage:
    name: str
    description: str
    unlock_after: int  # iterations before this stage activates
    reward_weights: dict[str, float]
    enable_growth: bool = False
    enable_communication: bool = False
    n_creatures: int = 1


DEFAULT_CURRICULUM = [
    CurriculumStage(
        name="pulse",
        description="Learn to contract and expand rhythmically",
        unlock_after=0,
        reward_weights={"survival": 1.0},
    ),
    CurriculumStage(
        name="move",
        description="Learn to move via coordinated contraction",
        unlock_after=100,
        reward_weights={"locomotion": 1.0, "survival": 0.5, "coordination": 0.3},
    ),
    CurriculumStage(
        name="grow",
        description="Learn to grow new body parts",
        unlock_after=300,
        reward_weights={"growth": 0.8, "locomotion": 0.5, "survival": 0.5},
        enable_growth=True,
    ),
    CurriculumStage(
        name="communicate",
        description="Develop inter-agent communication",
        unlock_after=500,
        reward_weights={
            "locomotion": 0.5, "growth": 0.3, "survival": 0.5, "coordination": 0.5,
        },
        enable_growth=True,
        enable_communication=True,
    ),
    CurriculumStage(
        name="ecosystem",
        description="Multiple creatures interacting",
        unlock_after=800,
        reward_weights={
            "locomotion": 0.3, "growth": 0.3, "survival": 1.0, "coordination": 0.5,
        },
        enable_growth=True,
        enable_communication=True,
        n_creatures=3,
    ),
]


class CurriculumScheduler:
    """Manages curriculum progression during training."""

    def __init__(self, stages: list[CurriculumStage] | None = None):
        self.stages = stages or DEFAULT_CURRICULUM
        self.current_stage_idx = 0

    def get_current_stage(self, iteration: int) -> CurriculumStage:
        """Get the active curriculum stage for the current iteration."""
        for i, stage in enumerate(self.stages):
            if iteration >= stage.unlock_after:
                self.current_stage_idx = i

        return self.stages[self.current_stage_idx]

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]
