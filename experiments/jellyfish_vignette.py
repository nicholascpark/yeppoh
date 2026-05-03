#!/usr/bin/env python3
"""Jellyfish vignette — aesthetic probe, no learning.

Loads a jellyfish (bell + 6 trailing tentacles), drives the bell with a
hand-coded sinusoidal contraction, lets the tentacles trail passively.
No PPO, no reward, no training. Single creature, single env, viewer on.

Question this answers: does Genesis + this morphology + this actuator
produce something that LOOKS alive when driven by an obvious heuristic?

Standard RL debugging step before any cloud training: prove the task is
solvable before paying compute to learn it. If a hand-tuned bell-pulse
already produces a creature-feeling motion, PPO has something to learn.
If it doesn't, change morphology / actuator / reward before training.

Run (with Genesis viewer window):
    python experiments/jellyfish_vignette.py
    python experiments/jellyfish_vignette.py --scenario rapid
    python experiments/jellyfish_vignette.py --gravity            # default off
    python experiments/jellyfish_vignette.py --no-viewer           # headless metrics only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import genesis as gs

sys.path.insert(0, str(Path(__file__).parent.parent))

from yeppoh.body import build_body
from yeppoh.body.actuators import ActuatorInterface


# ── Hand-coded scenarios ──────────────────────────────────────────────
# Each row is the 5-dim action ActuatorInterface expects:
#   [amplitude (-1..1), freq_norm (0..1 → 0.5..5.0 Hz), phase_norm (0..1 → 0..2π),
#    wave_x, wave_y]
# The bell's actuator turns these into a sine: amp * sin(2π·freq·t + phase).

SCENARIOS = {
    "still":      [0.0, 0.50, 0.0, 0.0, 0.0],   # control: no contraction
    "gentle":     [0.4, 0.20, 0.0, 0.0, 0.0],   # slow soft pulse, ~1.4 Hz
    "rhythmic":   [0.7, 0.35, 0.0, 0.0, 0.0],   # the obvious "jellyfish"
    "rapid":      [0.9, 0.70, 0.0, 0.0, 0.0],   # frantic pulses, ~3.7 Hz
    "asymmetric": [0.7, 0.35, 0.5, 0.0, 0.0],   # phase-shifted, off-rhythm
}


def main():
    parser = argparse.ArgumentParser(description="Jellyfish hand-coded vignette")
    parser.add_argument("--scenario", default="rhythmic", choices=list(SCENARIOS.keys()))
    parser.add_argument("--steps", type=int, default=400, help="Control steps (~4 s at dt=0.01)")
    parser.add_argument("--no-viewer", action="store_true", help="Skip Genesis viewer (metrics only)")
    parser.add_argument("--gravity", action="store_true", help="Enable gravity (default off — jellyfish floats)")
    args = parser.parse_args()

    print(f"═══ JELLYFISH VIGNETTE ═══")
    print(f"Scenario : {args.scenario}")
    print(f"Action   : {SCENARIOS[args.scenario]}")
    print(f"Steps    : {args.steps}")
    print(f"Gravity  : {'on' if args.gravity else 'off (zero-g, jellyfish floats)'}")
    print()

    # Init Genesis (gs.gpu auto-routes to Metal on macOS)
    try:
        gs.init(backend=gs.gpu)
    except Exception:
        pass  # already initialized in this process

    sim_kwargs = dict(dt=0.01, substeps=32)
    if not args.gravity:
        # Zero-g lets the jellyfish hold its rest pose — gravity-free is
        # also closer to "in water" since we don't simulate fluid here.
        sim_kwargs["gravity"] = (0.0, 0.0, 0.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(**sim_kwargs),
        show_viewer=not args.no_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    cfg = {
        "bell_radius": 0.25,
        "n_tentacles": 6,
        "tentacle_length": 0.4,
        "n_agents": 1,  # solo contract — one actuator per part
    }
    body_plan = build_body(scene, "jellyfish", cfg)
    bell = body_plan.parts[0]  # name="bell"

    scene.build(n_envs=1)

    actuator = ActuatorInterface(body_plan.parts, dt=0.01)
    action = torch.tensor([SCENARIOS[args.scenario]], dtype=torch.float32)  # (1, 5)

    print(f"Running... ({'with viewer' if not args.no_viewer else 'headless'})")
    print()

    bell_centroids: list[np.ndarray] = []

    for step in range(args.steps):
        actuator.step({bell.name: action})
        scene.step()

        pos = bell.entity.get_particles_pos()  # (1, n_particles, 3)
        centroid = pos.mean(dim=1).squeeze(0).detach().cpu().numpy()
        bell_centroids.append(centroid)

        if step > 0 and step % 50 == 0:
            print(f"  step {step:>3}: bell ({centroid[0]:+.3f}, {centroid[1]:+.3f}, {centroid[2]:+.3f})")

    bell_centroids = np.array(bell_centroids)

    # ── Diagnostic summary ─────────────────────────────────────────────
    drift = bell_centroids[-1] - bell_centroids[0]
    z_swing = bell_centroids[:, 2].max() - bell_centroids[:, 2].min()
    horiz = float(np.linalg.norm(bell_centroids[-1, :2] - bell_centroids[0, :2]))

    print()
    print("═══ Bell trajectory ═══")
    print(f"Start position    : ({bell_centroids[0, 0]:+.3f}, {bell_centroids[0, 1]:+.3f}, {bell_centroids[0, 2]:+.3f})")
    print(f"End   position    : ({bell_centroids[-1, 0]:+.3f}, {bell_centroids[-1, 1]:+.3f}, {bell_centroids[-1, 2]:+.3f})")
    print(f"Net drift         : ({drift[0]:+.3f}, {drift[1]:+.3f}, {drift[2]:+.3f})")
    print(f"Z-axis pulse swing: {z_swing:.4f}     (rhythmic contraction amplitude)")
    print(f"Horizontal travel : {horiz:.4f}    (translational locomotion)")

    # Aesthetic verdict — extremely simple heuristics, just to surface the question.
    print()
    if z_swing < 0.005:
        print("→ Bell barely pulsed. Either amplitude too low or actuator not engaged.")
    elif horiz < 0.02 and z_swing > 0.01:
        print("→ Bell pulses but doesn't translate. Looks alive *in place* — could be the aesthetic, or could mean we need fluid for locomotion.")
    elif horiz > 0.02:
        print("→ Bell pulses AND drifts. Closest to 'swimming' from a hand-coded baseline.")
    else:
        print("→ Mixed: bell moves but no clear pattern.")
    print()
    print("Watch the viewer (or rerun with different --scenario). The numbers")
    print("are scaffolding — the real verdict is your eye on the motion.")


if __name__ == "__main__":
    main()
