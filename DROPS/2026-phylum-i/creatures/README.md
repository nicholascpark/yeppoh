# Creatures

One subfolder per creature, named `NNN-<creature-name>/` (zero-padded, e.g.
`001-haneul/`).

Per-creature contents (per `DIRECTION.md` §6):

- `seed.txt` — integer seed used for the simulation run
- `checkpoint.pt` — trained policy weights at capture time
- `environment.yaml` — gravity, nutrient field, pheromone diffusion, etc.
- `life.mp4` — 30-120s life-story recording at 4K
- `quilt.png` — Looking Glass 45-view quilt at 8K
- `peak.stl` — mesh exported at peak-morphology frame
- `metadata.json` — NFT metadata (ERC-721 / ERC-1155 compliant)
- `notes.md` — curatorial notes: why this creature, what to see, naming rationale

Creatures are added as they are finalized. Once added and committed, a creature
folder is not modified.
