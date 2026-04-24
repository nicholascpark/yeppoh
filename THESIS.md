# Yeppoh — Thesis

*Longer-form artist statement. Companion to `MANIFESTO.md` (the short public summary). This document evolves forward; earlier sections are not retracted, only built upon.*

---

## 1. What Yeppoh Is

Yeppoh is not a research project. It is a **generative-biology art practice**.

The work: unique soft-body creatures whose morphology, behavior, and life-trajectory emerge through multi-agent reinforcement learning against physics. Each creature lives once — one simulation run, one seed, one training checkpoint, one environmental condition. That run cannot be repeated identically. The creature reaches maturity and is captured.

The artist's hand = pipeline design + environmental authorship + curation of which creatures, at which moment, in which form, become presented work.

**Artistic position:** Yeppoh explores **emergence as craft** — the refusal to specify form, in favor of conditions that produce form. It is a rebuke to prompt-in-image-out generative AI. Form discovers itself over training-time.

The name 예뿌 (Korean, baby-register for "pretty") locates the practice in a specific aesthetic lineage — 간결함 (simplicity), 여백 (negative space), 한 (longing). This is legible within contemporary Korean art and the global scene.

---

## 2. The Forward-Only Principle

Every drop, every creature, every decision is **permanent and additive**. Nothing is retracted, deleted, re-minted, or disavowed. Early work becomes historically important even when later work is more refined. Collectors buy **trajectory**, not individual pieces.

This principle governs:
- **Releases** — once a creature is minted, it exists forever. No "v2" of the same creature.
- **Aesthetic direction** — style evolves, does not reset. Rei Kawakubo's 1981 collection is still Rei Kawakubo's.
- **Technical pipeline** — old checkpoints are preserved as historical record, not deprecated into oblivion.
- **Public statements** — no apology tours, no abandoning old thesis. Only evolution of the next thesis on top.

**Concrete rule:** once a drop launches, its `DROPS/YYYY-<drop-name>/` folder is permanent and never modified. From this repository's first public push forward, its git history is sacred — no force-pushes, no squash-merges that erase the evolution record. Pre-launch scaffolding history is not covered by this rule.

---

## 3. Pygmalion — Hardware-Agnostic, Uncopyability-Through-Relationship (2026-04-21)

*Supersedes earlier framing that centered specific display technologies. The medium is not the work.*

Light-field displays and volumetric installations — however sophisticated (Looking Glass, SolidLight, Light Field Lab) — are still 2D surfaces doing clever tricks to fake 3D presence. They are not "the thing is in my room."

The active spine is **creatures in goggles**. Spatial computing (Vision Pro, Quest, whatever succeeds them) puts the creature in the collector's actual perceptual space — closer to a being you share space with than an artwork on a pedestal.

### Pygmalion is the thesis

The product is not the capture. It is not the sculpture. It is not the installation.

The product is the **relationship** — a collector falling in love with a specific creature that lives alongside them. The reference is Pygmalion: the sculptor's ivory became Galatea not because he carved well but because he loved it. The collector is the second Pygmalion. The creature is alive insofar as it is loved.

### Hardware-agnostic

The asset is the creature, not the display.

- Geometry + rig — exported as USDZ / glTF with extensions.
- Policy weights — captured at maturity.
- Behavior spec — how the creature observes, acts, and adapts.

The display is whatever renders 3D in a given year. Looking Glass today as a bridge, goggles tomorrow, direct retinal projection eventually. The creature file is permanent; rendering substrates rotate beneath it.

Consequences:
- NFT metadata does not lock to any specific device. Open 3D standards only.
- Hardware ↔ NFT binding is **social / contract enforcement only** (sale clause, gallery norms). No on-chain transfer gating — hardware failure must never brick a collector's creature.

### Uncopyability through relationship

A digital file can always be byte-copied. A trained neural policy can always be weight-cloned. File-level uniqueness is not a defensible claim and Yeppoh will not make it.

The defensible claim is different: **a creature is not the file, it is the file plus the history of attention it has received.** Mint ships the creature's base policy — byte-identical across the edition. After mint, the creature continues to live in the collector's space: observing, adapting, drifting. The observer signal (gaze, interaction, time spent, context, return visits) conditions the creature's ongoing adaptation. Two collectors of adjacent editions end up with materially different creatures after six months of cohabitation — because love shaped each one on a different path.

You cannot copy someone's dog by cloning its genome. What you would miss is every walk, every greeting, every evening on the couch. Same principle here. Pygmalion's ivory was identical to any other block of ivory; what made it Galatea was the devotion it received. **Uniqueness is authored jointly — by the artist (gestation) and the collector (living).**

This resolves the open tension in §1 — "each creature lives once":
- Training is **gestation** — species-level authorship by the artist.
- Mint is **birth** — the creature enters the collector's space and begins its actual life.
- Post-mint adaptation is the creature's life proper — off-chain, local to the collector, unique to that relationship.

Mechanical implications:
- On-chain: seed, environment, base weights, mint-moment provenance — permanent, public.
- Off-chain: lived-in state — local to the collector's runtime.
- Transfer carries lived-in state, not a reset. A creature with six months of life on it is not a freshly-minted creature.
- Persistence: if local state is lost, the creature reverts to birth. Mortality is a feature; optional backup trades off some of the uncopyability claim.

---

## 4. Technical stack for irreplaceability (2026-04-23)

*Extends §3. Engineering answer to "what makes a digital creature copy-resistant like a dog." Short form: **time, owner-specific conditioning, and cryptographically verifiable continuity.** The creature becomes unique because it actually lived somewhere with someone, and that fact is recorded in a way bytes alone can't reproduce.*

### Four mechanisms, stacked

Each closes a different copying route. Together they make irreplaceability defensible; individually none is sufficient.

**1. Owner-conditioned online adaptation (core).** Policy not frozen at mint. A LoRA-style adapter, episodic memory buffer, or preference model updates continuously in the collector's space. Input: gaze, gesture, voice, position, reaction timing from the goggle sensors. Adapter weights drift onto a path defined by this specific person over months. Edition-mates with identical base weights diverge into materially different behavior.
*Closes:* "clone at mint = same creature." The creature at month six is not the creature at mint.

**2. Rate-limited, irreversible adaptation.** Adaptation capped at ~1% weight-delta per week. No reset. No retrain. Current state is a function of *real elapsed time* with an owner. An attacker with stolen weights at time T cannot fast-forward to T+6mo.
*Closes:* "fork later, you've got the aged creature." Time is an input that can't be forged.

**3. Hardware-attested continuity chain.** Every session, full state (base + adapter + memory) is hashed and signed twice — owner's private key plus device secure enclave (Vision Pro's Secure Enclave, Quest attestation APIs, or equivalent). Signatures append to a merkle log. The log is the creature's biography and records presence — actual hours alive with someone. A copy starts a fresh chain, detectably new.
*Closes:* "mine behaves close enough, who cares." The chain is the distinguisher — same principle as painting authentication, with presence recorded alongside.

**4. Biometric-signal dependency.** Adaptation consumes owner-specific signals: voice prints, gaze patterns, gesture rhythms, reaction latencies. Even with full weight access, an attacker can't feed the same training signal going forward. The copy's future evolution is impoverished or drifts in a different direction.
*Closes:* "steal the weights, keep training." The owner's body is the training data.

### What this does not claim

- Bytes can be copied. They can.
- Mint-moment files are unique. They are not.
- Irreplaceability is about the file. It is not — it is about **time + owner + hardware**, reified as a signed chain.

### Transfer semantics

New owner inherits adapter + memory + chain. **Not a reset.** An 18-month-old creature sells as an 18-month-old creature. The chain continues under the new owner's signature. A "fresh creature" is a different edition, not a transferred one. The sale contract must state this plainly.

### The creative surface

Engineering pieces are standard. The creative work is tuning adaptation rates, choosing which signals count, and designing memory retention so the creature feels like it grows — not just drifts. Taste, not engineering. An MVP is: base policy + tiny LoRA adapter + append-only local log + signed-on-session state hash. Everything else is polish.

---

## 5. Repository as published posture (2026-04-23)

*Consequence of §4. Since the pipeline code is not the moat, most of it should be visible. Transparency of method is consistent with `MANIFESTO.md`'s stance and strengthens — not weakens — the uncopyability claim.*

### The split

**Public.** The repository is public. Simulation, multi-agent RL, physics integration, adapter architectures, adaptation-rate tuning, memory designs, rendering pipelines, deployed smart contracts, this document, `MANIFESTO.md`, `ARCHITECTURE.md`, `README.md`. Publishing these costs nothing. The creative surface — which signals count, what retention curves feel alive, how a creature grows versus drifts — cannot be copied by reading source code. Taste is not a file.

**Private.** A narrow band of code and documentation lives outside the public repository:
- **Enclave signing and attestation-log construction** (§4 mechanism 3). Publishing the signing pipeline would hand attackers a map for forging the "presence" signal.
- **Biometric signal ingestion** (§4 mechanism 4). Exact sensor-to-adapter pipelines shouldn't be a recipe.
- **Key material.** Trivially.
- **Commercial strategy** — pricing, projections, drop plans, collector CRM, fabricator relationships. Lives in a local-only `DIRECTION.md` and `SCENE/` directory.

### Why publish the posture

Stating the split publicly strengthens the claim. A reader of §4 and §5 sees: the author publishes the method; the narrow remainder is kept private not to hide the moat but to prevent forgery of it. This is the opposite of security-through-obscurity — a declaration that the moat is real.

---

*Document evolution: §1-§2 (2026-04-19 core), §3 (2026-04-21), §4-§5 (2026-04-23). Companion to `MANIFESTO.md` (short public summary) and local-only `DIRECTION.md` (private strategy). Author: Nicholas C. Park. Living document; evolve forward, never retract.*
