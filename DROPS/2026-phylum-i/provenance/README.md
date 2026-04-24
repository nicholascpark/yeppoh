# Provenance

Drop-specific provenance artifacts. Reusable contract source lives at the
top-level `PROVENANCE/` directory (not yet created — add when first contract
is drafted).

Planned contents:
- `deploy.md` — deployer address, chain (Base), deployment block, verifier link
- `mints.jsonl` — one line per mint (tokenId, recipient, tx hash, timestamp)
- `ipfs.jsonl` — IPFS CIDs for metadata, media, and checkpoint hashes
- `contract_snapshot.sol` — frozen copy of the deployed contract source at
  launch time. The top-level contract may evolve for Phylum II+; this drop's
  deployed bytecode is forever this exact version, so the source is pinned
  here as historical record.
