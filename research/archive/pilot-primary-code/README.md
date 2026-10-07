# Exact primary pilot source snapshot

These source files match the hashes recorded in both primary empirical runs. They were retained before the later exact-p-value underflow representation fix. They contain software and the protocol only; no individual data or fitted models.

The active implementation is under the repository root. Primary fitting/preprocessing is unchanged; the active runner represents extremely small exact probabilities with log10 values instead of numerical zero. This archive is for provenance, not a recommendation to report p=0. See research/VALIDATION.md and the aggregate manifests.
