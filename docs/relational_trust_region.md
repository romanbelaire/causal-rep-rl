# Relational trust region (implementation companion)

See also [`representation_space_trust_region.md`](../representation_space_trust_region.md).

Implemented in `src/metrics/relational_trust_region.py`. Default mode is
`observe`. Hard-gate restores encoder parameters and the representation
optimizer state exactly; a mismatch after restore is a `RuntimeError`.

The frozen set \(B\) is logged (shared with the PL probe or a distinct
snapshot). \(\tau_d\) excludes near-zero old distances. Policy KL is not this
certificate.
