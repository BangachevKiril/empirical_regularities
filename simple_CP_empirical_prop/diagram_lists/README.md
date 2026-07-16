# `diagram_lists`

This directory is package data for the `K = 4` ordinary-CP cumulant-propagation implementation.

## Files

- `k4_default_orbits.json`: the 26 canonical diagram-orbit rows from the supplied table. All 26 are enabled by default. The rows `123 · 12` and `12 · 234` preserve the table annotation `only_unfactorized`, but the annotation is metadata; the generic Cartesian CP evaluator may still evaluate them when requested.
- `k4_default_labelled.json`: the fully expanded 82 labelled vector partitions. This is a deterministic compiled cache of the orbit file.
- `k4_table_yes_only.json`: optional 24-orbit profile that removes the two rows marked `only_unfactorized` in the table.
- `k4_default_r1.json`, ..., `k4_default_r4.json`: the canonical orbit rows split by output cumulant order.
- `schema.json`: JSON schema for the canonical orbit format.
- `loader_reference.py`: reference loader that expands output-vertex orbits without multiplying by the orbit size.
- `validate_catalog.py`: standard-library validation of counts, coefficients, mixedness, connectedness, score, and orbit expansion.
- `k4_summary.json`: counts and SHA-256 hashes.

## Notation

Digits are one-indexed output vertices. Repetition records the multiplicity of that vertex inside a cumulant block. A centered dot separates blocks.

Examples:

```text
112        -> one block (2, 1)
1122       -> one block (2, 2)
12 · 111   -> blocks (1, 1) and (3, 0)
12 · 13 · 23 -> three covariance blocks on a triangle
```

For a block `u = (u_1, ..., u_r)`:

```text
block order = sum(u)
e(u) = 1 iff every u_v is even
score contribution = 1 + e(u) - sum(u)
```

Every default orbit is connected, `[<=2]`-mixed, has maximum block order four, and has total score at least `-3`.

## Coefficient convention

The manifest uses raw Wick/Hermite coefficient vectors. For an unordered vector partition with blocks `u`, the scalar coefficient is

```text
1 / [ product_over_distinct_block_types multiplicity(block)! 
      * product_over_blocks product_over_vertices u_v! ]
```

Apply this coefficient exactly once to every labelled member. Do not divide the Wick vectors by another factorial and do not multiply one representative by `orbit_size`.

## Validation

From the repository root:

```bash
python -m simple_CP_empirical_prop.diagram_lists.validate_catalog
```

Expected counts:

```text
canonical orbits: r=1:3, r=2:10, r=3:9, r=4:4
labelled diagrams: r=1:3, r=2:15, r=3:35, r=4:29
```
