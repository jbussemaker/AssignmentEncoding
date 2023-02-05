# Test Problems

Notation:
- Source/target nodes: `n` @ `m` [(rep)] (`n` nodes where each node accepts `m` connections), where `m` can be:
  - A fixed number: `m` (e.g. `0`, `1`, `5`)
  - A set of fixed numbers: `m,o,p` (e.g. `0,1`, `0,1,3`)
  - A minimum amount: `m+` (e.g. `0+`, `1+`)
  - A range of numbers (inclusive): `m..o` (e.g. `0..2`, `1..5`)
  - (rep) specifies whether repeated connections to/from that node are allowed, by default they are not allowed
- Excluded edges: `(i_src,i_tgt),...` (e.g. `(0,1),(4,3)`)
- Existence patterns: any repetition of a combination of source/target node definition

## Analytical Problems

Analytical problems are modeled after architecture patterns defined in
*Selva, Cameron and Crawley, "Patterns in System Architecture Decisions", 2016, DOI: 10.1002/sys.21370*.

Each problem is defined in terms of an assignment-encoding problem through a combination of node definitions,
excluded edges and existence patterns. The analytical problem has two objectives and no constraints.
The objectives are evaluated as follows:
- Define a matrix of coefficients of size (`n_src x n_tgt x 2`)
  - Elements at `i, :, 0` are defined by: `sin(linspace(0, 3*pi, n_tgt) + .25*pi*i) - 1`
  - Elements at `i, :, 1` are defined by: `cos(linspace(-pi, 1.5*pi, n_tgt) - .5*pi*i) + 1`
- Objectives are then calculated as follows:
  - `f[i_obj] = sum(coeff[i_src, i_tgt, i_obj] for each connected i_src,i_tgt)`
  - Optionally, the second objective can be inverted (`invert_f2`): `f[1] = -f[1]`
  - Optionally, the objectives can be skewed (`skew`): `f[1] -= .25*f[0]; f[0] -= .25*f[1]`

Following problems are defined:

| Pattern                            | Parameters      | Source nodes       | Target nodes        | Excluded edges    | Objective         |
|------------------------------------|-----------------|--------------------|---------------------|-------------------|-------------------|
| Combining                          | `n_tgt`         | `1 @ 1`            | `n_tgt @ 0,1`       |                   |                   |
| Assigning                          | `n_src, n_tgt`  | `n_src @ 0+`       | `n_tgt @ 0+`        |                   |                   |
| Assigning (injective)              | `n_src, n_tgt`  | `n_src @ 0+`       | `n_tgt @ 0,1`       |                   |                   |
| Assigning (surjective)             | `n_src, n_tgt`  | `n_src @ 0+`       | `n_tgt @ 1+`        |                   |                   |
| Assigning (bijective)              | `n_src, n_tgt`  | `n_src @ 0+`       | `n_tgt @ 1`         |                   | `invert_f2`       |
| Assigning (repeatable)             | `n_src, n_tgt`  | `n_src @ 0+ (rep)` | `n_tgt @ 0+  (rep)` |                   | `invert_f2, skew` |
| Assigning (repeatable, surjective) | `n_src, n_tgt`  | `n_src @ 0+ (rep)` | `n_tgt @ 1+  (rep)` |                   | `invert_f2, skew` |
| Partitioning                       | `n_src, n_tgt`  | `n_src @ 0+`       | `n_tgt @ 1`         |                   | `invert_f2`       |
| Partitioning (covering)            | `n_src, n_tgt`  | `n_src @ 0+`       | `n_tgt @ 1+`        |                   |                   |
| Downselecting                      | `n_tgt`         | `1 @ 0+`           | `n_tgt @ 0,1`       |                   |                   |
| Connecting                         | `n`             | `n @ 0+`           | `n @ 0+`            | `(i,j) if i >= j` |                   |
| Connecting (directed)              | `n`             | `n @ 0+`           | `n @ 0+`            | `(i,j) if i == j` |                   |
| Permuting                          | `n`             | `n @ 1`            | `n @ 1`             |                   |                   |
| Combinations                       | `n_take, n_tgt` | `1 @ n_take`       | `n_tgt @ 0,1`       |                   |                   |
| Combinations w/ replacement        | `n_take, n_tgt` | `1 @ n_take (rep)` | `n_tgt @ 0+ (rep)`  |                   |                   |

Compared to the patterns defined by Selva et al., the following extensions are made:
- For the assigning pattern, the "function" aspect (i.e. each source node has 1 connection) of injective, surjective
  and bijective problems are dropped, in order to increase the problem size
  and not be constrained by `n_src` relative to `n_tgt`
- Repeatable versions of the assigning versions are added in order to increase the number of test problems
  - Note that any repeatable node that has an upper bound of 0 or 1 in effect is non-repeatable
- Two additional patterns are added:
  - Combinations: take combinations of length `n` from a set, e.g. 2 from ABC: AB, AC, BC
    - This pattern is the same as a combination pattern if `n_take = 1`
    - It can also be seen as a downselecting problem with a fixed size of the selected set
  - Combinations with replacement: same as the above however values can be taken again: e.g. 2 from ABC: AA, AB, AC, BB, BC, CC
  - These two patterns can be seen as versions of the combination, permuting and assigning patterns,
    where the order of connections is not relevant (i.e. AB has the same fitness as BA)

As can be seen in the table above, there are normally only a few values used for the number of node connections:

| src↓, tgt→ | 1         | 0,1       | n            | 0+                                           | 1+                                                | 0+ (rep)               | 1+ (rep)                           |
|------------|-----------|-----------|--------------|----------------------------------------------|---------------------------------------------------|------------------------|------------------------------------|
| 1          | Permuting | Combining | (1)          | Assigning (bijective) == partitioning        | (2)                                               | (3)                    | (3)                                |
| 0,1        |           | (4)       | Combinations | Assigning (injective), downselecting         | (2)                                               | (3)                    | (3)                                |
| n          |           |           |              | Combinations with replacement                |                                                   | (3)                    | (3)                                |
| 0+         |           |           |              | Assigning, connecting, connecting (directed) | Assigning (surjective) == partitioning (covering) | (3)                    | (3)                                |
| 1+         |           |           |              |                                              |                                                   | (3)                    | (3)                                |
| 0+ (rep)   |           |           |              |                                              |                                                   | Assigning (repeatable) | Assigning (repeatable, surjective) |
| 1+ (rep)   |           |           |              |                                              |                                                   |                        |                                    |

(1) only possible if `n_src = n*n_tgt`, in which case it is effectively `n` independent permuting patterns.
(2) effectively permuting if `n_tgt = n_src`, otherwise only relevant if `n_src > n_tgt`.
(3) effectively their non-repeatable versions.
(4) combining with the possibility of not choosing an option.

### Combinations of Analytical Problems

### Multiple Analytical Problems in One

## GN&C Problem
