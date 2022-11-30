# AssignmentEncoding

Experiments for finding how to best encode assignment/permutation decisions in architecture optimization problems.

## Installation

Create a Python 3.7+ environment and install dependencies using pip: `pip install -r requirements.txt`

## Background

Architecture optimization is a field of research that deals with trying to find the best system architecture to fulfill
some design objectives by automatically generating and evaluating architectures. This can be done by first *modeling*
architectural decisions, then *encode* these decisions as optimizer design variables, and then using *optimization
algorithms* to find the best combination of design variables to minimize some objective(s) while satisfying constraints.

Different types of architectural decisions might be present in such design problems. Here, we deal with the component
connection decision, answering the question of how to best connect **between sources and targets**. In aircraft design,
one such a design can for example be the question of how to connect the different hydraulic systems (typically there
are 3 hydraulic systems on transport aircraft) to the different consumers (e.g. control surface actuators).

These types of assignment (or permutation) decisions are characterized as follows:
- There can be several *source* (`n_src`) and *target* (`n_tgt`) nodes
- Connections are made *from source to target*
- Each node specifies how many connections they accept/provide, specified as:
  - A list of specific connection numbers, e.g. `[1]`, `[0, 1, 2]`, `[1, 3, 4]`
  - A range of connection numbers, e.g. `1..4`, `0..1`
  - An unbounded range of connections: e.g. `0..*`, `1..*`
- Each node can define whether repeated connections to the same source/target are allowed
- Connections between specific nodes can be prohibited
- Some nodes might not always be present for all architectures

For more background, see section III.B.4 (Component Connection, page 6) of the paper linked below.

The encoding schemes should be usable by existing optimization algorithms in general, and usable by surrogate-based
optimization algorithms in particular. The latter pose a challenge, because they are designed to speed up the
optimization by constructing a mathematical model of the design space and using that to predict new interesting design
points. This means that the design variables have to be representative enough of the objective and constraint function
responses, otherwise the mathematical model cannot be used to make any such predictions.

For more background on architecture optimization and modeling architecture design spaces, see:
[System Architecture Design Space Exploration: An Approach to Modeling and Optimization](https://www.zenodo.org/record/4672182)

## Research Motivation and Method

The motivation for this research comes from the question on **how to encode these assignment decisions as discrete
design variables** understandable by existing optimization algorithms. The encoding step consists of two parts:
1. Find all the possible connection patterns between the source and target nodes
2. Encoding the possible connection patterns as discrete design variables

### Finding Possible Connection Patterns

The types of assignment decisions in this work are similar to the assignment pattern in
[Selva et al. (2017), "Patterns in System Architecture Decisions"](https://onlinelibrary.wiley.com/doi/10.1002/sys.21370).
Here, assignment decisions are represented using a matrix with `n_src` rows and `n_tgt` columns. Each element in the
matrix represents whether a connection (1) is made or not (0). This representation is extended with the following rules
to match with the above decision characteristics:
1. Constraints are placed on the *sum of values* in rows and columns, corresponding to the number of connections 
   accepted by source and target nodes, respectively. For example, if source node i=1 only can establish 1 connection to
   any of the target nodes, the sum of elements in the first row must be 1.
   This can be used to model connection slot constraint, stemming from for example physical or capacity constraints.
2. No upper limit is placed on matrix values if repeated connections are allowed between nodes.
   Repeated connections can be directly relevant in some architecture problems, for example when multiple connections
   between components (e.g. electrical connections) would increase robustness (at some cost) of the system.
   Repeated connections, however, can also be useful as a way to implement sequence-irrelevant connection problems:
   problems where the number of connections is relevant, but not the sequence that these connections are made in.
   This can be used to turn a permutation problem into a combinations or combinations-with-replacement problem.
3. Matrix elements can be constrained to 0 to represent explicitly forbidden connections between node pairs.
   This can be used to model network connections, where any connection between nodes is possible (i.e. network nodes are
   represented by both src and tgt nodes at the same time), except from/to itself.

This representation leads to a fundamental difference between node-wise connections and between-node connections:
- Node-wise connections are used to represent assignment patterns where only the amount of connections is relevant
- Whereas between-node connections represent patterns where both the amount, and the sequence of connections is relevant 

Given these rules and the definitions of the source and target nodes, all possible assignment patterns can be found and
represented in the **Aggregate Assignment Matrix (AAM)** of size
`n_pat` (number of valid assignment patterns) x `n_src` x `n_tgt`.

The algorithm for finding the *AAM* is not part of this research.

### Encoding As Discrete Design Variables

This is the main focus on this research. The goal is to find the best encoding scheme that lies between the following
two extremes:
1. Defining one design variable with `n_pat` options, each selecting the corresponding permutation: this would hide
   any information related to sensitivity or coupling of decisions, thereby reducing designer/optimizer insight.
2. Defining one design variable for each matrix element, each with the number of options as there are different values
   in the *AAM*: this would lead to many combinations of design variable values that would not correspond to any
   of the patterns in the matrix.

In general, any encoding scheme should consist of the following elements:
1. Some way of defining discrete design variables by (recursively) separating (a subset of) an AAM and mapping the sets
   to DV values
2. Some way of correcting incorrect design vectors to a feasible assignment pattern (i.e. *imputation*)

### Comparing Encoding Scheme Performance

The different encoding schemes will be compared based on the following metrics:
1. Minimization of information error, representing how accurate information related to the design variables can be
   modeled; measured by the Leave-one-out-Cross-Validation (LOOCV) accuracy of some surrogate model using the design
   variables as input and some objective function as output
2. Maximization of information index, representing how much design variables are used compared to a set of pure binary
   design variables (`n_opts = 2`): `inf_idx = (n_dv-1)/(log2(cumprod(n_opt_i))-1)`
3. Minimization of the imputation ratio: the ratio of the combinatorial design space size `prod(n_i)` and `n_pat`
  - A derived metric can be the "Relative imputation ratio", which would apply to a set of encoders and simply compares
    relative design space sizes `prod(n_i)` (i.e. `n_pat` is divided out), which would prevent the need for counting
    the total number of matrices

Note that the latter two can be calculated directly from the problem formulation itself, whereas the first one needs
objective/constraint evaluation and repeated surrogate model training.

Then to test them for real optimization performance, the following tests should be performed:
1. Compare different imputation algorithms: combined with the encoding scheme with the highest imputation ratio, solve
   with a genetic algorithm
2. Compare different encoding schemes, combined with the best imputation algorithm:
   1. Genetic algorithm to test the impact of the imputation ratio
   2. Surrogate-based optimization (SBO) algorithm to test the impact of both imputation ratio and information error

A suitable test problem can be the GN&C problem from Crawley et al, which can be dynamically tuned to be more or less
difficult by varying the number of possible sensors, computers and actuators. Optimization performance can be measured
using the `Delta HV` metric, for some fixed computational budget. The spread metric (Bussemaker2021) will not be used,
as the Pareto front is not necessarily continuous.

## Hypotheses

1. For encoders with high imputation ratios, there exists an imputation algorithm that has the best convergence rate,
   which is robust across analytical problems (both for eager and lazy encoders)
2. Information error and imputation ratio are correlated with convergence rate
    1. Information error is not correlated with GA convergence rate
    2. Information error is correlated with SBO convergence rate: lower error --> faster convergence
    3. Imputation ratio is correlated with GA convergence rate: lower ratio --> faster convergence
    4. Imputation ratio is correlated with SBO convergence rate: lower ratio --> faster convergence
    5. Imputation ratio and information error should be minimized
    6. Information index is a proxy for (i.e. correlated with) information error
3. Lazy encoders can be applied to encode very large problems time-efficiently
4. For each analytical architecture problem, a (set of) best encoders exist that is consistent over all problem sizes
   (separately for eager and lazy encoders)
5. It is possible to define rules for encoder selection, based on the node configurations
    1. This applies to the analytical problems (incl. repetition)
    2. This also applies to the GN&C problem (only combinations, with types, and with types + amounts)

### Observations

#### 1. Imputation Algorithm tests
01_imputation

- Both eager and lazy imputation algorithms exist that are effective
- For high imputation ratios (~10):
  - All imputation algorithms are similarly effective; most effective are:
  - Eager: Auto Mod
  - Lazy: Delta Imp
- For very high imputation ratios (~100):
  - First imputer and constraint violation imputers are not effective; they either only map to 1 (First Imp) or
    0 (CV Imp) valid point, meaning that during a random search for new points (e.g. DOE), there is a low chance of
    finding a diverse set of new points (this is also seen from the starting nr of evaluations of the First Imputer)
  - The Auto Mod Reverse (Eager) imputation algorithm is also not very effective, as it starts searching from the end of
    the design vector, where there is a higher chance of not hitting a valid design
  - Eager: both Closest Imp algorithms are similarly effective, with Auto Mod slightly more effective
  - Lazy: Delta Imp and both Closest Imp algorithms are similarly effective
    - The Delta Imputer is faster and needs less memory
    - For extremely high imputation ratios, lazy imputers might fail to find a valid design point, in which case they
      will behave as the constraint violation imputer

Conclusions:
- For eager encoders, use the Auto Mod imputer
- For lazy encoders, use the Delta imputer

#### 2. Metric Correlations
02_metric_correlation

- Information error is very difficult to determine; normally a very high standard deviation in estimated values is seen,
  and if more samples are taken metric values usually move closer together (i.e. there is less of a trend)
  - Information error and information index are therefore not correlated
- Information index and imputation ratio are independent parameters
- A better metric value (lower information error or imp ratio, higher information index), all lead to faster converging
  optimizers, both for NSGA2 and SBO
- Only imputation ratio influences performance after the initial DOE; this indicates that a lower imputation ratio
  greatly influences how diverse the initial DOE is

Conclusions:
- Information error does not need to be used to predict encoder performance
- Imputation ratio should be minimized and information index should be maximized
- It might be more important to minimize imputation ratio

#### 3. Encoding and Decoding Times
03_encoding_time

- Decoding time for lazy encoders depends both on the time to generate matrices on-demand (if needed)
  and to validate the generated matrices
- For lazy encoders where no matrices are generated (so only validation time counts):
  - Lazy encoding is between 50x faster than eager encoding (~1000x slower if the matrix is not cached)
  - Lazy decoding is between 1.2x-1.5x slower than eager decoding
- For lazy encoders where matrices are generated on-demand (as well as validated):
  - Lazy encoding is about 100x faster as eager encoding (~200x slower if the matrix is not cached)
  - Lazy decoding is about 2x slower than eager decoding
- Encoding time increases linearly with the amount of matrices for eager encoders
- Encoding time does not increase (or maximally with the log of amount of matrices) for lazy encoders
- Sampling time increases sub-linearly (due to some constant overhead) with the amount of samples in all cases
- Sampling time increases with the log of amount of matrices
- It should be noted that the amount of matrices might increase extremely fast for small increases in problem size,
  whereas this is not necessarily the case for the number of samples

Conclusions:
- Aggregate matrix caching is very effective at reducing eager encoding times
- Lazy encoders are very effective at keeping encoding times low, at a relatively low increase in sampling time
- Cut-off eager encoders after some fixed encoding time has been exceeded when looking for the best encoder

#### 4. Metric Consistency
04_metric_consistency

- GA algorithm are most effective for low imputation ratios or high information indices
- For eager Amount First groupers, the (Rel) Flat/Coord Idx groupers are very inefficient for large nr of matrices
  - Inefficient in time, memory usage, and imputation ratio
  - For some problems their imputation is low, but then other encoders with low imputation ratio are also available
- A time limit must be set for eager encoders to prevent runaway encoding time

Suggested selection:
- First try lazy encoders: try to select an encoder with as low imp ratio as possible and high inf index as possible
- If none are available, also try eager encoders
- Set a time limit on the encoding process (for each encoder)

#### 5. Selector Algorithm
05_selector_algo

- A selector algorithm is developed that selects the best encoder given some combination of nodes
  - The selection is cached, however the time to initially do the selection is of interest
  - Before doing the selection, the actual matrices are generated and cached
    - It turns out that matrix generation does not take a long time compared to some encoders
    - Main time cost comes from design variable grouping (`GroupedEncoder.group_by_values`)
- The selector algorithm consists of the following steps:
  - Pre-generate the matrix (and cache it)
  - Encode lazy encoders; for each calculate the imputation ratio (using pre-generated matrix) and information index
    - Additionally encode eager encoders if `n_mat <= n_mat_eager_max`
    - Stop encoding (and skip) if encoding time exceeds `encoding_time_limit`
  - Select the best lazy encoder according to the division-scoring algorithm (*not forced*)
  - If none is selected, encode eager encoders (calculate same metrics as for lazy encoders)
  - Select the best encoder according to the division-scoring algorithm (*forced*)
  - Regardless of how good the finally selected encoder is, this algorithm will always have at least one result:
    - The lazy direct matrix encoder (high information index, but also high imputation ratio) will always be included,
      as its encoding time is close to instant (no need for matrix generation and no need for DV grouping)
    - If eager encoders are also included, the one var encoder will also always be included, as except for the matrix
      generation (which is done anyway), its encoding time is close to instant too
- The division-scoring algorithm works as follows:
  - Given a set of encoders and associated imputation ratio and information index scores, divide them into priorities
  - Define multiple bands of imputation ratios: ==1, 1-3, 3-10, 10-30, 30-100, 100+
  - Define multiple bands of information indices: 0-.3, .3-.6, .6-1
  - This division is done according to the following division (see also the `selector_areas` plot):
    - For the first two imputation ratio bands, priorities are selected in descending information index order
    - After that, priority is selected in descending information index order per imputation ratio band
  - If not *forced*, only consider the first 4 priorities (i.e. imp ratio <= 3, inf inx >= .3)
  - The best encoder is selected from the division with the highest priority that contains 1 or more encoders:
    - Determine minimum imputation ratio within the division
    - From the encoders that have this minimum imputation ratio, select the one with the highest information index
- Total selection time (not cached) is greatly influenced by `n_mat_eager_max` and `encoding_time_limit`
  - For lower time values (`n_mat_eager_max ~ 1000`, `encoding_time_limit ~ .5`), many encoders are skipped
  - However, increasing the allowed time does not result in better encoding scores and optimization results
  - This could be because the encoders that take most time are the ones that use design variable grouping, which is
    normally used in encoders with very high imputation ratios, and are therefore usually not interesting anyway
- Selected encoders all reside either on the imputation ratio = 1 or information index = 1 lines
  - Many even lie at the intersection (which is the best encoding score possible)
  - Maximum imputation ratio was about 30, even for relatively large problems
  - Encoders with information index = 1 all reach very good optimization results
  - Encoders with imputation ratio = 1 reach very good optimization results if information index > approx .5
    - Below that, results are still acceptable

Conclusion:
- The selector algorithm is able to select the best encoder for the problem at hand, in a low amount of initial time
  - The result is cached, however, so subsequent encoding requests are close to instant
- Setting `n_mat_eager_max = 1000` and `encoding_time_limit = .5` result in good results, within max 8 sec
