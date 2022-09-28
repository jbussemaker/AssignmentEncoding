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
2. No upper limit is placed on matrix values if repeated connections are allowed between nodes.
3. Matrix elements can be constrained to 0 to represent explicitly forbidden connections between node pairs.

Given these rules and the definitions of the source and target nodes, all possible assignment patterns can be found and
represented in the **Aggregate Assignment Matrix (AAM)** of size `n_src` x `n_tgt` x `n_pat` (number of valid
assignment patterns).

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
1. Maximization of information content, representing the amount of information that can be gained from the design
   variables; measured by the Leave-one-out-Cross-Validation (LOOCV) accuracy of some surrogate model using the design
   variables as input and some objective function as output.
2. Maximization of information content, represented by the amount of design variables: `n_dv`
3. Minimization of the imputation ratio: the ratio of the combinatorial design space size `prod(n_i)` and `n_pat`

Then to test them for real optimization performance, the following tests should be performed:
1. Compare different imputation algorithms: combined with the encoding scheme with the highest imputation ratio, solve
   with a genetic algorithm
2. Compare different encoding schemes, combined with the best imputation algorithm:
   1. Genetic algorithm to test the impact of the imputation ratio
   2. Surrogate-based optimization algorithm to test the impact of both imputation ratio and information content

A suitable test problem can be the GN&C problem from Crawley et al, which can be dynamically tuned to be more or less
difficult by varying the number of possible sensors, computers and actuators. Optimization performance can be measured
using the `Delta HV` and spread metrics (see Bussemaker 2021), for some fixed computational budget.
