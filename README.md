# AssignmentEncoding

Experiments for finding how to best encode assignment/permutation decisions in architecture optimization problems.

## Installation

Create a Python 3.7+ environment and install using pip: `pip install -e .`

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

## Usage

Defining an assignment pattern:
```python
from assign_enc.matrix import *

# Define source and target nodes
src = [
    Node([1]),  # A node that always needs 1 connection
    Node([0, 1]),  # Either 0 or 1 connections
    Node(min_conn=0),  # 0 or more connections
    Node(min_conn=1),  # 1 or more connections
    Node(min_conn=0, max_conn=3),  # Between 0 and 3 connections
]
tgt = [
    Node([1], repeated_allowed=False),  # 1 connection, no repetitions between same src-tgt pairs are allowed
    Node([0, 1], repeated_allowed=False),
    Node(min_conn=0, repeated_allowed=False),
]

# Combine in a settings object
settings = MatrixGenSettings(
    src=src,
    tgt=tgt,
)

# Optionally exclude specific src-tgt pairs
settings_ex = MatrixGenSettings(
    src=src, tgt=tgt,
    excluded=[(0, 1)],  # src, tgt indices; also possible to supply Node objects
)

# Define different node-existence scenarios
all_exists = NodeExistence()
node_existence = NodeExistence(src_exists=[True, True, False, False, False])
n_conn_existence = NodeExistence(tgt_n_conn_override={
  0: [0, 1],  # tgt 0 now can have 0 or 1 connections
})
settings_exist = MatrixGenSettings(
  src=src, tgt=tgt,
  existence=NodeExistencePatterns([all_exists, node_existence, n_conn_existence]),
)
```

Generating all valid connection matrices:
```python
from assign_enc.matrix import *

matrix_gen = AggregateAssignmentMatrixGenerator(settings)

# Generate all matrices, returned for each existence scenario
all_matrices_by_existence = matrix_gen.get_agg_matrix()

# Get the matrices for the first existence scenario (by default this is the "all exist" scenario)
matrices = list(all_matrices_by_existence.values())[0]

# Or loop over existence
for existence, matrices in all_matrices_by_existence.items():
    print(f'{existence!r}: {matrices.shape}')
```

Automatically get the best encoding algorithm:
```python
from assign_enc.selector import EncoderSelector

selector = EncoderSelector(settings)

# Get best encoding algorithm (wrapped in an AssignmentManager)
manager = selector.get_best_assignment_manager()

# Get design variables
design_vars = manager.design_vars
n_des_vars = len(design_vars)
des_var_n_opts = [dv.n_opts for dv in design_vars]

# Get matrix for a given design vector
x = manager.get_random_design_vector()
x_imputed, x_is_active, matrix = manager.get_matrix(x)
x_imputed, x_is_active, matrix = manager.get_matrix(x, existence=existence)  # For a specific existence scenario

# Only correct the vector without generating the matrix
x_imputed, x_is_active = manager.correct_vector(x)

# Or get as connections list
x_imputed, x_is_active, connections = manager.get_conns(x)  # As a list of (Node, Node) pairs
x_imputed, x_is_active, connection_indices = manager.get_conn_idx(x)  # As a list of (node_idx, node_idx) pairs

# Generate all possible design vector (may not be possible due to memory limits)
x_all_by_existence = manager.get_all_design_vectors()
```
