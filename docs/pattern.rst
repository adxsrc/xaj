Design Patterns
===============

While auto vectorizing and parallel maps ``vmap`` and ``pmap`` are
straightforward for pure functions because they are embarrassingly
(i.e., naturally) parallelizable, numerical integrations are not.
Depending on initial conditions, different realizations of ODE
solution may take different stepsize.
To maintain composability with ``JAX`` transformations, we need to
limit the scope of ``XAJ``, which leads to interesting design
patterns.
