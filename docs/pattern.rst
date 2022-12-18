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


Stepping Engine
---------------

Consider a scenario that we evolve ``n`` planets around the sun.
Because their different orbital time scales and ellipticities, after,
e.g., 4 steps, their solutions would advance to

::

   p0 |---->|-->|--->|---->|
   p1 |-->|-->|-->|-->|
   p2 |->|->|>|>|
   p3 |>|>|>|>|
      ^t0     ^t1     ^t2

where ``t0`` is the time that we specify the initial conditions, and
``t1`` is the minimal final time for all the planets.

Suppose we want to evolve all planets up to ``t2``, the most
computationally efficient way is to change the batch size and
integrate *only* ``p2`` and ``p3`` up to ``t2``.
However, we *do not* support this in ``XAJ``'s core API.
This is because such a smart rebatching is inconsistent with the
assumptions in ``JAX`` and would break composability.

Nevertheless, this does not mean we cannot do better than the above
chart.
For ``p0``, the changing stepsize suggests that there are multiple
trial in the integration.
Hence, a better presentation this job is probably

::

   p0 |---->|--->|
            |-->|--->|
   p1 |-->|-->|-->|-->|
   p2 |->|->|->|
            |>|>|
   p3 |>|>|>|>|
      ^t0     ^t1     ^t2

In this new scenario, while each particle went through 4 steps, for
``p0`` and ``p2``, only 3 steps satisfy the tolerance and contribute
to the solution.
Given the SIMD architecture in GPU and TPU, a naive nested loop
implementation would evolve the above problem in the following steps

::

   p0 |S|T:S|S:D|E|
   p1 |S|S:D|S:D|S|
   p2 |S|S:D|T:S|E|
   p2 |S|S:D|S:D|S|

where ``S`` stands for (successful) stepping, ``T`` stands for trial,
``D`` stands for dropped calculation, ``E`` stands for extra, and
``:`` seperates the steps in an inner loop.
Although the 2 trail ``T`` calculations in the above chart are
unavoidable, there are 6 dropped ``D`` and 2 extra ``E`` calculations
that do not help to achieve the required numerical solution.

One interesting observation here is that the trail and the actual
successful calculations are computationally identical.
Therefore, if we integrate the step controller with the driver, it is
possible to fuse the two types of calculations.
This will result

::

   p0 |S|T:S|S|
   p1 |S|S|S|S|
   p2 |S|S|T:S|
   p2 |S|S|S|S|

which saves 33% of the computation from the naive implementation; or
in other words avoid 50% of wasted computation from the optimal
implementation.


Dense Output
------------

In ``XAJ``'s :doc:`interface signatures <interface>`, ``x(t, t0, x0,
aux)`` can in principle be a pure function without side effect (and
internal states).
However, many adaptive integrators support dense output, where the
numerical solutions are saved as piece-wise polynomials for later
interpolation.
This is particularly efficient if we want to obtain numerical
solutions on a large number of sampling points.

How should we design ``XAJ`` to take advantage of dense output?

A natural choice is to cache the dense outputs with ``x(t, t0, x0,
aux)``, and reuse the cache whenever possible.
Given that the numerical solutions in general depend on ``t0``,
``x0``, and ``aux``, we can use the tuple ``(t0, x0, aux)`` as the
cache index.
The cache should also be a
`pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_
in order to be composable with other ``JAX`` functions.
