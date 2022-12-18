Interface Design
================

``JAX`` is an extensible system for transforming numerical functions.
The basic transformations are: ``grad``, ``jit``, ``vmap``, and
``pmap``.
To make ``XAJ`` interpolate well with ``JAX``, we design ``odeint`` to
behave the same way as ``grad`` and be composable with other ``JAX``
transformations.
This is challenging because we are mixing autodiff, which works well
with pure functions, with numerical integration, which requires some
internal states.
