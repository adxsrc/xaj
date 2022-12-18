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


Solution Callable
-----------------

Consider the ordinary differential equations (ODEs)

.. math::

   \frac{d\mathbf{x}}{dt} = f(t, \mathbf{x})

and use a python **callable** ``x(t)`` to represent its (numerical)
solution.
We want to design ``XAJ`` in a way that it works naturally with
``JAX``'s automatic differentiation interface.

1. We would like ``jax.jacfwd(x)`` to return the right hand side
   callable ``f(t, x)``.

2. We would like evaluating ``x(t)`` at different ``t`` as simple as
   function calls, i.e., ``x(0.0)``, ``x(1.0)``, ``x(2.0)``, ...

In addition, motivated by general relativistic ray tracing (GRRT)
applications, where it is common to integrate the geodesics backward
in the affine parameter, we also want the following.

3. ``XAJ`` should support integrating in both positive and negative
   ``t`` directions.


Composability
-------------

To make ``XAJ`` work with the rest of the ``JAX`` ecosystem, we
require the following.

4. ``odeint`` should have built-in pytree support just like ``grad``.
   See, e.g., the ``JAX``
   `MLP example <https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html#example-ml-model-parameters>`_.

5. The numerical solution ``x(t)`` should support other ``JAX``
   transformations such as ``jit`` and ``vmap``.

Because of the Single Instruction, Multiple Data (SIMD) architectures
of GPUs and TPUs, we don't expect performance gain by evaluating
different systems of ODEs at the same time.
Hence, ``vmap`` over different ``f(t, x)`` is not supported in
``XAJ``, just like ``JAX`` does not support vmapping to multiple
functions ``vmap(grad)([f1, f2, f3])``.


Call Signature
--------------

How should we design the call signature of ``odeint``, its invert
function, and the numerical solution?
Because ODEs are uniquely specified only when the initial conditions
are given, ``XAJ``'s integration interface is more complicated than
``JAX``'s derivative interface.

Let's consider a specific ODE

.. math::

   \frac{dx}{dt} = f(t, x) = x + t.

It has analytical solution :math:`x(t) = c e^t - t - 1`.
Using the initial condition :math:`x(t_0) = x_0`, we can rewrite the
analytical solution as

.. math::

   x(t; t_0, x_0) = (x_0 + t_0 + 1)e^{t - t_0} - t - 1.

It is straightforward to verify :math:`\partial_t x(t; t_0, x_0) = x -
t = f(t, x)`, where :math:`t_0` and :math:`x_0` can be seen as
parameters of the solution :math:`x`.
Given that ``jacfwd`` (or ``jacrev`` or ``grad``) by default takes the
derivatives with respect to only the first argument, we can use the
convention

.. code-block::

   odeint: f(t, x, aux) -> x(t, t0, x0, aux)
   jacfwd: x(t, t0, x0, aux) -> f(t, x, aux)

where ``t`` and ``t0`` are scalars and ``f``, ``x``, and ``x0`` may be
arrays.
In the special case that ``f`` is independent of ``t``, we have

.. code-block::

   odeint: f(x, aux) -> x(t-t0, x0, aux)
   jacfwd: x(t-t0, x0, aux) -> f(x, aux)

We may see ``odeint`` as a functional (or high-order function) that
adds a new independent variable ``t``, while ``jacfwd`` is its invert
and removes the independent variable.
