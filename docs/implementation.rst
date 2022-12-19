Implementation
==============

To better implement ``JAX``'s
:doc:`interface <interface>` and
:doc:`design patterns <pattern>`,
we need to break its numerical solutions into multiple components.


Stepper
-------

An ``xaj.core.Stepper`` advances the state of a system of ODEs to a
different state.
It is implemented as a pure function using python function closure,
and is composible with standard ``JAX`` transformations such as
``jit`` and ``vmap``.


Error Estimator
---------------

An ``xaj.core.ErrorEstimator`` estimates the numerical error of a step
and is used in adaptive stepsize control.
It may be based on step-doubling or embedding technique, or can be a
custom function based on, e.g., some conserved quantities in the
system of ODEs.
It is implemented as a pure function using python function closure,
and is composible with standard ``JAX`` transformations such as
``jit`` and ``vmap``.


Stepsize Controller
-------------------

An ``xaj.core.StepController`` adjusts the step size based on the
error estimator and/or custom function.
While it may be more natural to implement it as a stateful object, we
choose not to do it to match better with the ``JAX`` ecosystem.
It is implemented as a pure function using python function closure,
and is composible with standard ``JAX`` transformations such as
``jit`` and ``vmap``.


Stepping Engine
---------------

We use ``jax.lax.while_loop`` to implement the
:ref:`sec_pattern_stepping-engine`, which has the semantics

.. code-block:: python3

   def while_loop(cond, body, state):
       while cond(state):
           state = body(state)
       return state

Because the functions ``cond`` and ``body`` are pure, we need to pass
to them the full state, which includes the current step size, the
current solution of the ODEs, the shared states across steps, etc.
The body function can be logically implement with the following code.

.. code-block:: python3

   def body(state):
       _, (t,x), (h,k), (i,r) = state

       E, T, X, K = step(h, t, x, k)
       H, retry   = ctrl(h, t, x, E, T, X)

       if not retry:
           return _, (T,X), (H,k), (i+1,0)   # continue
       else:
           return _, (T,X), (H,k), (i+1,r+1) # retry
