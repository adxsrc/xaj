[![Lint & test](https://github.com/adxsrc/xaj/actions/workflows/python-package.yml/badge.svg)](https://github.com/adxsrc/xaj/actions/workflows/python-package.yml)
[![PyPI](https://github.com/adxsrc/xaj/actions/workflows/python-publish.yml/badge.svg)](https://pypi.org/project/xaj/)

> [!WARNING]
>
>   This version of `XAJ` works only with `jax` and `jaxlib` 0.3.
>   But `jaxlib` 0.3 is not available on PyPI.
>   To install this version of `XAJ`, please first install `jaxlib`
>   with:
>
>       pip install jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html


# `XAJ`

Ordinary differential equation (ODE) integrator compatible with
[Google's JAX](https://github.com/google/jax).

`XAJ` implements the Runge-Kutta Dormand-Prince 4/5 pair (DP5) with
adaptive step control and dense output according to the
[Numerical Recipes](http://numerical.recipes/).
It provides a fast and efficient way to solve non-stiff ODEs.
The programming interface is designed so it feels similar to the
derivative functions in `JAX`.

Specifically, `XAJ` provides a single function `odeint()`.
Applying it to another function `rhs()` and the initial conditions
returns the numerical solution as a callable `ns`, which interpolates
the dense output.

    from xaj import odeint
    from jax import numpy as np

    rhs = lambda x, y: y
    x0  = 0
    y0  = 1

    ns  = odeint(rhs, x0, y0, 1)

    x   = np.linspace(0, 5)
    y   = ns(x)

The numerical integration happens in a "lazy" way, which is triggered
by the extreme values of the argument of `ns`.
Alternatively, it is possible to obtain the numerical solutions at the
full steps by

    xs = ns.xs
    ys = ns.ys

Demos on how to use `XAJ` can be found in the [`demos`](demos/)
directory.
