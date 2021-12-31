# Copyright (C) 2020,2021 Chi-kwan Chan
# Copyright (C) 2020,2021 Steward Observatory
#
# This file is part of XAJ.
#
# XAJ is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# XAJ is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with XAJ.  If not, see <http://www.gnu.org/licenses/>.


from jax import numpy as np


def Init(rhs):
    """Turn the RHS of a system of ODEs to a DP5 initializer

    This function takes the right-hand-side (RHS) of a system of
    ordinary differential equations (ODEs) into an initializer for a
    5th-order Runge-Kutta Dormand-Prince (DP5) stepper.

    Args:
        rhs:   Callable that takes exactly 2 arguments and return 1
               argument, i.e. K0 = rhs(x, y).  Here, x is a scalar, y
               is a pytree (see JAX documentation), and K0 is a pytree
               with exact same signature as y.

    Returns:
        init:  Callable that takes exactly 3 arguments and return 1
               arguments, i.e., K = step(x, y).  Here, x is scalars; y
               and Y are pytrees with exact same signature.  And K is
               a list of the same pytree.

    Both the input `rhs` and output `init` should be xmappable by JAX.
    When `rhs` or/and `init` is/are xmapped, then `x` is allowed to be
    JAX DeviceArray's with shapes that are broadcastable to the
    xmapped axes.

    """
    def init(x, y): # closure oh rhs, may be xmapped
        k = rhs(x, y)
        return [k * np.nan] * 6 + [k]

    return init


def Step(rhs):
    """Turn the RHS of a system of ODEs to a DP5 stepper

    This function takes the right-hand-side (RHS) of a system of
    ordinary differential equations (ODEs) into a 5th-order
    Runge-Kutta Dormand-Prince (DP5) stepper.

    Args:
        rhs:   Callable that takes exactly 2 arguments and return 1
               argument, i.e. K0 = rhs(x, y).  Here, x is a scalar, y
               is a pytree (see JAX documentation), and K0 is a pytree
               with exact same signature as y.

    Returns:
        step:  Callable that takes exactly 4 arguments and return 3
               arguments, i.e., Y, E, K = step(x, y, h, k).  Here, x
               and h are scalars; y, Y, E are all pytrees with exact
               same signature.  And k and K are list of the same
               pytree.

    Both the input `rhs` and output `step` should be xmappable by JAX.
    When `rhs` or/and `step` is/are xmapped, then `x` and `h` are
    allowed to be JAX DeviceArray's with shapes that are broadcastable
    to the xmapped axes.

    """
    c = {0:    0.0,    1:     1/5,    2:    3/10,    3:   4/5,    4:     8/9,      5: 1.0,   6: 1.0 }
    a = {
      0:{                                                                                           },
      1:{0:    1/5                                                                                  },
      2:{0:    3/40,   1:     9/40                                                                  },
      3:{0:   44/45,   1:-   56/15,   2:   32/9                                                     },
      4:{0:19372/6561, 1:-25360/2187, 2:64448/6561,  3:-212/729                                     },
      5:{0: 9017/3168, 1:-  355/33,   2:46732/5247,  3:  49/176,  4:- 5103/18656                    },
      6:{0:   35/384,                 2:  500/1113,  3: 125/192,  4:- 2187/6784,   5:11/84          },
    }
    e = {0:   71/57600,               2:-  71/16695, 3:  71/1920, 4:-17253/339200, 5:22/525, 6:-1/40}

    def step(x, y, h, k): # closure on rhs, may be xmapped
        K = [] if k is None else k[-1:]
        for i in range(len(K),7):
            X = x + h * c[i]
            Y = y + h * sum(v * K[j] for j, v in a[i].items()) # TODO: make it work for generic pytrees
            K.append(rhs(X, Y))
        E = h * sum(v * K[j] for j, v in e.items())
        return Y, E, K

    return step


def Dense(x, X, y, Y, K):
    """Dense output for DP5

    This function takes outputs of a PD5 step and return a callable
    for interpolation.

    Args:
        x:     Initial independent variable, scalar or JAX DeviceArray
               broadcastable to the xmapped shape.
        X:     Final independent variable, scalar or JAX DeviceArray
               broadcastable to the xmapped shape.
        y:     Initial state, which can be a pytree.
        Y:     Final state, same pytree as `y`.
        K:     List of same pytree with length 7 for generate the
               interpolation coefficients.

    Returns:
        dense: Callable that takes exactly a list of values of
               independent variable and return interpolated pytree.

    The inputs can be output of xmapped functions, or we can xmap
    `dense` to perform the interpolation.  When this happens, then
    `xs` is allowed to be multi-dimensional JAX DeviceArray with
    shapes that are broadcastable to the xmapped axes.

    """
    d = {
        0:-12715105075/11282082432,
        2: 87487479700/32700410799,
        3:-10690763975/1880347072,
        4:701980252875/199316789632,
        5:- 1453857185/822651844,
        6:    69997945/29380423,
    }
    h    = X - x
    dy   = Y - y # TODO: make it work for generic pytrees
    bspl = h * K[0] - dy
    r    = (y, dy, bspl, dy - h * K[6] - bspl, h * sum(v * K[j] for j, v in d.items()))
    cast = (...,) + (np.newaxis,) * y.ndim

    def dense(xs): # closure oh x, h, r, and cast, may be xmapped
        s = (np.array(xs)[cast] - x) / h
        t = 1 - s
        assert min(s) >= 0 and min(t) >= 0
        return r[0] + s * (r[1] + t * (r[2] + s * (r[3] + t * r[4])))

    return dense
