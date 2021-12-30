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


def RErr(eqax=[0], atol=1e-4, rtol=1e-4):
    """Turn some parameters into a error function

    This function takes some setings such as `atol` and `rtol` and
    return a callable that comptues a "relative error".

    Args:
        eqax:  Axes for the root-mean-sqaure to run on.
        atol:  Absolute tolerance.
        rtol:  Relative tolerance.

    Returns:
        rerr:  A callable to compute the relative error from
               initial state `y`, final state `Y`, and estimated error
               `E`.  `y`, `Y`, and `E` can all be pytree with same
               signature.

    """
    def rerr(y, Y, E): # closure on atol and rtol
        r = E / (atol + rtol * np.maximum(abs(y), abs(Y))) # TODO: make it work for generic pytrees
        return np.sqrt(np.mean(r * r, axis=eqax))

    return rerr


def Scale(safe=0.875, alpha=None, beta=None, minscale=0.125, maxscale=8.0, order=5):
    """Turn some parameters into a scaling function

    This function takes some setings such as `safe`, `minscale`, and
    `maxscale` and return a callable that computes a "scaling factor"
    for time step control.  The formulation is based on the Numerical
    Recipe, with some specific choice on the default parameters.

    Args:
        TODO...

    Returns:
        scale: A callable to compute the scaling factor for time step
               control.  The formulation is based on the Numerical
               Recipe.  It takes the last and current relative errors
               `g` and `G`, and propose a scaling `s` for the next
               step.  `g`, `G`, and `s` can be pytree with the same
               signature.

    """
    if beta  is None:
        beta  = 0.4 / order
    if alpha is None:
        alpha = 1.0 / order - 0.75 * beta

    def scale(r, R, passed=True): # closure on safe, alpha, beta, minscale, maxscale, xmappable
        # TODO: make it work for generic pytrees
        if R == 0.0:
            s = maxscale
        else:
            s = np.clip(safe * r**beta * R**-alpha, minscale, maxscale)
        if passed:
            return s
        else:
            return min(s, 1.0)

    return scale
