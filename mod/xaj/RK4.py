# Copyright (C) 2020 Chi-kwan Chan
# Copyright (C) 2020 Steward Observatory
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


def Step(rhs):
    """Turn the RHS of a system of ODEs to a RK4 stepper

    This function takes the right-hand-side (RHS) of a system of
    ordinary differential equations (ODEs) into a classic 4th-order
    Runge-Kutta stepper.

    Args:
        rhs:  Callable that takes exactly 2 arguments and return 1
              argument, i.e. K0 = rhs(x, y).  Here, x is a scalar, y
              is a pytree (see JAX documentation), and K0 is a pytree
              with exact same signature as y.

    Returns:
        step: Callable that takes exactly 4 arguments and return 3
              arguments, i.e., Y, E, K = step(x, y, h, k).  Here, x
              and h are scalars; y, Y, E; are all pytree with exact
              same signatures.  And k and K are list of the same
              pytree.

    Both the input `rhs` and output `step` should be xmappable by JAX.
    When `rhs` or/and `step` is/are xmapped, then `x` and `h` are
    allowed to be JAX DeviceArray's with shapes that are broadcastable
    to the xmapped axes.

    """
    def step(x, y, h, k): # closure on rhs
        hK0 = h * rhs(x,           y            )
        hK1 = h * rhs(x + 0.5 * h, y + 0.5 * hK0)
        hK2 = h * rhs(x + 0.5 * h, y + 0.5 * hK1)
        hK3 = h * rhs(x +       h, y +       hK2)
        Y   = y + hK0 / 6 + hK1 / 3 + hK2 / 3 + hK3 / 6
        return Y, None, None

    return step
