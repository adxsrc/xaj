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
              argument, i.e. k = rhs(x, y).  Here, x is a scalar, y
              is a pytree (see JAX documentation), and k is a pytree
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
        K1 = h * rhs(y           )
        K2 = h * rhs(y + 0.5 * K1)
        K3 = h * rhs(y + 0.5 * K2)
        K4 = h * rhs(y +       K3)
        Y  = y + K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6
        return Y, None, [K1, K2, K3, K4]

    return step
