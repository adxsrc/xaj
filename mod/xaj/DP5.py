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


from jax import numpy as np


class DP5:
    """Dormand-Prince 4/5 Scheme"""

    c  = np.array([0,           1/5,        3/10,        4/5,      8/9,          1])
    a1 = np.array([1/5])
    a2 = np.array([3/40,        9/40])
    a3 = np.array([44/45,      -56/15,      32/9])
    a4 = np.array([19372/6561, -25360/2187, 64448/6561, -212/729])
    a5 = np.array([9017/3168,  -355/33,     46732/5247,  49/176,  -5103/18656])
    a6 = np.array([35/384,      0,          500/1113,    125/192, -2187/6784,    11/84])
    e  = np.array([71/57600,    0,         -71/16695,    71/1920, -17253/339200, 22/525, -1/40])

    def __init__(self, rhs):

        def step(x, y, h, k6): # closure on coefficients and rhs

            xch = x + self.c  * h
            a1h =     self.a1 * h
            a2h =     self.a2 * h
            a3h =     self.a3 * h
            a4h =     self.a4 * h
            a5h =     self.a5 * h
            a6h =     self.a6 * h
            eh  =     self.e  * h

            k0  = k6 # of previous step, i.e., == rhs(x, y) for this step
            k1  = rhs(xch[1], y + a1h[0]*k0)
            k2  = rhs(xch[2], y + a2h[0]*k0 + a2h[1]*k1)
            k3  = rhs(xch[3], y + a3h[0]*k0 + a3h[1]*k1 + a3h[2]*k2)
            k4  = rhs(xch[4], y + a4h[0]*k0 + a4h[1]*k1 + a4h[2]*k2 + a4h[3]*k3)
            k5  = rhs(xch[5], y + a5h[0]*k0 + a5h[1]*k1 + a5h[2]*k2 + a5h[3]*k3 + a5h[4]*k4)
            Y   =             y + a6h[0]*k0 +             a6h[2]*k2 + a6h[3]*k3 + a6h[4]*k4 + a6h[5]*k5
            k6  = rhs(xch[5], Y) # will be used in the next step, hence the output
            E   =                 eh [0]*k0 +             eh [2]*k2 + eh [3]*k3 + eh [4]*k4 + eh [5]*k5 + eh [6]*k6

            return DP5Output(x, y, h, k0, k2, k3, k4, k5, k6, Y), Y, E, k6

        self.step = step


class DP5Output:
    """Dense Output of the DP5 Schemef"""

    d = np.array([
        -12715105075/11282082432,
        +0,
        +87487479700/32700410799,
        -10690763975/1880347072,
        +701980252875/199316789632,
        -1453857185/822651844,
        +69997945/29380423,
    ])

    def __init__(self, x, y, h, k0, k2, k3, k4, k5, k6, Y):

        ydiff = Y - y
        bspl  = k0 * h - ydiff
        dh    = self.d * h

        self.x  = x
        self.h  = h
        self.r0 = y
        self.r1 = ydiff
        self.r2 = bspl
        self.r3 = ydiff - k6 * h - bspl
        self.r4 = dh[0]*k0 + dh[2]*k2 + dh[3]*k3 + dh[4]*k4 + dh[5]*k5 + dh[6]*k6

    def denseout(self, x):
        s  = (x - self.x) / self.h
        assert 0 <= s and s <= 1
        s1 = 1 - s
        return self.r0 + s*(self.r1 + s1*(self.r2 + s*(self.r3 + s1*self.r4)))
