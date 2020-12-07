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

    def __init__(
            self, rhs,
            x=0, y=None, h=None,
            atol=1e-4, rtol=1e-4,
            alpha=None, beta=None,
            safe=0.9, minscale = 0.2, maxscale = 10.0
    ): # may not be xmapped

        if beta is None:
            beta = 0.08
        if alpha is None:
            alpha = 0.2 - beta * 0.75

        # Required
        self.rhs = rhs

        # Internal states
        self.x  = x
        self.y  = y
        self.h  = h
        self.k6 = None if y is None else rhs(x, y)

        # Parameters
        self.atol     = atol
        self.rtol     = rtol
        self.alpha    = alpha
        self.beta     = beta
        self.safe     = safe
        self.minscale = minscale
        self.maxscale = maxscale

    def reset(self, x, y, h=None): # may be xmapped
        self.x  = x
        self.y  = y
        self.h  = h
        self.k6 = self.rhs(x, y)

    def step(self, h): # may be xmapped, would not change internal states
        xch = self.x + self.c  * h
        a1h =          self.a1 * h
        a2h =          self.a2 * h
        a3h =          self.a3 * h
        a4h =          self.a4 * h
        a5h =          self.a5 * h
        a6h =          self.a6 * h
        eh  =          self.e  * h

        k0  = self.k6 # of previous step, i.e., == rhs(x, y) for this step
        k1  = self.rhs(xch[1], self.y + a1h[0]*k0)
        k2  = self.rhs(xch[2], self.y + a2h[0]*k0 + a2h[1]*k1)
        k3  = self.rhs(xch[3], self.y + a3h[0]*k0 + a3h[1]*k1 + a3h[2]*k2)
        k4  = self.rhs(xch[4], self.y + a4h[0]*k0 + a4h[1]*k1 + a4h[2]*k2 + a4h[3]*k3)
        k5  = self.rhs(xch[5], self.y + a5h[0]*k0 + a5h[1]*k1 + a5h[2]*k2 + a5h[3]*k3 + a5h[4]*k4)
        Y   =                  self.y + a6h[0]*k0 +             a6h[2]*k2 + a6h[3]*k3 + a6h[4]*k4 + a6h[5]*k5
        k6  = self.rhs(xch[5], Y) # will be used in the next step, hence the output
        E   =                           eh [0]*k0 +             eh [2]*k2 + eh [3]*k3 + eh [4]*k4 + eh [5]*k5 + eh [6]*k6

        return Y, E, (k0, k1, k2, k3, k4, k5, k6)

    def goodness(self, y, Y, E): # may be xmapped
        rerr = E / (self.atol + self.rtol * np.maximum(abs(y), abs(Y)))
        return np.sqrt(np.mean(rerr * rerr))

    def scale(self, g, G, rejected):

        if G == 0.0:
            s = self.maxscale
        else:
            s = np.clip(self.safe * g**self.beta * G**-self.alpha, self.minscale, self.maxscale)

        if rejected:
            return np.min(np.array([s, 1.0]))
        else:
            return s

    def preint(self, X, verbose=False):
        s = np.sign(X - self.x)
        h = X - self.x if self.h is None else s * abs(self.h)
        r = False
        g = 1e-4

        while self.x < X if s >= 0 else self.x > X:
            Y, E, k = self.step(h)

            G = self.goodness(self.y, Y, E)
            if G <= 1.0:
                if verbose:
                    msg = f'Accepted: x,h = {self.x:.3f},{h:.3f} -> '

                self.x += h
                self.y  = Y
                self.h  = h
                self.k6 = k[6]

                h *= self.scale(g, G, r)
                r  = False
                g  = np.max(np.array([G, 1e-4]))
            else:
                if verbose:
                    msg = f'Rejected: x,h = {self.x:.3f},{h:.3f} -> '

                h *= self.scale(1, G, r)
                r  = True

            if verbose:
                print(msg + f'x={self.x:.3f},{h:.3f}')


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

    def __init__(self, x, X, y, Y, k):

        h     = X - x
        ydiff = Y - y
        bspl  = k[0] * h - ydiff
        dh    = self.d * h

        self.x  = x
        self.h  = h
        self.r0 = y
        self.r1 = ydiff
        self.r2 = bspl
        self.r3 = ydiff - k[6] * h - bspl
        self.r4 = dh[0]*k[0] + dh[2]*k[2] + dh[3]*k[3] + dh[4]*k[4] + dh[5]*k[5] + dh[6]*k[6]

    def __call__(self, x):
        s  = (x - self.x) / self.h
        assert all(0 <= s) and all(s <= 1)
        s1 = 1 - s
        return self.r0 + s*(self.r1 + s1*(self.r2 + s*(self.r3 + s1*self.r4)))