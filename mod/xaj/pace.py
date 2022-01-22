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


from .NR import RErr, Scale

from jax import numpy as np
from jax import jit


def nanmask(m, v):
    """Mask a value or an array `v` with `nan` according to `m`"""
    return np.select([m], [v], np.nan)


def wrapper(step, rerr, filter=None):

    def do(x, y, h, k):
        print('jit(do); input:', x, y, h, k) # will be jitted away
        Y, E, K = step(x, y, h, k)
        R       = rerr(y, Y, E)
        return Y, R, K

    def masked_do(x, y, h, k):
        print('jit(masked_do); input:', x, y, h, k) # will be jitted away
        m       = filter(x, y)
        Y, E, K = step(x, y, nanmask(m, h), k)
        R       = rerr(y, Y, E)
        return Y, R, K

    if filter is None:
        return do
    else:
        return masked_do


class Pace:
    """Pace the system of ODEs forward by a step

    Compared to step(), pace() has internal states and keep track of
    the optimal step size.  If the proposed step size is too small,
    pace() would try `n` times until a small enough step size is
    found.

        Pace (verb): walk at a steady and consistent speed, especially
        back and forth and as an expression of one's anxiety or
        annoyance.

    """
    def __init__(self,
        step, h,       hmin=1e-6, hlim=None, filter=None,
        n=8,  r=0.125, rmin=1e-4,
        eqax=None, atol=1e-4, rtol=1e-4,
        **kwargs,
    ):
        rerr  = RErr(eqax, atol, rtol)
        scale = Scale(**kwargs)

        # Internal methods
        self.step  = jit(wrapper(step, rerr, filter))
        self.hlim  = jit(hlim) if hlim else hlim
        self.scale = scale

        # Constant settings
        self.n    = n
        self.hmin = hmin
        self.rmin = rmin

        # Varying states
        self.h = h
        self.p = True
        self.r = r

    def sign(self):
        if np.isnan(self.r):
            print('All equations were filtered out')
            return 0
        elif abs(self.h) < self.hmin:
            print('Step size became too small')
            return 0
        else:
            return int(np.sign(self.h))

    def __call__(self, x, y, k):
        # Step size limiter
        if self.hlim is not None:
            H = self.hlim(x, y)
            H = np.nanmin(H)
            if abs(self.h) > H:
                self.h = self.sign() * H

        # Try adjust step size
        for _ in range(self.n):
            Y, R, K = self.step(x, y, self.h, k)
            X       = x + self.h
            R       = np.nanmax(R) # xaj supports only global step for now
            if np.isnan(R):
                break # do not update `self.h` and `self.p`

            P       = R <= 1.0
            self.h *= self.scale(self.r, R, self.p, P)
            self.p  = P
            if P:
                break
        else:
            raise RuntimeError(f'Refinement fails, try increase refinement step self.n={self.n}')

        # Done; update internal states
        self.r = max(R, self.rmin) # unlike self.p, self.r is only updated if pass or R == nan
        return X, Y, K
