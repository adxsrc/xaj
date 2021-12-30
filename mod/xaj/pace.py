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
    def __init__(self, step, scale, h, n=8, r=0.125):
        self.step  = step
        self.scale = scale
        self.n     = n
        self.h     = h
        self.p     = True
        self.r     = r

    def __call__(self, x, y, k):
        for _ in range(self.n):
            Y, R, K = self.step(x, y, self.h, k)
            X       = x + self.h
            R       = np.max(R) # xaj supports only global step for now
            P       = R <= 1.0
            self.h *= self.scale(self.r, R, self.p, P)
            self.p  = P
            if P:
                break
        self.r = max(R, 1e-4) # unlike self.p, self.r is only updated if pass
        return X, Y, K
