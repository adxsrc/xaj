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


from .DP5  import Init, Step, Dense
from .pace import Pace

from jax import numpy as np


class Trek:
    """Trek the system of ODEs with multi-step and provide interpolations

    Compared to pace(), trek() keeps lists of `x`, `y`, and the dense
    outputs.  It is more "stateful" in the sense that it is used by
    calling the trek.extend() function, which updates the internal
    states.  The direction of the extension is determined by the
    initial `h`.

        Trek (verb): go on a long arduous journey, typically on foot.

    """
    def __init__(self,
        rhs, x, y, h,
        N=None, steps=True, dense=True,
        names=None,
        **kwargs,
    ):
        self.pace  = Pace(Step(rhs), h, **kwargs)
        self.N     = 10000 if N is None else N
        self.steps = steps
        self.dense = Dense if dense else None
        self.names = names
        self.ds    = [ ] # self.ds always has one less element than xs and ys
        self.xs    = [x]
        self.ys    = [y]
        self.k     = Init(rhs)(x, y)

    def done(self, Xt):
        s = self.pace.sign()
        return s * self.xs[-1] >= s * Xt

    def extend(self, Xt, N=None):
        if self.done(Xt):
            print(f'target = {Xt} exceeded; SKIP')
            return

        if N is None:
            N = self.N

        pbar = None
        if self.names is not None:
            x0  = self.xs[0]
            ind = self.names['ind']

        for _ in range(N):
            X, Y, K = self.pace(self.xs[-1], self.ys[-1], self.k)
            if self.done(Xt):
                break

            if self.dense is not None:
                self.ds.append(self.dense(self.xs[-1], X, self.ys[-1], Y, K))

            if self.steps:
                self.xs.append(X)
                self.ys.append(Y)
            else:
                self.xs[-1] = X
                self.ys[-1] = Y

            self.k = K

            if self.names is not None:
                p = int(100 * np.clip((X - x0) / (Xt - x0), 0, 1))
                if pbar is None:
                    from tqdm import tqdm
                    pbar = tqdm(initial=p, total=100, position=0, leave=True)
                pbar.set_postfix_str(f'{ind}={X:<#.3g}, d{ind}={self.pace.h:<#9.3g}')
                pbar.update(p - pbar.n)
        else:
            print(f'Integration unfinished, try increasing allowed number of steps N={N}')

        if pbar is not None:
            pbar.close()

    def evaluate(self, xs):
        f = self.pace.h > 0
        l = []
        n = xs if f else xs[::-1]
        for x, d in zip(self.xs[1:], self.ds):
            m = n <= x if f else n >= x
            if m.sum() > 0:
                l.append(d(n[m]))
                n = n[~m]
        if len(n) > 0:
            l.append(np.full([len(n)]+list(self.ys[-1].shape), np.nan))
        ys = np.concatenate(l)
        return ys if f else ys[::-1,...]
