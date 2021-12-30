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


class Trek:
    """Trek the system of ODEs with multi-step and provide interpolations

    Compared to pace(), trek() keeps lists of `x`, `y`, and the dense
    outputs.  It is more "stateful" in the sense that it is used by
    calling the trek.extend() function, which updates the internal
    states.  The direction of the extension is determined by the
    initial `h`.

        Trek (verb): go on a long arduous journey, typically on foot.

    """
    def __init__(self, pace, dense, x, y, k):
        self.pace  = pace
        self.dense = dense
        self.ds = [ ] # self.ds always has one less element than xs and ys
        self.xs = [x]
        self.ys = [y]
        self.k  =  k

    def done(self, Xt):
        if self.pace.h > 0:
            return self.xs[-1] >= Xt
        else:
            return self.xs[-1] <= Xt

    def extend(self, Xt):
        while not self.done(Xt):
            X, Y, K = self.pace(self.xs[-1], self.ys[-1], self.k)
            self.ds.append(self.dense(self.xs[-1], X, self.ys[-1], Y, K))
            self.xs.append(X)
            self.ys.append(Y)
            self.k = K

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
