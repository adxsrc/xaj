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


from .DP5 import Step, Dense
from .NR  import RErr, Scale

from jax import numpy as np
from jax import lax

from collections import namedtuple


def wrapper(step, rerr, filter=None, slices=None):

    def do(x, y, h, k):
        Y, E, K = step(x, y, h, k)
        R       = rerr(y, Y, E)
        return Y, R, K

    def masked_do(x, y, h, k):
        m = filter(x, y)
        if slices is None:
            hm = h / m
        else:
            hm = h / m[slices]
        Y, E, K = step(x, y, hm, k)
        R       = rerr(y, Y, E)
        return Y, np.select([m], [R], -np.inf), K

    if filter is None:
        return do
    else:
        return masked_do


class Sided:

    def __init__(self, step, dense, rerr, scale, x, y, h, filter=None, slices=None):
        self.step  = wrapper(step, rerr, filter, slices)
        self.dense = dense
        self.scale = scale

        self.x  = x
        self.y  = y
        self.k  = None

        self.h  = h
        self.r  = 0.125
        self.p  = True

        self.xs = []
        self.ys = []
        self.ds = []

    def done(self, Xt):
        if self.h > 0:
            return self.x >= Xt
        else:
            return self.x <= Xt

    def extend(self, Xt):
        while not self.done(Xt) and abs(self.h) > 1e-3:
            X       = self.x + self.h
            Y, R, K = self.step(self.x, self.y, self.h, self.k)
            R       = np.max(R)
            if not np.isfinite(R):
                break

            P = R <= 1.0
            if P: # pass
                self.xs.append(X)
                self.ys.append(Y)
                self.ds.append(self.dense(self.x, X, self.y, Y, K))

                self.x = X
                self.y = Y
                self.k = K

                self.h *= self.scale(self.r, R, self.p)
                self.r  = max(R, 1e-4)
                self.p  = P

            else: # fail and retry
                self.h *= self.scale(1, R, self.p)
                self.p  = P

    def evaluate(self, xs):
        l = []
        n = xs if self.h > 0 else xs[::-1]
        for x, d in zip(self.xs, self.ds):
            m = n <= x if self.h > 0 else n >= x
            if m.sum() > 0:
                l.append(d(n[m]))
                n = n[~m]
        if len(n) > 0:
            l.append(np.full([len(n)]+list(self.y.shape), np.nan))
        ys = np.concatenate(l)
        return ys if self.h > 0 else ys[::-1,...]


class odeint:

    IC = namedtuple('IC', ['x', 'y', 'h'])

    def __init__(
        self, rhs, x, y, h,
        eqax=None, filter=None,
        atol=1e-3, rtol=1e-3, dtype=np.float32,
    ):
        assert h > 0
        if eqax is None:
            eqax   = list(range(rhs.ndim if hasattr(rhs, 'ndim') else 1))
            slices = None
        else:
            from jax.experimental.maps import xmap
            iax    = {i:i for i   in range(y.ndim) if i not in eqax}
            oax    = {o:i for o,i in enumerate(iax.values())}
            slices = tuple(None if i in eqax else slice(None) for i in range(y.ndim))
            rhs    = xmap(rhs,    in_axes=({}, iax), out_axes=iax)
            filter = xmap(filter, in_axes=({}, iax), out_axes=oax)

        self.algo   = [Step(rhs), Dense, RErr(eqax, atol=atol, rtol=rtol), Scale()]
        self.data   = [self.IC(x, np.array(y, dtype=dtype), h), None, None]
        self.filter = filter
        self.slices = slices

    def extend(self, Xt):
        s = int(np.sign(Xt - self.data[0].x))
        if s != 0:
            if self.data[s] is None:
                ic = self.data[0]
                self.data[s] = Sided(*self.algo, ic.x, ic.y, s * ic.h, self.filter, self.slices)
            self.data[s].extend(Xt)

    @property
    def xs(self):
        xs = [self.data[0].x]
        if self.data[-1] is not None:
            xs = self.data[-1].xs[::-1] + xs
        if self.data[ 1] is not None:
            xs = xs + self.data[1].xs
        return np.array(xs)

    @property
    def ys(self):
        ys = [self.data[0].y]
        if self.data[-1] is not None:
            ys = self.data[-1].ys[::-1] + ys
        if self.data[ 1] is not None:
            ys = ys + self.data[1].ys
        return np.array(ys)

    def evaluate(self, xs):
        l = [self.data[0].y[np.newaxis, ...]] * (xs == self.data[0].x).sum()
        xm = xs[xs < self.data[0].x]
        if len(xm) > 0:
            l = [self.data[-1].evaluate(xm)] + l
        xp = xs[xs > self.data[0].x]
        if len(xp) > 0:
            l = l + [self.data[ 1].evaluate(xp)]
        return np.concatenate(l)

    def __call__(self, xs):
        self.extend(max(xs))
        self.extend(min(xs))
        return self.evaluate(xs)
