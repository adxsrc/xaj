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


from .trek import Trek

from jax import numpy as np
from collections import namedtuple


class odeint:

    IC = namedtuple('IC', ['x', 'y', 'h'])

    def __init__(self,
        rhs, x, y, h,
        eqax=None, hlim=None, filter=None,
        dtype=np.float32,
        **kwargs,
    ):
        assert h > 0

        x = np.asarray(x, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
        h = np.asarray(h, dtype=dtype)

        if eqax is not None:
            from jax.experimental.maps import xmap
            fax = {i:i for   i in range(y.ndim) if i not in eqax}
            sax = {o:i for o,i in enumerate(fax.values())}
            rhs = xmap(rhs, in_axes=({}, fax), out_axes=fax)
            if hlim   is not None:
                hlim   = xmap(hlim,   in_axes=({}, fax), out_axes=sax)
            if filter is not None:
                xmf    = xmap(filter, in_axes=({}, fax), out_axes=sax)
                slices = tuple(None if i in eqax else slice(None) for i in range(y.ndim))
                filter = lambda x, y: xmf(x, y)[slices] # closure on xmf and slices

        kwargs['eqax'  ] = eqax
        kwargs['hlim'  ] = hlim
        kwargs['filter'] = filter

        self.rhs    = rhs
        self.data   = [self.IC(x, y, h), None, None]
        self.kwargs = kwargs

    def extend(self, Xt, **kwargs):
        s = int(np.sign(Xt - self.data[0].x))
        if s != 0:
            if self.data[s] is None:
                ic = self.data[0]
                self.data[s] = Trek(self.rhs, ic.x, ic.y, s * ic.h, **self.kwargs)
            self.data[s].extend(Xt, **kwargs)

    @property
    def xs(self):
        xs = [self.data[0].x]
        if self.data[-1] is not None:
            xs = self.data[-1].xs[:0:-1] + xs
        if self.data[ 1] is not None:
            xs = xs + self.data[1].xs[1::]
        return np.array(xs)

    @property
    def ys(self):
        ys = [self.data[0].y]
        if self.data[-1] is not None:
            ys = self.data[-1].ys[:0:-1] + ys
        if self.data[ 1] is not None:
            ys = ys + self.data[1].ys[1::]
        return np.array(ys)

    def evaluate(self, xs):
        l = [self.data[0].y[np.newaxis,...]] * (xs == self.data[0].x).sum()
        xm = xs[xs < self.data[0].x]
        if len(xm) > 0:
            l = [self.data[-1].evaluate(xm)] + l
        xp = xs[xs > self.data[0].x]
        if len(xp) > 0:
            l = l + [self.data[ 1].evaluate(xp)]
        return np.concatenate(l)

    def __call__(self, xs, **kwargs):
        self.extend(max(xs), **kwargs)
        self.extend(min(xs), **kwargs)
        return self.evaluate(xs)
