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


def RErr(axis, atol=1e-4, rtol=1e-4):

    def rerr(y, Y, E): # closure on atol and rtol
        r = E / (atol + rtol * np.maximum(abs(y), abs(Y)))
        return np.sqrt(np.mean(r * r, axis=axis))

    return rerr


def Scale(safe=0.875, alpha=None, beta=None, minscale=0.125, maxscale=8.0, order=5):

    if beta  is None:
        beta  = 0.4 / order
    if alpha is None:
        alpha = 1.0 / order - 0.75 * beta

    def scale(g, G, passed=True): # closure on safe, alpha, beta, minscale, maxscale
        if G == 0.0:
            s = maxscale
        else:
            s = np.clip(safe * g**beta * G**-alpha, minscale, maxscale)
        if passed:
            return s
        else:
            return min(s, 1.0)

    return scale
