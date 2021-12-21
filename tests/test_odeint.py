# Copyright (C) 2021 Chi-kwan Chan
# Copyright (C) 2021 Steward Observatory
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


from xaj import odeint
from jax import numpy as np

def test_odeint():

    def rhs(x, y):
        return np.array([y[1], -y[0]])

    x0 = 0
    y0 = np.array([0,1])

    ns = odeint(rhs, x0, y0, 1, atol=1e-6, rtol=0)

    x  = np.linspace(-5, 5, num=101)
    yn = ns(x)[:,0]
    ya = np.sin(x)

    err = max(abs(yn-ya))
    print('Maximum error:', err)
    assert err <= 1e-6
