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


def DP5(rhs):

    c = {0:    0.0,    1:     1/5,    2:    3/10,    3:   4/5,    4:     8/9,      5: 1.0,   6: 1.0 }
    a = {
      0:{                                                                                           },
      1:{0:    1/5                                                                                  },
      2:{0:    3/40,   1:     9/40                                                                  },
      3:{0:   44/45,   1:-   56/15,   2:   32/9                                                     },
      4:{0:19372/6561, 1:-25360/2187, 2:64448/6561,  3:-212/729                                     },
      5:{0: 9017/3168, 1:-  355/33,   2:46732/5247,  3:  49/176,  4:- 5103/18656                    },
      6:{0:   35/384,                 2:  500/1113,  3: 125/192,  4:- 2187/6784,   5:11/84          },
    }
    e = {0:   71/57600,               2:-  71/16695, 3:  71/1920, 4:-17253/339200, 5:22/525, 6:-1/40}

    def step(x, y, h, k): # closure on rhs, may be xmapped
        K = [] if k is None else k[-1:]
        for i in range(len(K),7):
            X = x + h * c[i]
            Y = y + h * sum(v * K[j] for j, v in a[i].items())
            K.append(rhs(X, Y))
        E = h * sum(v * K[j] for j, v in e.items())
        return Y, E, K

    return step


def DP5dense(x, X, y, Y, K):

    d = {
        0:-12715105075/11282082432,
        2: 87487479700/32700410799,
        3:-10690763975/1880347072,
        4:701980252875/199316789632,
        5:- 1453857185/822651844,
        6:    69997945/29380423,
    }
    h    = X - x
    dy   = Y - y
    bspl = h * K[0] - dy
    r    = (y, dy, bspl, dy - h * K[6] - bspl, h * sum(v * K[j] for j, v in d.items()))
    cast = (...,) + (np.newaxis,) * y.ndim

    def dense(xs): # closure oh x, h, r, and cast
        s = (np.array(xs)[cast] - x) / h
        t = 1 - s
        assert min(s) >= 0 and min(t) >= 0
        return r[0] + s * (r[1] + t * (r[2] + s * (r[3] + t * r[4])))

    return dense
