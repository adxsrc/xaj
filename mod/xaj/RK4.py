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


def Step(rhs):

    def step(x, y, h, k): # closure on rhs
        K1 = h * rhs(y           )
        K2 = h * rhs(y + 0.5 * K1)
        K3 = h * rhs(y + 0.5 * K2)
        K4 = h * rhs(y +       K3)
        Y  = y + K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6
        return Y, None, [K1, K2, K3, K4]

    return step
