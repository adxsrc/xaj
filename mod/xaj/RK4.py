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


def RK4(rhs, state, dt):
    k1 = dt * rhs(state           )
    k2 = dt * rhs(state + 0.5 * k1)
    k3 = dt * rhs(state + 0.5 * k2)
    k4 = dt * rhs(state +       k3)
    return state + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
