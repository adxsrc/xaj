# Copyright 2022 Chi-kwan Chan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Demonstrating JAX's properties"""


from jax import numpy as np, vmap
from jax.lax import fori_loop, while_loop, scan

from chex import assert_tree_all_close as assert_agree


def test_fori_loop():
    def inc(i, v):
        return v + 1
    def incfor(i, j, v):
        return fori_loop(i, j, inc, v)
    vincfor = vmap(incfor, (0, None, None))

    a = np.array([0, 5])
    b = vincfor(a, 10, 10)

    assert_agree(b,
        np.array([20,15])
    )


def test_while_loop():
    def cond(v):
        i, x = v
        return x < 100
    def body(v):
        i, x = v
        return i+1, x+1
    def until(i, x):
        return while_loop(cond, body, (i, x))
    vuntil = vmap(until, (0, 0))

    i = np.array([0, 0])
    x = np.array([0,50])
    b = vuntil(i, x)

    assert_agree(b, (
        np.array([100,  50]), # second element needs only 50 steps
        np.array([100, 100]), # both elements go all the way to 100
    ))
