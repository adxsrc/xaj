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


"""Test XAJ's scan() function"""


from jax.config import config
config.update("jax_enable_x64", True)

from xaj.loop import scan
from jax import vmap, jit
from jax import numpy as np


def test_scan():

    def func(carry, scanee):
        carry += 1
        return carry, carry

    sf  = lambda *args, **kwargs: scan(func, *args, **kwargs)
    jsf = jit(sf)
    vsf = vmap(sf, (0,None))

    c, s = sf(0, length=4)
    assert c == 4 and np.all(s == np.array([1, 2, 3, 4]))

    c, s = sf(0, np.arange(4))
    assert c == 4 and np.all(s == np.array([1, 2, 3, 4]))

    c, s = sf(0, np.arange(4), length=4)
    assert c == 4 and np.all(s == np.array([1, 2, 3, 4]))

    c, s = jsf(0, np.arange(4))
    assert c == 4 and np.all(s == np.array([1, 2, 3, 4]))

    c, s = vsf(np.array([0,10,20]), np.arange(4))
    assert (
        np.all(c == np.array([4,14,24])) and
        np.all(s == np.array([
            [ 1,  2,  3,  4],
            [11, 12, 13, 14],
            [21, 22, 23, 24],
        ]))
    )


    def filt(carry, scanee):
        return True, carry < 2

    sf  = lambda c: scan(func, c, filt=filt, length=None)
    jsf = jit(sf)
    vsf = vmap(sf)

    c, s = vsf(np.array([0,10,20]))
    assert (
        np.all(c == np.array([2,12,22])) and
        np.all(s == np.array([
            [ 1,  2],
            [11, 12],
            [21, 22],
        ]))
    )
