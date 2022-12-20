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


from jax.config import config
config.update("jax_enable_x64", True)

from xaj.engine import *
from jax import vmap
from jax import numpy as np


def rhs(t, x):
    return x


def test_engine():

    step = Step(rhs)
    ctrl = StepControl()
    engn = Engine(step, ctrl)

    (t,x0), (h,k) = vmap(engn, (0, (None), (None)))(
        np.array([1.0, 2.0, 3.0]),
        (0.0,1.0),
        (0.01,None)
    )

    print(x0)

    # Manual Euler method
    x1 = 1.0
    for _ in range(100):
        x1 += 0.01 * x1

    x2 = 1.0
    for _ in range(200):
        x2 += 0.01 * x2

    x3 = 1.0
    for _ in range(301):
        x3 += 0.01 * x3

    # Compare
    assert np.all(x0 == np.array([x1, x2, x3]))
