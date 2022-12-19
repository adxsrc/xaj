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


from jax import numpy as np

from chex import (
    assert_shape,
    assert_equal_shape,
    assert_rank,
    assert_type,
)


a0 = np.array(0.0)
a1 = np.zeros((2))
a2 = np.zeros((2, 3))
a3 = np.zeros((2, 3, 5))
a4 = np.zeros((2, 3, 5, 7))
a5 = np.zeros((2, 3, 5, 7))
a6 = np.zeros((2, 3, 5, 7))


def test_shape():
    assert_shape(a2, (2, 3))  # `a2` has shape (2, 3)
    assert_shape([a0, a2], [(), (2, 3)])  # `a0` is scalar and `a2` has shape (2, 3)
    assert_equal_shape([a4, a5, a6])  # `a4`, `a5`, and `a6` have equal shapes

def test_rank():
    assert_rank(a0, 0)  # `a0` is scalar
    assert_rank([a0, a2], [0, 2])  # `a0` is scalar and `a2` is a rank-2 array
    assert_rank([a0, a2], {0, 2})  # `a0` and `a2` are scalar OR rank-2 arrays

def test_type():
    assert_type(1, int)  # 1 has type `int` (x can be an array)
    assert_type([1, a0], [int, float])  # 1 has type `int` and `a0` has type `float`
