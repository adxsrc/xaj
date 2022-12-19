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


"""Sync across multiple elements in a vmap"""


from jax.core import Primitive, ShapedArray

from jax.interpreters.mlir     import register_lowering,  lower_fun
from jax.interpreters.batching import primitive_batchers, not_mapped

from jax import numpy as np


def Sync(name, f):

    def impl(x):
        return f(x)

    # Make function call work
    p = Primitive(name)
    p.def_impl(impl)

    # Make jit() work
    register_lowering(p, lower_fun(impl, multiple_results=False))
    p.def_abstract_eval(lambda x: ShapedArray([], x.dtype))

    # Make vmap() work
    primitive_batchers[p] = lambda a, _: (p.bind(*a), not_mapped)

    return p.bind


any = Sync("any", np.any)
