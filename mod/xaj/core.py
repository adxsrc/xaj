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


from jax import core
from jax.interpreters import mlir, batching


def Primitive(
    name,
    impl,
    abst=lambda x: core.ShapedArray([], x.dtype),
    axis=batching.not_mapped,
):
    """Create new JAX primitive"""

    # Make function call work
    p = core.Primitive(name)
    p.def_impl(impl)

    # Make jit() work
    mlir.register_lowering(p, mlir.lower_fun(impl, multiple_results=False))
    p.def_abstract_eval(abst)

    # Make vmap() work
    batching.primitive_batchers[p] = lambda args, axes: (
        p.bind(*args),
        axis(args, axes) if callable(axis) else axis,
    )

    return p.bind
