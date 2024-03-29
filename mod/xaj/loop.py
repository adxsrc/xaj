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


"""scan(), a conditional lux.scan() with early termination"""


from jax import lax
from jax import numpy as np
from xaj.sync import any


def iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def scan(func, carry, scanees=None, filt=None, length=None, reverse=False, unroll=1):
    """An improved `jax.lax.scan()`

    `jax.lax` provides two different loops, `while_loop()` and
    `scan()`.  The first one can terminate dynamically depending on
    `cond()` but is not capable to create output at each step.  The
    second one, on the other hand, can store output at each step but
    is not capable to terminate dynamically.

    For ODE integration, the evaluation of `rhs()` is usually
    expensive and we are interested in outputting every step.  Hence,
    it is necessary to combine the features of `jax.lax.while_loop()`
    and `jax.lax.scan()`.

    xaj.loop.scan() is exactly such a function that allows early
    termination.  In addition, it can filter out unwanted output to
    reduce memory usage.

    """
    # No filter, no early stop; use lax.scan()
    if filt is None:
        return lax.scan(func, carry, scanees, length, reverse, unroll)

    #=======================================================================
    # With filter; check arguments first

    if scanees is not None and length is not None and len(scanees) != length:
        raise ValueError(
            f'`length`=={length} cannot be different len(scanees)=={len(scanees)}')

    if not callable(filt):
        raise NotImplementedError(
            f'type(filt) == {type(filt)} is not supported.')

    if reverse:
        raise NotImplementedError(
            f'Reverse scan is not implemented yet.')

    if unroll != 1:
        raise NotImplementedError(
            f'Unroll every {unroll} steps is not implemented yet.')

    #-----------------------------------------------------------------------
    # The actual loop
    i  = 0
    xs = []

    while length is None or i < length:
        scanee = scanees[i] if iterable(scanees) else scanees

        keep, cont = filt(carry, scanee)
        if not any(cont):
            break

        carry, x = func(carry, scanee)
        if any(keep):
            xs.append(x)

        i += 1

    return carry, np.array(xs)
