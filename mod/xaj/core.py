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


"""Numerical schemes"""


from jax.lax  import switch, while_loop, select
from xaj.sync import any


def Step(rhs):
    """Stepper Factory"""

    def Euler(h, t, x, k):
        """Forward Euler scheme"""
        return None, t + h, x + h * rhs(t, x), None

    return Euler


def StepControl():
    """Step Controller Factory"""

    def constant(h, t, x, E, T, X):
        """Constant step"""
        return h, False

    return constant


def Engine(step, ctrl, imax=1024, rmax=32):
    """Stepping Engine Factory

    This function takes the necessary configurations and return a
    stepping engine as a pure function.

    The variables used here have the following meanings:

    t, T: current and next independent variable, usually "time"
    x, X: current and next dependent variable(s), usually "position"
    h, H: current and next step size
    k, K: substeps data for stepper; may be `None` if not used
    i, r: iteration and refinement counters

    """

    def cond(state):
        """Condition for the lax while loop

        Closure on imax and rmax.

        """
        target, (t,_), (h,_), (i,r) = state
        return (i < imax) & (r < rmax) & select(h < 0.0, target < t, t < target)

    def body(state):
        """Body function for the lax while loop

         Closure on step and ctrl.

        """
        _, (t,x), (h,k), (i,r) = state

        E, T, X, K = step(h, t, x, k)
        H, retry   = ctrl(h, t, x, E, T, X)

        return switch(int(retry), [
            lambda: (_, (T,X), (H,K), (i+1, 0)), # continue
            lambda: (_, (t,x), (h,k), (i, r+1)), # retry
        ])

    def engine(target, tx, hk):
        """Stepping Engine

        Closure on cond() and body().

        """
        _, tx, hk, (i,r) = while_loop(cond, body, (target, tx, hk, (0,0)))

        if any(i >= imax):
            raise RuntimeWarning(
                f"Number of iterations i={i} exceed imax={imax}")

        if any(r >= rmax):
            raise RuntimeWarning(
                f"Number of step refinements r={r} reaches rmax={rmax}")

        return tx, hk

    return engine
