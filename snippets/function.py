# function.py ---
#
# Filename: function.py
# Description:
# Author: Subhasis Ray
# Maintainer:
# Created: Tue Sep  9 17:59:50 2014 (+0530)
# Version:
# Last-Updated: Tue Jun 17 15:55:30 2025 (+0530)
#           By: Subhasis Ray
#     Update #: 221
# URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change log:
#
#
#
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
#
#

# Code:

import numpy as np
import sys
import matplotlib.pyplot as plt

import moose


def function_example(simtime=1.0):
    """Function objects can be used to evaluate expressions with arbitrary
    number of variables and constants. We can assign expression of the
    form::

        f(t, c0, c1, ..., cM, x0, x1, ..., xN, y0,..., yP )

    where `c_i`'s are constants and `x_i`'s and `y_i`'s are variables.
    The syntax of the expression follows exprtk
    (https://github.com/ArashPartow/exprtk), the C++ library used for
    parsing and evaluating it.
    
    The constants must be defined before setting the expression.
    This is done by assigning key, value pairs to the lookup field `c`.

    The identifier `t` represents simulated time. This is always taken
    from the clock ticks associated with the function object.
    
    Function differentiates variables based on how their values are
    obtained. There are two arrays of variables: `x` and `y`.

    - The field element `x` has an 'input' destination field in every
      entry. You can connect any source field sending out double to
      this field. Thus values of `x` variables are 'pushed' by source
      objects.

    - If ``allowUnknownVariables`` field is `False`, then field element
      entry `x[i]` corresponds to 'xi' in `expr`. If
      `allowUnknowVariables` field is `True`, then identifiers not
      defined as a constant are treated as variable.

      Among these, those that are not of the form 'yi', where `i` is a
      nonnegative integer, are stored in the ``x`` array. The index
      `i` can be looked up in the ``xindex`` field. `index =
      function.xindex[v]` will give the index of the variable 'v' in
      `expr`.

   - If you want to include a value field `field` of another element
     as a variable, you must use an identifier of the form 'yi' for it
     in the expression, and connect the `requestOut` source field of
     the function element to the `get{Field}` destination field on the
     target element. Thus values for `y` variables are 'pulled' from
     the target element field.

     The y-index follows the sequence of connection, starting with
     0. Thus, if you call::

       `moose.connect(function, 'requestOut', a, 'getSomeField')`
       `moose.connect(function, 'requestOut', b, 'getAnotherField')`

    then ``a.someField`` will be assigned to ``y0`` and
    ``b.anotherField`` will be assigned to ``y1``.

    In this example we evaluate the expression:

    `Q * exp(-t / tau) + stim + cos(y0)`

    where `tau` is a constant, `Q` is pushed from a ``PulseGen``
    object, `stim` is pushed from a ``StimulusTable``, and ``y0`` is
    pulled from another ``StimulusTable`` object.

    Along with the value of the expression itself we also compute its
    derivative with respect to 'y0' and its derivative with respect to
    time (rate).

    Note that Function elements are assigned a slow clock tick by
    default, hence we need to update the `dt` of the clock using
    ``moose.setClock`` function to that of its input elements. Also
    notice the blip at the onset.

    """
    demo = moose.Neutral('/model')
    function = moose.Function('/model/fn')
    function.c['tau'] = 1.0
    function.expr = 'Q * exp(-t/tau) + stim + cos(y0)'
    # mode 0 - evaluate function value, derivative and rate
    # mode 1 - just evaluate function value,
    # mode 2 - evaluate derivative,
    # mode 3 - evaluate only rate
    function.mode = 0
    function.independent = 'y0'

    pg = moose.PulseGen(f'{demo.path}/pg')
    pg.firstDelay = 0.5
    pg.firstWidth = 1.0
    pg.firstLevel = 1.0
    pg.baseLevel = 0.0
    # Stimulus tables allow you to store sequences of numbers which
    # are delivered via the 'output' message at each time step. This
    # is a placeholder and in real scenario you will be using any
    # sourceFinfo that sends out a double value.
    input_x = moose.StimulusTable(f'{demo.path}/xtab')
    nsteps = int(simtime / function.dt)
    xarr = np.linspace(0.0, 1.0, nsteps)
    input_x.vector = xarr
    input_x.startTime = 0.0
    input_x.stepPosition = xarr[0]
    input_x.stopTime = simtime

    # retrieve the indices of the variables
    q_idx = function.xindex['Q']
    stim_idx = function.xindex['stim']

    moose.connect(pg, 'output', function.x[q_idx], 'input')
    moose.connect(input_x, 'output', function.x[stim_idx], 'input')

    # Pulled variable
    yarr = np.linspace(-np.pi, np.pi, nsteps)
    input_y = moose.StimulusTable(f'{demo.path}/ytab')
    input_y.vector = yarr
    input_y.startTime = 0.0
    input_y.stepPosition = yarr[0]
    input_y.stopTime = simtime
    moose.connect(function, 'requestOut', input_y, 'getOutputValue')

    # data recording
    data = moose.Neutral('/data')
    value = moose.Table(f'{data.path}/value_tab')
    moose.connect(value, 'requestOut', function, 'getValue')
    derivative = moose.Table(f'{data.path}/derivative_tab')
    moose.connect(derivative, 'requestOut', function, 'getDerivative')
    rate = moose.Table(f'{data.path}/rate_tab')
    moose.connect(rate, 'requestOut', function, 'getRate')
    q_rec = moose.Table(f'{data.path}/Q_tab')
    moose.connect(q_rec, 'requestOut', pg, 'getOutputValue')
    x_rec = moose.Table(f'{data.path}/xrec')
    moose.connect(x_rec, 'requestOut', input_x, 'getOutputValue')
    y_rec = moose.Table(f'{data.path}/yrec')
    moose.connect(y_rec, 'requestOut', input_y, 'getOutputValue')

    for comp in moose.wildcardFind(f'{demo.path}/##'):
        print(comp.path, 'tick =', comp.tick, 'dt =', comp.dt)
        # Notice that by default Function objects are on a slow tick. Make it same as the inputs

    moose.setClock(function.tick, pg.dt)

    
    moose.reinit()

    moose.start(simtime)

    plt.subplot(411)
    plt.title('Variables')
    t = np.arange(len(value.vector)) * value.dt
    plt.plot(t, q_rec.vector, label='Q')
    plt.plot(t, x_rec.vector, label='stim')
    plt.plot(t, y_rec.vector, label='y0')
    plt.legend()
    plt.subplot(412)
    plt.title('Function value')
    plt.plot(t, value.vector, label=f'value = {function.expr}')
    z = q_rec.vector * np.exp(-t/function.c['tau']) + x_rec.vector + np.cos(y_rec.vector)
    plt.plot(t, z, '--', label='numpy computed')
    plt.legend()

    plt.subplot(413)
    plt.title('Derivateive wrt y0')
    plt.plot(y_rec.vector, derivative.vector, label='d(value)/dy0')
    plt.legend()

    plt.subplot(414)
    plt.title('Rate (derivative wrt t)')
    # *** BEWARE *** The first two entries are spurious. Entry 0 is
    # *** from reinit sending out the defaults. Entry 2 is because
    # *** there is no lastValue for computing real forward difference.
    plt.plot(np.arange(2, len(rate.vector), 1) * rate.dt, rate.vector[2:], label='d(value)/dt')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    function_example()



#
# function.py ends here
