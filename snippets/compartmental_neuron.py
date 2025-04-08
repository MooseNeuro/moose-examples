# compartmental_neuron.py ---
#
# Filename: compartmental_neuron.py
# Description:
# Author:Subhasis Ray
# Maintainer:
# Created: Tue Aug  7 10:27:26 2012 (+0530)
# Version:
# Last-Updated: Wed Jan 15 15:07:08 2025 (+0530)
#           By: Subhasis Ray
#     Update #: 133
# URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
# Demostrates how to connect two compartments to form a simple neuron.
# All units are in SI
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
"""
Setting up a compartmental neuron from scratch using MOOSE classes.

A simple electrophysiological simulation of a neuron composed of two
passive compartments. A current pulse is injected into the `soma`
compartment and the membrane voltage in both compartments are recorded
and saved into a csv file.

"""

import numpy as np
import matplotlib.pyplot as plt

import moose

# Create the somatic compartment
model = moose.Neutral('/model')  # This is a container for the model
soma = moose.Compartment('/model/soma')
soma.Em = -65e-3  # Leak potential
soma.initVm = -65e-3  # Initial membrane potential
soma.Rm = 5e9  # Total membrane resistance of the compartment
soma.Cm = 1e-12  # Total membrane capacitance of the compartment
soma.Ra = 1e8  # Total axial resistance of the compartment
# Create the axon
axon = moose.Compartment('/model/axon')
axon.Em = -65e-3
axon.initVm = -65e-3
axon.Rm = 2.5e9
axon.Cm = 2e-12
axon.Ra = 2e5
# Connect the soma to the axon. Note the order of raxial-axial
# connection decides which Ra is going to be used in the computation.
# `raxial` message sends the Ra and Vm of the source to destionation,
# `axial` gets back the Vm of the destination. Try:
# moose.doc('Compartment.axial') in python interpreter for details.
moose.connect(soma, 'raxial', axon, 'axial')

# Setup data recording
data = moose.Neutral('/data')
axon_Vm = moose.Table('/data/axon')
soma_Vm = moose.Table('/data/soma')
moose.connect(axon_Vm, 'requestOut', axon, 'getVm')
moose.connect(soma_Vm, 'requestOut', soma, 'getVm')

# #######################################################################
#                            SCHEDULING START
# Explicitly scheduling the sequence of operations and time resolutions.
# As moose now automates the scheduling, this is not necessary unless
# you need fine control over scheduling
moose.setClock(0, 0.025e-3)
moose.setClock(1, 0.025e-3)
moose.setClock(2, 0.25e-3)

# useClock(clockNo, path, method)
# In this case:
# clockNo: int
#     clock no. (nonnegative integer) to use
# path: str
#     wildcard path matching all elements of type Compartment
#     or explicit path of the tables
# method: str
#     the processing function to be executed at each tick of clock

moose.useClock(0, '/model/#[TYPE=Compartment]', 'init')
moose.useClock(1, '/model/#[TYPE=Compartment]', 'process')
moose.useClock(2, axon_Vm.path, 'process')
moose.useClock(2, soma_Vm.path, 'process')
#                            SCHEDULING END
# #######################################################################

# Now initialize everything and get set
moose.reinit()

# Here we take a simple approach to try current injection experiemnt:
# Run the simulation for some time. Then set the `inject` field of the
# soma to a positive value and run for some more time. This will
# emulate a current injection for that duration. Finally, remove the
# current injection by setting the `inject` field back to 0 and run
# for some more time to get the discharging curve.

# Normally you should use a PulseGen object for this instead of
# stopping the simulation midway to set the inject field..
moose.start(50e-3)
# Now apply 1 pA current injection to soma
soma.inject = 1e-12
# Run for 100 ms
moose.start(100e-3)
# Stop the current injection
soma.inject = 0.0
# Run for 500 ms
moose.start(500e-3)


## This is old style of creating time points
# clock = moose.Clock('/clock')  # Get a handle to the global clock
# time = np.linspace(0, clock.currentTime, len(axon_Vm.vector))

## You can create the time points using the dt of the clock tick
## associated with a table
time = np.arange(len(axon_Vm.vector)) * axon_Vm.dt
data = np.vstack((time, axon_Vm.vector, soma_Vm.vector))

np.savetxt(
    'compartmental_neuron.csv', data.T, delimiter=',', header='time,axonVm,somaVm'
)
print('Saved data in compartmental_neuron.csv')

###########################################################
# Plot the membrane voltages
###########################################################
plt.plot(time * 1e3, soma_Vm.vector * 1e3, label=soma_Vm.name)
plt.plot(time * 1e3, axon_Vm.vector * 1e3, '--', label=axon_Vm.name)
plt.ylabel('Membrane potential (mV)')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()

#
# compartmental_neuron.py ends here
