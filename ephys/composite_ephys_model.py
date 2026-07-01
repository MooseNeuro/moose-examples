# composite_ephys_model.py ---
#
# Filename: composite_ephys_model.py
# Description:
# Author: Subhasis Ray
# Created: Wed Jul  1 15:24:27 2026 (+0530)
#

# Code:
"""An example of creating a composite model using morphology and
channel database bundled with moose"""


import numpy as np
import matplotlib.pyplot as plt

import moose

# Import the morphology utilities
from moose import morphologies as morph
# Import the channel database
from moose import channels as chan

# Utility functions to plot morphology
import moose.plot_utils as mplt


if moose.exists('/model'):
    moose.delete('/model')

model = moose.Neutral('/model')
# CM = 0.01 F/m^2 (1 uF/cm^2) gives a fast membrane; the earlier 0.03 tripled
# the membrane time constant and made the spikes too sluggish to fire a train.
result = morph.load('traub91_CA3', f'{model.path}/CA3_pyr', RM=1.0, CM=0.01)

# Hyperpolarised leak reversal (-70 mV).  ReadSwc leaves Em = initVm = -60 mV,
# but at -60 mV the Na window current is unopposed (the leak current is zero
# there) and the cell drifts up to a depolarised plateau instead of spiking.
# Setting EL below the Na activation range gives a stable rest and lets the
# injected current drive spikes.
EL = -70e-3
for comp in moose.wildcardFind(f'{result.root.path}/##[TYPE=Compartment]'):
    comp.Em = EL
    comp.initVm = EL

mplt.plotMorphologyGraph(result.root)
plt.show()

# Conductances are given as densities (S/m^2) times compartment surface area.
#
# gNa = 800 S/m^2 (not higher): the Na "window" current is inward even at rest,
# so with the weak leak (RM=1) a large gNa lets Vm creep to threshold and the
# cell fires spontaneously with no injection.  800 keeps a stable ~-67 mV rest
# while still firing a train on injection.
naf_list = chan.load(f'{model.path}/##[TYPE=Compartment]',
                     icg_id=1684,
                     gbar=lambda c: morph.surface_area(c) * 800.0,
                     Ek=50e-3)
kdr_list = chan.load(f'{model.path}/##[TYPE=Compartment]',
                     icg_id=1682,
                     gbar=lambda c: morph.surface_area(c) * 400.0,
                     Ek=-95e-3)

pg = moose.PulseGen(f'{model.path}/pg')

soma = moose.element(f'{result.root.path}/soma')
moose.connect(pg, 'output', soma, 'injectMsg')



vmtabs = []
for comp in moose.wildcardFind(f'{result.root.path}/#[TYPE=Compartment]'):
    print(comp.path)
    tab = moose.Table(f'{model.path}/Vm_{comp.name}')
    moose.connect(tab, 'requestOut', comp, 'getVm')
    vmtabs.append(tab)


pg.firstDelay = 20e-3
pg.firstWidth = 100e-3
pg.firstLevel = 0.2e-9
pg.secondDelay = 1e9

# HH channels need a small integration timestep.
for i in range(10):
    moose.setClock(i, 2.5e-6)

moose.reinit()

simtime = 150e-3
moose.start(simtime)

for tab in vmtabs:
    t = np.linspace(0, simtime, len(tab.vector))
    plt.plot(t, tab.vector)
    break

plt.show()
#
# composite_ephys_model.py ends here
