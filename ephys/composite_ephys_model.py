# composite_ephys_model.py ---
#
# Filename: composite_ephys_model.py
# Description:
# Author: Subhasis Ray
# Created: Wed Jul  1 15:24:27 2026 (+0530)
#

# Code:
"""An example of creating a composite model using morphology and
channel database bundled with moose. If `ee` is passed as an argument,
then use exponential Euler method otherwise use HSolve for solving the
neuronal model.

"""
import sys

import numpy as np
import matplotlib.pyplot as plt

import moose

# Import the morphology utilities
from moose import morphologies as morph
# Import the channel database
from moose import channels as chan

# Utility functions to plot morphology
import moose.plot_utils as mplt


# ====================== Constants ================================
USE_HSOLVE = True
if len(sys.argv) > 1 and sys.argv[1] == 'ee':
    USE_HSOLVE = False

DT         = 2.5e-6      # integration timestep (s) -- small enough for fast Na
EL         = -70e-3   # leak / resting reversal (V)
G_NA       = 800.0    # NaF conductance density (S/m^2)
G_K        = 400.0    # KDR conductance density (S/m^2)
E_NA       = 50e-3    # Na reversal (V), physiological convention
E_K        = -95e-3   # K reversal (V)
SIMTIME    = 150e-3   # simulation duration (s)


# ================= Clean up if model exists =======================
if moose.exists('/model'):
    moose.delete('/model')
model = moose.Neutral('/model')
# CM = 0.01 F/m^2 (1 uF/cm^2) gives a fast membrane; the 0.03 in the
# paper tripled the membrane time constant and made the spikes too
# sluggish to fire a train.
result = morph.load('traub91_CA3', f'{model.path}/CA3_pyr', RM=1.0, CM=0.01)

# Hyperpolarised leak reversal (-70 mV).  ReadSwc leaves Em = initVm = -60 mV,
# but at -60 mV the Na window current is unopposed (the leak current is zero
# there) and the cell drifts up to a depolarised plateau instead of spiking.
# Setting EL below the Na activation range gives a stable rest and lets the
# injected current drive spikes.
for comp in moose.wildcardFind(f'{result.root.path}/##[TYPE=Compartment]'):
    comp.Em = EL
    comp.initVm = EL

mplt.plotMorphologyGraph(result.root)
plt.show()

# ============== Insert ion channels =========================
# Conductances are given as densities (S/m^2) times compartment surface area.
naf_list = chan.load(f'{model.path}/##[TYPE=Compartment]',
                     icg_id=1684,
                     gbar=lambda c: morph.surface_area(c) * 800.0,
                     Ek=50e-3)
kdr_list = chan.load(f'{model.path}/##[TYPE=Compartment]',
                     icg_id=1682,
                     gbar=lambda c: morph.surface_area(c) * 400.0,
                     Ek=-95e-3)

soma = moose.element(f'{result.root.path}/soma')
pg = moose.PulseGen(f'{model.path}/pg')
moose.connect(pg, 'output', soma, 'injectMsg')


# =============== Setup recording tables =====================
vm_tabs = []
for comp in moose.wildcardFind(f'{result.root.path}/#[TYPE=Compartment]'):
    print(comp.path)
    tab = moose.Table(f'{model.path}/Vm_{comp.name}')
    moose.connect(tab, 'requestOut', comp, 'getVm')
    vm_tabs.append(tab)

inject_tab = moose.Table(f'{model.path}/Inject_soma')
moose.connect(inject_tab, 'requestOut', pg, 'getOutputValue')

# ========= Current injection protocol ==================
pg.firstDelay = 20e-3
pg.firstWidth = 100e-3
pg.firstLevel = 0.2e-9
pg.secondDelay = 1e9

# HSolve to handle the stiff system, which requires very small
# integration time step
if USE_HSOLVE:
    solver = moose.HSolve(f'{model.path}/solver')
    solver.target = soma.path
else:
    # Electrical models need a small integration timestep.
    for ii in range(10):
        moose.setClock(ii, DT)

# ========== Initialize the model and simulate ============
moose.reinit()

simtime = 150e-3
moose.start(simtime)

# ===================== Plot data =========================
for tab in vm_tabs:
    t = np.linspace(0, simtime * 1e3, len(tab.vector))
    plt.plot(t, tab.vector * 1e3)

plt.plot(t, inject_tab.vector * 1e10 + tab.vector.min() * 1e3, label='Injected current')
plt.xlabel('Time (ms)')
plt.ylabel('Vm (mV)')
plt.legend()
plt.show()

#
# composite_ephys_model.py ends here
