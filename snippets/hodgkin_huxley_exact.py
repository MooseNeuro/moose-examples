# hodgkin_huxley_exact.py ---
#
# Filename: hodgkin_huxley_exact.py
# Description:
# Author: Subhasis Ray
# Maintainer:
# Created: Tue May  7 12:11:22 2013 (+0530)
# Version:
# Last-Updated: Fri Mar 28 11:55:07 2025 (+0530)
#           By: Subhasis Ray
#     Update #: 428
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

"""This reimplements the Hodgkin-Huxley squid giant axon model using
formula-based channel (HHChannelF) as opposed to the standard
HHChannel which uses table-lookup for computing the m, h, and n
parameters. Formula evaluation is slower but more accurate. The
difference in accuracy is not significant in case of simple
voltage-gated channels. However, for Ca dependent channels, if the [Ca2+]
changes logarithmically, table lookup may not be accurate enough.

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import moose
from moose import utils


EREST_ACT = -70e-3


def create_na_chan(parent='/library', name='na'):
    """
    Create a Hodgkin-Huxley Na channel under `parent`.

    """
    na = moose.HHChannelF('%s/%s' % (parent, name))
    na.Xpower = 3
    na.Ypower = 1
    m_gate = moose.element(f'{na.path}/gateX')
    # the multiplications with 1e3 are for converting from
    # physiological units (ms, mV) to SI units (s, V)
    m_gate.alphaExpr = (
        f'1e3 * 0.1 * (25 - 1e3 * (v - ({EREST_ACT}))) /'
        f'(exp((25 - 1e3 * (v - ({EREST_ACT}))) / 10) - 1)'
    )
    m_gate.betaExpr = f'1e3 * 4 * exp(- (v - ({EREST_ACT}))/ 18e-3)'
    h_gate = moose.element(f'{na.path}/gateY')
    h_gate.alphaExpr = f'1e3 * 0.07 * exp(- 1e3 * (v - ({EREST_ACT}))/ 20)'
    h_gate.betaExpr = f'1e3 / (exp((30 - 1e3 * (v - ({EREST_ACT}))) / 10) + 1)'
    return na


def create_k_chan(parent='/library', name='k'):
    """Create a Hodgkin-Huxley K channel under `parent`."""
    k = moose.HHChannelF(f'{parent}/{name}')
    k.Xpower = 4
    n_gate = moose.element(f'{k.path}/gateX')
    n_gate.alphaExpr = (
        f'1e3 * 0.01 * (10 - 1e3 * (v - ({EREST_ACT}))) /'
        f'(exp((10 - 1e3 * (v - ({EREST_ACT})))/10) - 1)'
    )
    n_gate.betaExpr = f'1e3 * 0.125 * exp(-1e3 * (v - ({EREST_ACT})) / 80)'
    return k


def create_passive_comp(
    parent='/library', name='comp', diameter=30e-6, length=0.0
):
    """Creates a single compartment with squid axon Em, Cm and
    Rm. Does not set Ra

    """
    comp = moose.Compartment('%s/%s' % (parent, name))
    comp.Em = EREST_ACT + 10.613e-3
    comp.initVm = EREST_ACT
    if length <= 0:
        sarea = np.pi * diameter * diameter
    else:
        sarea = np.pi * diameter * length
    # specific conductance gm = 0.3 mS/cm^2, convert to SI
    comp.Rm = 1 / (0.3e-3 * sarea * 1e4)
    # Specific capacitance cm = 1 uF/cm^2, convert to SI
    comp.Cm = 1e-6 * sarea * 1e4
    return comp, sarea


def create_hhcomp(
    parent='/library', name='hhcomp', diameter=-30e-6, length=0.0
):
    """Create a compartment with Hodgkin-Huxley type ion channels (Na
    and K).

    Returns a 3-tuple: (compartment, nachannel, kchannel)

    """
    comp, sarea = create_passive_comp(parent, name, diameter, length)
    if moose.exists('/library/na'):
        na_chan = moose.element(moose.copy('/library/na', comp.path, 'na'))
    else:
        na_chan = create_na_chan(parent=comp.path)
    # Na-conductance 120 mS/cm^2
    na_chan.Gbar = 120e-3 * sarea * 1e4
    na_chan.Ek = 115e-3 + EREST_ACT
    moose.connect(comp, 'channel', na_chan, 'channel')
    if moose.exists('/library/k'):
        k_chan = moose.element(moose.copy('/library/k', comp.path, 'k'))
    else:
        k_chan = create_k_chan(parent=comp.path)
    # K-conductance 36 mS/cm^2
    k_chan.Gbar = 36e-3 * sarea * 1e4
    k_chan.Ek = -12e-3 + EREST_ACT
    moose.connect(comp, 'channel', k_chan, 'channel')
    return comp, na_chan, k_chan


def simulate_hhcomp():
    """Create and simulate a single spherical compartment with
    Hodgkin-Huxley Na and K channel.

    Plots Vm, injected current, channel conductances.

    """
    model = moose.Neutral('/model')
    data = moose.Neutral('/data')
    comp, na_chan, k_chan = create_hhcomp(parent=model.path)
    print(
        'Setup compartment with:' f'Rm={comp.Rm}\n' f'Cm={comp.Cm}\n',
        f'EK(Na)={na_chan.Ek}\n'
        f'Gbar(Na)={na_chan.Gbar}\n'
        f'EK(K)={k_chan.Ek}\n'
        f'Gbar(K)={k_chan.Gbar}',
    )

    pg = moose.PulseGen('%s/pg' % (model.path))
    pg.firstDelay = 20e-3
    pg.firstWidth = 40e-3
    pg.firstLevel = 1e-9
    pg.secondDelay = 1e9
    moose.connect(pg, 'output', comp, 'injectMsg')
    inj = moose.Table(f'{data.path}/pulse')
    moose.connect(inj, 'requestOut', pg, 'getOutputValue')
    vm = moose.Table(f'{data.path}/Vm')
    moose.connect(vm, 'requestOut', comp, 'getVm')
    gK = moose.Table(f'{data.path}/gK')
    moose.connect(gK, 'requestOut', k_chan, 'getGk')
    gNa = moose.Table(f'{data.path}/gNa')
    moose.connect(gNa, 'requestOut', na_chan, 'getGk')
    simtime = 100e-3
    moose.reinit()
    moose.start(simtime)
    t = np.arange(len(vm.vector)) * vm.dt * 1e3  # s to ms
    plt.subplot(211)
    plt.plot(t, vm.vector * 1e3, label='Vm (mV)')
    plt.plot(t, inj.vector * 1e9, label='injected (nA)')
    plt.legend()
    plt.title('Vm')
    plt.subplot(212)
    plt.title('Conductance (uS)')
    plt.plot(t, gK.vector * 1e6, label='K')
    plt.plot(t, gNa.vector * 1e6, label='Na')
    plt.legend()
    plt.show()
    plt.close()


def main():
    """A compartment with hodgkin-huxley ion channels"""
    simulate_hhcomp()


if __name__ == '__main__':
    main()
#
# hodgkin_huxley_exact.py ends here
