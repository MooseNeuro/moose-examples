# hhcomp.py ---
#
# Filename: hhcomp.py
# Description:
# Author: Subhasis Ray
# Maintainer:
# Created: Tue May  7 12:11:22 2013 (+0530)
# Version:
# Last-Updated: Sun Apr 20 14:05:09 2025 (+0530)
#           By: Subhasis Ray
#     Update #: 342
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

import numpy as np
import matplotlib.pyplot as plt

import moose
from moose import utils

EREST_ACT = -70e-3
per_ms = 1e3


def create_na_chan(
    parent='/library', name='na', vmin=-110e-3, vmax=50e-3, vdivs=3000
):
    """
    Create a Hodgkin-Huxley Na channel under `parent`.

    vmin, vmax, vdivs: voltage range and number of divisions for gate tables

    """
    na = moose.HHChannel(f'{parent}/{name}')
    na.Xpower = 3
    na.Ypower = 1
    v = np.linspace(vmin, vmax, vdivs + 1) - EREST_ACT
    m_alpha = (
        per_ms * (25 - v * 1e3) / (10 * (np.exp((25 - v * 1e3) / 10) - 1))
    )
    m_beta = per_ms * 4 * np.exp(-v * 1e3 / 18)
    m_gate = moose.element(f'{na.path}/gateX')
    m_gate.min = vmin
    m_gate.max = vmax
    m_gate.divs = vdivs
    m_gate.tableA = m_alpha
    m_gate.tableB = m_alpha + m_beta
    h_alpha = per_ms * 0.07 * np.exp(-v / 20e-3)
    h_beta = per_ms * 1 / (np.exp((30e-3 - v) / 10e-3) + 1)
    h_gate = moose.element(f'{na.path}/gateY')
    h_gate.min = vmin
    h_gate.max = vmax
    h_gate.divs = vdivs
    h_gate.tableA = h_alpha
    h_gate.tableB = h_alpha + h_beta
    plt.subplot(2, 1, 1)
    v += EREST_ACT
    plt.plot(v, m_alpha / (m_alpha + m_beta), label='m_inf')
    plt.plot(v, h_alpha / (h_alpha + h_beta), label='h_inf')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(v, 1 / (m_alpha + m_beta), label='tau_m')
    plt.plot(v, 1 / (h_alpha + h_beta), label='tau_h')
    plt.legend()
    plt.show()
    plt.close()
    if parent == 'library':
        na.tick = -1
    return na


def create_k_chan(
    parent='/library', name='k', vmin=-120e-3, vmax=40e-3, vdivs=3000
):
    """Create a Hodgkin-Huxley K channel under `parent`.

    vmin, vmax, vdivs: voltage range and number of divisions for gate tables

    """
    k = moose.HHChannel(f'{parent}/{name}')
    k.Xpower = 4
    v = np.linspace(vmin, vmax, vdivs + 1) - EREST_ACT
    n_alpha = (
        per_ms * (10 - v * 1e3) / (100 * (np.exp((10 - v * 1e3) / 10) - 1))
    )
    n_beta = per_ms * 0.125 * np.exp(-v * 1e3 / 80)
    n_gate = moose.element(f'{k.path}/gateX')
    n_gate.min = vmin
    n_gate.max = vmax
    n_gate.divs = vdivs
    n_gate.tableA = n_alpha
    n_gate.tableB = n_alpha + n_beta
    plt.subplot(211)
    plt.plot(v, n_alpha / (n_alpha + n_beta))
    plt.subplot(212)
    plt.plot(v, 1 / (n_alpha + n_beta))
    plt.show()
    plt.close()
    if parent == '/library':
        k.tick = -1
    return k


def test_channel_gates():
    """Creates prototype channels under `/library` and plots the time
    constants (tau) and activation (minf, hinf, ninf) parameters for the
    channel gates.

    Does not execute any simulation.

    """
    lib = moose.Neutral('/library')
    na_proto = create_na_chan(parent=lib.path)
    k_proto = create_k_chan(parent=lib.path)
    m = moose.element(f'{na_proto.path}/gateX')
    h = moose.element(f'{na_proto.path}/gateY')
    n = moose.element(f'{k_proto.path}/gateX')
    v = np.linspace(m.min, m.max, m.divs + 1)
    plt.subplot(211)
    plt.plot(v, 1 / m.tableB, label='tau_m')
    plt.plot(v, 1 / h.tableB, label='tau_h')
    plt.plot(v, 1 / n.tableB, label='tau_n')
    plt.legend()
    plt.subplot(212)
    plt.plot(v, m.tableA / m.tableB, label='m_inf')
    plt.plot(v, h.tableA / h.tableB, label='h_inf')
    plt.plot(v, n.tableA / n.tableB, label='n_inf')
    plt.legend()
    plt.show()


def create_passive_comp(
    parent='/library', name='comp', diameter=30e-6, length=0.0
):
    """Creates a single compartment with squid axon Em, Cm and Rm. Does
    not set Ra"""
    comp = moose.Compartment('%s/%s' % (parent, name))
    comp.Em = EREST_ACT + 10.613e-3
    comp.initVm = EREST_ACT
    if length <= 0:
        sarea = np.pi * diameter * diameter
    else:
        sarea = np.pi * diameter * length
    # specific conductance gm = 0.3 mS/cm^2
    comp.Rm = 1 / (0.3e-3 * sarea * 1e4)
    # Specific capacitance cm = 1 uF/cm^2
    comp.Cm = 1e-6 * sarea * 1e4
    return comp, sarea


def create_hhcomp(
    parent='/library', name='hhcomp', diameter=30e-6, length=0.0
):
    """Create a compartment with Hodgkin-Huxley type ion channels (Na and
    K).

    Returns a 3-tuple: (compartment, nachannel, kchannel)

    """
    comp, sarea = create_passive_comp(parent, name, diameter, length)
    if moose.exists(f'/library/na'):
        na = moose.element(moose.copy('/library/na', comp.path, 'na'))
    else:
        na = create_na_chan(parent=comp.path)
    # Na-conductance 120 mS/cm^2
    na.Gbar = 120e-3 * sarea * 1e4
    na.Ek = 115e-3 + EREST_ACT
    moose.connect(comp, 'channel', na, 'channel')
    if moose.exists('/library/k'):
        k = moose.element(moose.copy('/library/k', comp.path, 'k'))
    else:
        k = create_k_chan(parent=comp.path)
    # K-conductance 36 mS/cm^2
    k.Gbar = 36e-3 * sarea * 1e4
    k.Ek = -12e-3 + EREST_ACT
    moose.connect(comp, 'channel', k, 'channel')
    return comp, na, k


def test_hhcomp():
    """Create and simulate a single spherical compartment with
    Hodgkin-Huxley Na and K channel.

    Plots Vm, injected current, channel conductances.

    """
    model = moose.Neutral('/model')
    data = moose.Neutral('/data')
    comp, na, k = create_hhcomp(parent=model.path)
    print(
        'Setup compartment with:' f'Rm={comp.Rm}\n' f'Cm={comp.Cm}\n',
        f'EK(Na)={na.Ek}\n'
        f'Gbar(Na)={na.Gbar}\n'
        f'EK(K)={k.Ek}\n'
        f'Gbar(K)={k.Gbar}',
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
    moose.connect(gK, 'requestOut', k, 'getGk')
    gNa = moose.Table(f'{data.path}/gNa')
    moose.connect(gNa, 'requestOut', na, 'getGk')
    simtime = 100e-3
    moose.reinit()
    moose.start(simtime)
    t = np.arange(len(vm.vector)) * vm.dt * 1e3
    plt.subplot(211)
    plt.plot(t, vm.vector * 1e3, label='Vm (mV)')
    plt.plot(t, inj.vector * 1e9, label='injected (nA)')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.title('Vm')
    plt.subplot(212)
    plt.title('Conductance (uS)')
    plt.plot(t, gK.vector * 1e6, label='K')
    plt.plot(t, gNa.vector * 1e6, label='Na')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.show()


def main():
    """A compartment with hodgkin-huxley ion channels"""
    test_channel_gates()
    test_hhcomp()


if __name__ == '__main__':
    main()
#
# hhcomp.py ends here
