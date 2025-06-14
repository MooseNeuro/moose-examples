# hh_syn_sequence.py ---
#
# Filename: hh_syn_sequence.py
# Description:
# Author: Subhasis Ray
# Created: Thu Jun 12 16:15:04 2025 (+0530)
#

# Code:
"""This code creates 3 single compartmental HH models and connects
them via double exponential synapses. The first neuron is stimulated
by a pulsegen"""


import numpy as np
import matplotlib.pyplot as plt

import moose

EREST_ACT = -70e-3
per_ms = 1e3


def create_na_chan(
    parent='/library', name='na', vmin=-110e-3, vmax=50e-3, vdivs=3000
):
    """
    Create a Hodgkin-Huxley Na channel under `parent`.

    vmin, vmax, vdivs: voltage range and number of divisions for gate tables

    """
    na = moose.HHChannel('%s/%s' % (parent, name))
    na.Xpower = 3
    na.Ypower = 1
    m_gate = moose.element('%s/gateX' % (na.path))
    m_gate.alphaExpr = (
        f'1e3 * 0.1 * (25 - 1e3 * (v - ({EREST_ACT}))) /'
        f'(exp((25 - 1e3 * (v - ({EREST_ACT}))) / 10) - 1)'
    )
    v = np.linspace(vmin, vmax, vdivs + 1)
    alpha = (
        1e3
        * 0.1
        * (25 - 1e3 * (v - EREST_ACT))
        / (np.exp((25 - 1e3 * (v - EREST_ACT)) / 10) - 1)
    )
    m_gate.betaExpr = f'1e3 * 4 * exp(- (v - ({EREST_ACT}))/ 18e-3)'
    beta = 1e3 * 4 * np.exp(-1e3 * (v - EREST_ACT) / 18)
    h_gate = moose.element('%s/gateY' % (na.path))
    h_gate.alphaExpr = f'1e3 * 0.07 * exp(- 1e3 * (v - ({EREST_ACT}))/ 20)'
    h_gate.betaExpr = f'1e3 / (exp((30 - 1e3 * (v - ({EREST_ACT}))) / 10) + 1)'
    for gate in (m_gate, h_gate):
        gate.min = vmin
        gate.max = vmax
        gate.divs = vdivs
        gate.useInterpolation = True
        gate.fillFromExpr()
    return na


def create_k_chan(
    parent='/library', name='k', vmin=-120e-3, vmax=40e-3, vdivs=3000
):
    """Create a Hodgkin-Huxley K channel under `parent`.

    vmin, vmax, vdivs: voltage range and number of divisions for gate tables

    """
    k = moose.HHChannel('%s/%s' % (parent, name))
    k.Xpower = 4
    n_gate = moose.element('%s/gateX' % (k.path))
    n_gate.alphaExpr = (
        f'1e3 * 0.01 * (10 - 1e3 * (v - ({EREST_ACT}))) /'
        f'(exp((10 - 1e3 * (v - ({EREST_ACT})))/10) - 1)'
    )
    n_gate.betaExpr = f'1e3 * 0.125 * exp(-1e3 * (v - ({EREST_ACT})) / 80)'
    n_gate.min = -100e-3
    n_gate.max = 100e-3
    n_gate.divs = 1000
    n_gate.useInterpolation = True
    n_gate.fillFromExpr()
    v = np.linspace(n_gate.min, n_gate.max, n_gate.divs + 1)
    alpha = (
        1e3
        * 0.01
        * (10 - 1e3 * (v - (EREST_ACT)))
        / (np.exp((10 - 1e3 * (v - (EREST_ACT))) / 10) - 1)
    )
    beta = 1e3 * 0.125 * np.exp(-1e3 * (v - (EREST_ACT)) / 80)
    return k


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
    parent='/library', name='hhcomp', diameter=-30e-6, length=0.0
):
    """Create a compartment with Hodgkin-Huxley type ion channels (Na and
    K).

    Returns a 3-tuple: (compartment, nachannel, kchannel)

    """
    comp, sarea = create_passive_comp(parent, name, diameter, length)
    if moose.exists('/library/na'):
        moose.copy('/library/na', comp.path, 'na')
    else:
        create_na_chan(parent=comp.path)
    na = moose.element('%s/na' % (comp.path))
    # Na-conductance 120 mS/cm^2
    na.Gbar = 120e-3 * sarea * 1e4
    na.Ek = 115e-3 + EREST_ACT
    moose.connect(comp, 'channel', na, 'channel')
    if moose.exists('/library/k'):
        moose.copy('/library/k', comp.path, 'k')
    else:
        create_k_chan(parent=comp.path)
    k = moose.element('%s/k' % (comp.path))
    # K-conductance 36 mS/cm^2
    k.Gbar = 36e-3 * sarea * 1e4
    k.Ek = -12e-3 + EREST_ACT
    moose.connect(comp, 'channel', k, 'channel')
    return comp, na, k


def create_proto(syn_weight=1.0):
    lib = moose.Neutral('/library')
    proto, nachan, kchan = create_hhcomp(lib.path)
    spikegen = moose.SpikeGen(f'{proto.path}/spike')
    synchan = moose.SynChan(f'{proto.path}/synchan')
    synhandler = moose.SimpleSynHandler(f'{proto.path}/synhandler')
    synchan.Gbar = 1e-7
    synchan.Ek = 0.0
    synchan.tau1 = 1e-3
    synchan.tau2 = 5e-3
    synhandler.numSynapses = 1
    synhandler.synapse[0].weight = syn_weight
    synhandler.synapse[0].delay = 1e-3
    moose.connect(proto, 'VmOut', spikegen, 'Vm')
    moose.connect(synchan, 'channel', proto, 'channel')
    moose.connect(synhandler, 'activationOut', synchan, 'activation')
    # Disable clock ticks on elements in library to prevent their
    # simulation
    for item in moose.wildcardFind(f'{lib.path}/##'):
        print(f'Disabling clock for {item.path}')
        item.tick = -1
    return {
        'comp': proto,
        'spikegen': spikegen,
        'synchan': synchan,
        'synhandler': synhandler,
    }


def create_model(mroot='/model', droot='/data', nn=3, syn_weight=1):
    if moose.exists(mroot):
        moose.delete(mroot)
        moose.delete(droot)
    model = moose.Neutral(mroot)
    data = moose.Neutral(droot)
    proto_dict = create_proto(syn_weight=syn_weight)
    comps = []
    vm_tabs = []
    gk_tabs = []
    for ii in range(nn):
        comp = moose.copy(proto_dict['comp'], model, f'comp_{ii}')
        vm_tab = moose.Table(f'{data.path}/Vm_{comp.name}')
        moose.connect(vm_tab, 'requestOut', comp, 'getVm')
        gk_tab = moose.Table(f'{data.path}/Gk_{comp.name}')
        moose.connect(
            gk_tab,
            'requestOut',
            moose.element(f'{comp.path}/synchan'),
            'getGk',
        )
        comps.append(comp)
        vm_tabs.append(vm_tab)
        gk_tabs.append(gk_tab)

    for ii in range(1, nn):
        pre = comps[ii - 1]
        post = comps[ii]
        spikegen = moose.element(f'{pre.path}/spike')
        synhandler = moose.element(f'{post.path}/synhandler')
        moose.connect(spikegen, 'spikeOut', synhandler.synapse[0], 'addSpike')
    pulse = moose.PulseGen(f'{mroot}/pulse')
    pulse.firstDelay = 5e-3
    pulse.firstWidth = 40e-3
    pulse.firstLevel = 1e-9
    pulse.secondDelay = 1e9
    moose.connect(pulse, 'output', comps[0], 'injectMsg')
    return {'comps': comps, 'vm_tabs': vm_tabs, 'gk_tabs': gk_tabs}


def test_main(simtime=50e-3, nn=3, syn_weight=1):
    mdict = create_model(nn=nn, syn_weight=syn_weight)
    moose.reinit()
    moose.start(simtime)
    plt.subplot(211)
    for vm_tab in mdict['vm_tabs']:
        vm = vm_tab.vector
        t = np.arange(len(vm_tab.vector)) * vm_tab.dt
        plt.plot(t * 1e3, vm * 1e3, label=vm_tab.name)
    plt.legend()
    plt.ylabel('Vm (mV)')
    plt.subplot(212)
    plt.title(f'weight={syn_weight}')
    for gk_tab in mdict['gk_tabs']:
        print(gk_tab.name, gk_tab.vector)
        t = np.arange(len(gk_tab.vector))
        plt.plot(
            t * 1e3, gk_tab.vector * 1e6, label=f'{gk_tab.name}'
        )
    plt.legend()
    plt.ylabel('Gk (uS)')
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig(f'Vm_gk_weight{syn_weight:0.2f}.png')
    plt.show()


if __name__ == '__main__':
    test_main(syn_weight=1)


#
# hh_syn_sequence.py ends here
