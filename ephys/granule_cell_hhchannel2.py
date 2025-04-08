# granule_deschutter.py ---
#
# Filename: granule_cell_hhchannelf.py
# Description:
# Author: Subhasis Ray
# Created: Fri Feb 28 14:49:44 2025 (+0530)
#

# Code:
"""Implementation of the cerebeller granule cell model from Maex, R
and De Schutter, E. Synchronization of Golgi and Granule Cell Firing
in a Detailed Network Model of the Cerebellar Granule Cell Layer,
1998.

This example demonstrates how to explicitly create a model with
Hodgkin-Huxley type ion channels that depend on voltage as well as
calcium concentration. Here we are using `HHChannelF2D` class which
produces more accurate results for 2D gates by explicitly evaluating
the formulae for the gate parameters.

It is easier to load a predefined model in a standard format like
NeuroML.

"""
import numpy as np
import moose
from matplotlib import pyplot as plt


TEMP_0 = 17.35  # model specification temperature
TEMP_1 = 32.0  # simulation temperature
Q10 = 3.0

Q10_MUL = Q10 ** (0.1 * (TEMP_1 - TEMP_0))

vmin = -150e-3
vmax = 100e-3
vdivs = 1000

def make_channel_CaHVA(parent='/library', name='CaHVA'):
    """Create a High-Voltage Activated (called CaL for Long lasting)
    Ca channel from Maex and DeSchutter 1998"""
    chan = moose.HHChannel(f'{parent}/{name}')
    chan.Xpower = 2
    chan.Ypower = 1
    chan.Ek = 80e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    alpha = '1600/(1 + exp(-72*((v -10e-3) - 5e-3)))'
    mgate.alphaExpr = f'~(alpha:={alpha}, {Q10_MUL} * alpha)'
    beta = (
        '100 * ((v - 10e-3) - (-8.9e-3))/(-0.005) /'
        ' (1 - exp(200 * ((v - 10e-3) - (-8.9e-3))/(-0.005)))'
    )
    mgate.betaExpr = f'~(beta:={beta}, {Q10_MUL} * beta)'
    alpha = '(v - 10e-3) < -0.060? 5.0: 5 * exp(-50 * ((v - 10e-3) - (-0.06)))'
    hgate.alphaExpr = f'~(alpha:={alpha},{Q10_MUL} * alpha)'
    beta = (
        '(v - 10e-3) < -0.060? 0.0:'
        ' (5 - 5 * exp(-50 * ((v - 10e-3) - (-0.060))))'
    )
    hgate.betaExpr = f'~(beta:={beta},{Q10_MUL} * beta)'
    return chan


def make_channel_H(parent='/library', name='H'):
    chan = moose.HHChannel(f'{parent}/{name}')
    chan.Xpower = 1
    chan.Ek = -42e-3
    mgate = moose.element(f'{chan.path}/gateX')
    # OSB version uses 0.8, but 4 in paper
    alpha = '0.8 * exp(-90.9 * ((v - 10e-3) -(-75e-3)))'
    mgate.alphaExpr = f'~(alpha:={alpha},{Q10_MUL} * alpha)'
    beta = '0.8 * exp(90.9 * ((v - 10e-3) - (-75e-3)))'
    mgate.betaExpr = f'~(beta:={beta},{Q10_MUL} * beta)'
    return chan


def make_channel_KA(parent='/library', name='KA'):
    """Create an A-type K channel from Maex and DeSchutter 1998"""
    chan = moose.HHChannel(f'{parent}/{name}')
    chan.Xpower = 3
    chan.Ypower = 1
    chan.Ek = -90e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    tau = '0.41e-3 * exp(-((v - 10e-3) - (-43.5e-3))/0.0428) + 0.167e-3'
    mgate.tauExpr = f'~(tau:={tau}, {Q10_MUL} * tau)'
    mgate.infExpr = '1 / (1 + exp(-((v - 10e-3) - (-46.7e-3))/19.8))'
    tau = (
        '0.001 * (10.8 + 30 * (v - 10e-3) +'
        ' 1 / (57.9 * exp((v - 10e-3) * 127) +'
        ' 134e-6 * exp(-59 * (v - 10e-3))))'
    )
    hgate.tauExpr = f'~(tau:={tau}, {Q10_MUL} * tau)'
    hgate.infExpr = '1 / (1 + exp(-((v - 10e-3) - (-78.8e-3))/0.0084))'
    return chan


def make_channel_KCa(parent='/library', name='KCa'):
    """Create Ca dependent K channel (KC in paper)"""
    chan = moose.HHChannel2D(f'{parent}/{name}')
    chan.Xpower = 1
    chan.Xindex = 'VOLT_C1_INDEX'
    chan.Ek = -90e-3
    mgate = moose.element(f'{chan.path}/gateX')
    alpha = '2500 / ( 1 + 1.5e-3 * exp(-85 * (v - 10e-3)) / c)'
    mgate.alphaExpr = f'~(alpha:={alpha}, {Q10_MUL} * alpha)'  # Why 2500 instead of 1250 in OSB/NeuroML2
    beta = '1500 / (1 + c / (1.5e-4 * exp(-77 * (v - 10e-3))))'
    mgate.betaExpr = f'~(beta:={beta}, {Q10_MUL} * beta)'
    return chan


def make_channel_KDR(parent='/library', name='KDR'):
    """Create delayed rectifier type K channel"""
    chan = moose.HHChannel(f'{parent}/{name}')
    chan.Xpower = 4
    chan.Ypower = 1
    chan.Ek = -90e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    alpha = '170 * exp(73 * ((v - 10e-3) - (-38e-3)))'
    mgate.alphaExpr = f'~(alpha:= alpha, {Q10_MUL} * alpha)'
    # KDR_m_beta: in NeuroML A is 170, in paper 0.85e3, in GENESIS code they load the table from a data file
    beta = '170 * exp(-18 * ((v - 10e-3) - (-38e-3)))'
    mgate.betaExpr = f'~(beta:={beta}, {Q10_MUL} * beta)'

    alpha = '(v - 10e-3) > -0.046? 0.76: 0.7 + 0.065 * exp(-80 * (v - 10e-3) - (-46e-3))'
    hgate.alphaExpr = f'~(alpha:={alpha}, {Q10_MUL} * alpha)'
    beta = '1.1/(1 + exp(-80.7 * ((v - 10e-3) - (-0.044))))'  # numerator A=5.5 in paper
    hgate.betaExpr = f'~(beta:={beta}, {Q10_MUL} * beta)'
    return chan


def make_channel_NaF(parent='/library', name='NaF'):
    """Create fast Na channel"""
    chan = moose.HHChannel(f'{parent}/{name}')
    chan.Xpower = 3
    chan.Ypower = 1
    chan.Ek = 55e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    # These are based on NeuroML - for some reason the A parameter is
    # double that in the paper
    # Note: I am using 0.01 ms and 0.045 ms as the lower limits for
    # tau_m and tau_h as described in the paper. The Q10 multiplier is
    # incorporated in alpha and beta already. In contrast OSB/NeuroML
    # model puts keeps the limits scaled down by Q10 multiplier so
    # that the final result remains the same
    alpha = f'{Q10_MUL} * 1500 * exp(81 *((v - 10e-3) - (-39e-3)))'
    beta = f'{Q10_MUL} * 1500 * exp(-66 * ((v - 10e-3) - (-39e-3)))'
    mgate.tauExpr = (
        f'~(alpha:={alpha}, beta:={beta},'
        f' tau:= alpha + beta == 0? 0: 1/(alpha + beta), max(tau, 1e-5))'
    )
    mgate.infExpr = f'~(alpha:={alpha}, beta:={beta}, alpha/(alpha+beta))'
    alpha = f'{Q10_MUL} * 120*exp(-89 * ((v - 10e-3) - (-0.05)))'
    beta = f'{Q10_MUL} * 120 * exp(89 * ((v - 10e-3) - (-0.05)))'
    hgate.tauExpr = (
        f'~(alpha:={alpha}, beta:={beta},'
        f' tau:= alpha + beta == 0? 0: 1/(alpha+beta), max(tau, 0.045e-3))'
    )
    hgate.infExpr = f'~(alpha:={alpha}, beta:={beta}, alpha/(alpha+beta))'
    moose.showfields(chan)
    moose.showfields(hgate)
    moose.showfields(mgate)
    for gate in (mgate, hgate):
        gate.min = vmin
        gate.max = vmax
        gate.divs = vdivs
        gate.useInterpolation = True
        gate.tabFillExpr()
        np.savetxt(f'{chan.name}.{gate.name}.A.dat', np.c_[np.linspace(vmin, vmax, vdivs+1), gate.tableA])
        np.savetxt(f'{chan.name}.{gate.name}.B.dat', np.c_[np.linspace(vmin, vmax, vdivs+1), gate.tableB])
        
    return chan


def make_CaPool(parent='/library', name='Ca'):
    ca = moose.CaConc(f'{parent}/{name}')
    ca.CaBasal = 75.5e-6  # 75.5 nM = 75.5e6 mM
    ca.tau = 10e-3
    ca.thick = 0.084e-6
    ca.diameter = 10e-6  # 10 um dia
    ca.length = 0  # spherical shell
    return ca


def make_Granule_98():
    comp = moose.Compartment('comp')
    comp.length = 0
    comp.diameter = 10e-6
    sarea = np.pi * comp.diameter * comp.diameter
    comp.Cm = sarea * 1e-2  # CM = 1 uF/cm^2
    comp.Rm = 1 / (0.330033 * sarea)
    comp.initVm = -65e-3
    comp.Em = -65e-3
    chan_makers = {
        'naf': make_channel_NaF,
        'kdr': make_channel_KDR,
        'ka': make_channel_KA,
        'kca': make_channel_KCa,
        'cal': make_channel_CaHVA,
        'h': make_channel_H,
    }
    gdensity = {
        'naf': 557.227,
        'kdr': 0,  # 88.9691,
    }
    channels = {}
    for name, maker in chan_makers.items():
        density = gdensity.get(name, 0)
        if density > 0:
            channels[name] = maker(comp.path)
            for gt in ('X', 'Y', 'Z'):
                if getattr(channels[name], f'{gt}power') > 0:
                    gate = moose.element(getattr(channels[name], f'gate{gt}'))
                    print(gate.path)
                    gate.min = vmin
                    gate.max = vmax
                    gate.divs = vdivs
                    gate.useInterpolation = True

            channels[name].Gbar = density * sarea            
            print(f'{name}.Gbar = {channels[name].Gbar}')

    # capool = make_CaPool(comp.path)
    # moose.connect(cal, 'IkOut', capool, 'current')
    # moose.connect(capool, 'concOut', kca, 'concen')
    for name, chan in channels.items():
        print(name * 70)
        moose.connect(comp, 'channel', chan, 'channel')
        print(f'-{name}' * 35)
    capool = make_CaPool(comp.path)
    # These values are taken from OSB/NeuroML2 version of the model
    # cal.Gbar = 0  # 9.084216 * sarea
    # chan_h.Gbar = 0  # 0.3090506 * sarea
    # ka.Gbar = 0  # 11.4567 * sarea
    # kca.Gbar = 0  # 179.811 * sarea
    # kdr.Gbar = 0.0 # 88.9691 * sarea
    # naf.Gbar = 557.227 * sarea
 
    return {
        'compartment': comp,
        'channels': channels,
        'ca': capool,
    }


def setup_recording(model_dict):
    data = moose.Neutral('data')
    vmtab = moose.Table(f'{data.path}/Vm')
    catab = moose.Table(f'{data.path}/conc_Ca')
    comp = model_dict['compartment']
    moose.connect(vmtab, 'requestOut', comp, 'getVm')
    gk_tabs = {}
    state_tabs = {}
    for name, chan in model_dict['channels'].items():
        gk_tabs[name] = moose.Table(f'{data.path}/g_{name}')
        moose.connect(gk_tabs[name], 'requestOut', chan, 'getGk')
        states = []
        if chan.Xpower > 0:
            m_tab = moose.Table(f'{data.path}/m_{name}')
            moose.connect(m_tab, 'requestOut', chan, 'getX')
            states.append(m_tab)
        if chan.Ypower > 0:
            h_tab = moose.Table(f'{data.path}/h_{name}')
            moose.connect(h_tab, 'requestOut', chan, 'getY')
            states.append(h_tab)
        state_tabs[name] = states
    # moose.connect(catab, 'requestOut', model_dict['ca'], 'getCa')
    return {'Vm': vmtab, 'gk': gk_tabs, 'state': state_tabs}  # , 'ca': catab}


if __name__ == '__main__':
    model_dict = make_Granule_98()
    data = setup_recording(model_dict)
    fig, axes = plt.subplots(nrows=3, ncols=1)
    prestim = 100e-3
    stimtime = 500e-3
    poststim = 100e-3
    for tick in range(10):
        moose.setClock(tick, 1e-6)
    moose.reinit()
    m = moose.element(model_dict['channels']['naf'].gateX)
    h = moose.element(model_dict['channels']['naf'].gateY)
    fig, ax = plt.subplots(nrows=2)
    v = np.linspace(m.min, m.max, m.divs+1)
    ax[0].plot(v, m.tableA/m.tableB, label='minf')
    ax[0].plot(v, h.tableA/h.tableB, label='hinf')
    ax[1].plot(v, 1/m.tableB, label='mtau')
    ax[1].plot(v, 1/h.tableB, label='htau')
    for x in ax.flat:
        x.legend()
    plt.show()
    moose.start(prestim)
    model_dict['compartment'].inject = 10e-12
    moose.start(stimtime)
    model_dict['compartment'].inject = 0.0
    moose.start(poststim)

    t = data['Vm'].dt * np.arange(len(data['Vm'].vector))

    # Save data into text files
    np.savetxt('Vm.f.dat', np.c_[t, data['Vm'].vector])
    for name, gk in data['gk'].items():
        np.savetxt(f'{gk.name}.f.dat', np.c_[t, gk.vector])
    for name, states in data['state'].items():
        for state_tab in states:
            np.savetxt(f'{state_tab.name}.f.dat', np.c_[t, state_tab.vector])
    axes[0].plot(t, data['Vm'].vector)
    for name, tab in data['gk'].items():
        axes[1].plot(t, tab.vector, label=tab.name)
    axes[1].legend()
    for name, states in data['state'].items():
        for state_tab in states:
            axes[2].plot(t, state_tab.vector, label=state_tab.name)
    # axes[2].plot(t, data['ca'].vector)
    axes[2].legend()
    fig.tight_layout()
    plt.show()

#
# granule_deschutter.py ends here
