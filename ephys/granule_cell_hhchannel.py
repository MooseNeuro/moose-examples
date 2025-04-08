# granule_deschutter.py ---
#
# Filename: granule_deschutter.py
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


from numpy import exp

vmin = -150e-3
vmax = 100e-3
vdivs = 640
v = np.linspace(vmin, vmax, vdivs + 1)

temp_exp = 17.350264793
temp = 32.0
Q10 = 3
q10_mul = Q10 ** (0.1 * (temp - temp_exp))


v_offset = 10e-3
V = v - v_offset


def plot_hhgate(hhgate, axes=None):
    alpha = hhgate.tableA
    beta = hhgate.tableB - hhgate.tableA
    v = np.linspace(hhgate.min, hhgate.max, hhgate.divs + 1)
    np.savetxt(
        f'{hhgate.parent.name}.{hhgate.name}.alpha.moose.dat', np.c_[v, alpha]
    )
    np.savetxt(
        f'{hhgate.parent.name}.{hhgate.name}.beta.moose.dat', np.c_[v, beta]
    )
    np.savetxt(
        f'{hhgate.parent.name}.{hhgate.name}.tau.moose.dat',
        np.c_[v, 1 / hhgate.tableB],
    )
    np.savetxt(
        f'{hhgate.parent.name}.{hhgate.name}.inf.moose.dat',
        np.c_[v, hhgate.tableA / hhgate.tableB],
    )
    if axes is None:
        fig, axes = plt.subplots(ncols=3, sharex='all')
    else:
        assert (
            len(axes.shape) == 1 and axes.shape[0] >= 3
        ), 'axes must be 1D array of at least 3 Axes'
        fig = axes[0].get_figure()
    axes[0].plot(v * 1e3, alpha, label=f'{hhgate.name}.alpha')
    axes[0].plot(v * 1e3, beta, label=f'{hhgate.name}.beta')
    axes[0].set_ylabel('1/s')
    axes[1].plot(v * 1e3, 1e3 / hhgate.tableB, label=f'{hhgate.name}.tau')
    axes[2].plot(
        v * 1e3, hhgate.tableA / hhgate.tableB, label=f'{hhgate.name}.inf'
    )
    axes[1].set_ylabel('ms')
    for ax in axes.flat:
        ax.legend()
        ax.set_xlabel('mV')
    fig.set_size_inches(7, 2.5)
    return fig, axes


def plot_hhchannel_params(hhchannel):
    title = hhchannel.name
    if hhchannel.Xpower > 0:
        mgate = moose.element(f'{hhchannel.path}/gateX')
        fig, axes = plot_hhgate(mgate)
        title += f', m: {hhchannel.Xpower}'
    if hhchannel.Ypower > 0:
        mgate = moose.element(f'{hhchannel.path}/gateY')
        fig, axes = plot_hhgate(mgate, axes=axes)
        title += f', h: {hhchannel.Ypower}'
    fig.suptitle(title)


def make_channel_CaHVA(parent='/library', name='CaHVA'):
    """Create a High-Voltage Activated (called CaL for Long lasting)
    Ca channel from Maex and DeSchutter 1998"""
    chan = moose.HHChannel(f'{parent}/{name}')
    chan.Xpower = 2
    chan.Ypower = 1
    chan.Ek = 80e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    alpha_m = (1600 / (1 + exp(-72 * (V - 5e-3)))) * q10_mul
    x = (V - (-8.9e-3)) / (-0.005)
    beta_m = 100 * x / (1 - exp(-x)) * q10_mul
    alpha_h = 5 * exp(-50 * (V - (-0.06))) * q10_mul
    alpha_h[V < -0.060] = 5.0 * q10_mul
    beta_h = (5 - 5 * exp(-50 * (V - (-0.060)))) * q10_mul
    beta_h[V < -0.060] = 0.0
    mgate.tableA = alpha_m
    mgate.tableB = alpha_m + beta_m
    hgate.tableA = alpha_h
    hgate.tableB = alpha_h + beta_h
    for gate in (mgate, hgate):
        gate.min = vmin
        gate.max = vmax
        gate.divs = vdivs
        gate.useInterpolation = True


def make_channel_H(parent='/library', name='H'):
    chan = moose.HHChannelF(f'{parent}/{name}')
    chan.Xpower = 1
    chan.Ek = -42e-3
    mgate = moose.element(f'{chan.path}/gateX')
    # OSB version uses 0.8, but 4 in paper
    # in OSB NeuroML midpoint is -0.065 V, adding 10 mV offset?
    mgate.alpha = '0.8 * exp(-90.9 * ((v - 10e-3) -(-75e-3)))'
    mgate.beta = '0.8 * exp(90.9 * ((v - 10e-3) - (-75e-3)))'
    mgate.min = vmin
    mgate.max = vmax
    mgate.divs = vdivs
    return chan


def make_channel_KA(parent='/library', name='KA'):
    """Create an A-type K channel from Maex and DeSchutter 1998"""
    chan = moose.HHChannelF(f'{parent}/{name}')
    chan.Xpower = 3
    chan.Ypower = 1
    chan.Ek = -90e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    mgate.tau = '0.41e-3 * exp(-((v - 10e-3) - (-43.5e-3))/0.0428) + 0.167e-3'
    mgate.inf = '1 / (1 + exp(-((v - 10e-3) - (-46.7e-3))/19.8))'
    hgate.tau = (
        '0.001 * (10.8 + 30 * (v - 10e-3) + '
        '1 / (57.9 * exp((v - 10e-3) * 127) + '
        '134e-6 * exp(-59 * (v - 10e-3) )))'
    )
    hgate.inf = '1 / (1 + exp(-((v - 10e-3) - (-78.8e-3))/0.0084))'    
    return chan


def make_channel_KCa(parent='/library', name='KCa'):
    """Create Ca dependent K channel (KC in paper)"""
    chan = moose.HHChannelF2D(f'{parent}/{name}')
    chan.Xpower = 1
    chan.Xindex = 'VOLT_C1_INDEX'
    chan.Ek = -90e-3
    mgate = moose.element(f'{chan.path}/gateX')
    # Why 2500 instead of 1250 in OSB/NeuroML2
    mgate.alpha = '2500 / ( 1 + 1.5e-3 * exp(-85 * (v - 10e-3)) / c)'
    mgate.beta = '1500 / (1 + c / (1.5e-4 * exp(-77 * (v - 10e-3))))'
    return chan


def make_channel_KDR(parent='/library', name='KDr'):
    """Create delayed rectifier type K channel"""
    chan = moose.HHChannel(f'{parent}/{name}')
    chan.Xpower = 4
    chan.Ypower = 1
    chan.Ek = -90e-3
    alpha_m = q10_mul * 170 * exp(73 * (V - (-38e-3)))
    # KDR_m_beta: in NeuroML A is 170, in paper 0.85e3, in GENESIS code they load the table from a data file
    beta_m = q10_mul * 170 * exp(-18 * (V - (-38e-3)))
    mgate = moose.element(f'{chan.path}/gateX')
    mgate.tableA = alpha_m
    mgate.tableB = alpha_m + beta_m
    alpha_h = np.where(
        V > -0.046, 0.76, 0.7 + 0.065 * exp(-80 * (V - (-46e-3)))
    )
    beta_h = 1.1 / (
        1 + exp(-80.7 * (V - (-0.044)))
    )  # numerator A=5.5 in paper
    alpha_h *= q10_mul
    beta_h *= q10_mul
    hgate = moose.element(f'{chan.path}/gateY')
    hgate.tableA = alpha_h
    hgate.tableB = alpha_h + beta_h
    for gate in (mgate, hgate):
        gate.min = vmin
        gate.max = vmax
        gate.divs = vdivs
        gate.useInterpolation = True
    return chan


def make_channel_NaF(parent='/library', name='NaF'):
    """Create fast Na channel"""
    chan = moose.HHChannel(f'{parent}/{name}')
    chan.Xpower = 3
    chan.Ypower = 1
    chan.Ek = 55e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    for gate in (mgate, hgate):
        gate.min = vmin
        gate.max = vmax
        gate.divs = vdivs
        gate.useInterpolation = True
    # These are based on NeuroML - for some reason the A parameter is
    # double that in the paper
    alpha_m = 1500 * exp(81 * (V - (-39e-3)))
    beta_m = 1500 * exp(-66 * (V - (-39e-3)))
    tau_m = (
        np.where(
            alpha_m + beta_m == 0, 0, np.maximum(1 / (alpha_m + beta_m), 5e-5)
        )
        / q10_mul
    )
    inf_m = alpha_m / (alpha_m + beta_m)
    mgate.tableA = inf_m / tau_m
    mgate.tableB = 1 / tau_m
    mgate.useInterpolation = True
    alpha_h = 120 * exp(-89 * (V - (-0.05)))
    beta_h = 120 * exp(89 * (V - (-0.05)))
    inf_h = alpha_h / (alpha_h + beta_h)
    tau_h = (
        np.where(
            alpha_h + beta_h == 0,
            0,
            np.maximum(1 / (alpha_h + beta_h), 2.25e-4),
        )
        / q10_mul
    )
    hgate.tableA = inf_h / tau_h
    hgate.tableB = 1 / tau_h
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
            channels[name].Gbar = density * sarea
            print(f'{name}.Gbar = {channels[name].Gbar}')

    # capool = make_CaPool(comp.path)
    for name, chan in channels.items():
        moose.connect(comp, 'channel', chan, 'channel')
        plot_hhchannel_params(chan)
    # moose.connect(cal, 'IkOut', capool, 'current')
    # moose.connect(capool, 'concOut', kca, 'concen')
    # These values are taken from OSB/NeuroML2 version of the model
    # cal.Gbar = 0  # 9.084216 * sarea
    # chan_h.Gbar = 0  # 0.3090506 * sarea
    # ka.Gbar = 0  # 11.4567 * sarea
    # kca.Gbar = 0  # 179.811 * sarea
    return {
        'compartment': comp,
        'channels': channels,
        # 'ca': capool,
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
    moose.start(prestim)
    model_dict['compartment'].inject = 10e-12
    moose.start(stimtime)
    model_dict['compartment'].inject = 0.0
    moose.start(poststim)

    t = data['Vm'].dt * np.arange(len(data['Vm'].vector))

    # Save data into text files
    np.savetxt('Vm.dat', np.c_[t, data['Vm'].vector])
    for name, gk in data['gk'].items():
        np.savetxt(f'{gk.name}.dat', np.c_[t, gk.vector])
    for name, states in data['state'].items():
        for state_tab in states:
            np.savetxt(f'{state_tab.name}.dat', np.c_[t, state_tab.vector])
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
