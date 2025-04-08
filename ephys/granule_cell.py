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


def make_channel_CaHVA(parent='/library', name='CaHVA'):
    """Create a High-Voltage Activated (called CaL for Long lasting)
    Ca channel from Maex and DeSchutter 1998"""
    chan = moose.HHChannelF(f'{parent}/{name}')
    chan.Xpower = 2
    chan.Ypower = 1
    chan.Ek = 80e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    mgate.alpha = '1600/(1 + exp(-72*((v -10e-3) - 5e-3)))'
    mgate.beta = '100 * ((v - 10e-3) - (-8.9e-3))/(-0.005)/(1 - exp(200 * ((v - 10e-3) - (-8.9e-3))))'
    hgate.alpha = '(v - 10e-3) < -0.060? 5.0: 5 * exp(-50 * ((v - 10e-3) - (-0.06)))'
    hgate.beta = '(v - 10e-3) < -0.060? 0.0: 5 - 5 * exp(-50 * ((v - 10e-3) - (-0.060)))'    
    return chan


def make_channel_H(parent='/library', name='H'):
    chan = moose.HHChannelF(f'{parent}/{name}')
    chan.Xpower = 1
    chan.Ek = -42e-3
    mgate = moose.element(f'{chan.path}/gateX')
    # OSB version uses 0.8, but 4 in paper
    mgate.alpha = '0.8 * exp(-90.9 * ((v - 10e-3) -(-75e-3)))'  # in OSB NeuroML midpoint is -0.065 V, adding 10 mV offset?
    mgate.beta = '0.8 * exp(90.9 * ((v - 10e-3) - (-75e-3)))'
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
    hgate.tau = '0.001 * (10.8 + 30 * (v - 10e-3) + 1 / (57.9 * exp((v - 10e-3) * 127) + 134e-6 * exp(-59 * (v - 10e-3) )))'
    hgate.inf = '1 / (1 + exp(-((v - 10e-3) - (-78.8e-3))/0.0084))'
    return chan

    
def make_channel_KCa(parent='/library', name='KCa'):
    """Create Ca dependent K channel (KC in paper)"""
    chan = moose.HHChannelF2D(f'{parent}/{name}')
    chan.Xpower = 1
    chan.Xindex = 'VOLT_C1_INDEX'
    chan.Ek = -90e-3
    mgate = moose.element(f'{chan.path}/gateX')
    mgate.alpha = '2500 / ( 1 + 1.5e-3 * exp(-85 * (v - 10e-3)) / c)'   # Why 2500 instead of 1250 in OSB/NeuroML2
    mgate.beta = '1500 / (1 + c / (1.5e-4 * exp(-77 * (v - 10e-3))))'
    return chan
    
    
def make_channel_KDr(parent='/library', name='KDr'):
    """Create delayed rectifier type K channel"""
    chan = moose.HHChannelF(f'{parent}/{name}')
    chan.Xpower = 4
    chan.Ypower = 1
    chan.Ek = -90e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    mgate.alpha = '170 * exp(73 * ((v - 10e-3) - (-38e-3)))'
    # KDR_m_beta: in NeuroML A is 170, in paper 0.85e3, in GENESIS code they load the table from a data file
    mgate.beta = '170 * exp(-18 * ((v - 10e-3) - (-38e-3)))'  
    hgate.alpha = '(v - 10e-3) > -0.046? 0.76: 0.7 + 0.065 * exp(-80 * (v - 10e-3) - (-46e-3))'
    hgate.beta = '1.1/(1 + exp(-80.7 * ((v - 10e-3) - (-0.044))))'  # numerator A=5.5 in paper
    return chan
    
    
def make_channel_NaF(parent='/library', name='NaF'):
    """Create fast Na channel"""
    chan = moose.HHChannelF(f'{parent}/{name}')
    chan.Xpower = 3
    chan.Ypower = 1
    chan.Ek = 55e-3
    mgate = moose.element(f'{chan.path}/gateX')
    hgate = moose.element(f'{chan.path}/gateY')
    # These are based on NeuroML - for some reason the A parameter is
    # double that in the paper
    mgate.tau = '~(alpha:=1500 * exp(81 *((v - 10e-3) - (-39e-3))), beta:=1500 * exp(-66 * ((v - 10e-3) - (-39e-3))), alpha + beta == 0? 0: max(1/(alpha+beta), 5e-5))'
    mgate.inf = '~(alpha:=1500 * exp(81 *((v - 10e-3) - (-39e-3))), beta:=1500 * exp(-66 * ((v - 10e-3) - (-39e-3))), alpha/(alpha+beta))'
    hgate.tau = '~(alpha:=120*exp(-89 * ((v - 10e-3) - (-0.05))), beta:=120 * exp(89 * ((v - 10e-3) - (-0.05))), alpha + beta == 0? 0: min(1/(alpha+beta), 2.25e-4))'
    hgate.inf = '~(alpha:=120*exp(-89 * ((v - 10e-3) - (-0.05))), beta:=120 * exp(89 * ((v - 10e-3) - (-0.05))), alpha/(alpha+beta))'
    
    return chan


def make_CaPool(parent='/library', name='Ca'):
    ca = moose.CaConc(f'{parent}/{name}')
    ca.CaBasal = 75.5e-6  # 75.5 nM = 75.5e6 mM
    ca.tau = 10e-3
    ca.thick = 0.084e-6
    ca.diameter = 10e-6  # 10 um dia
    ca.length = 0   # spherical shell
    return ca


def make_Granule_98():
    comp = moose.Compartment('comp')
    comp.length = 0
    comp.diameter = 10e-6
    sarea = np.pi * comp.diameter * comp.diameter
    comp.Cm = sarea * 1e-2   # CM = 1 uF/cm^2
    naf = make_channel_NaF(comp.path)
    kdr = make_channel_KDr(comp.path)
    ka = make_channel_KA(comp.path)
    kca = make_channel_KCa(comp.path)
    cal = make_channel_CaHVA(comp.path)
    chan_h = make_channel_H(comp.path)
    capool = make_CaPool(comp.path)
    for chan in (naf, kdr, ka, kca, cal, chan_h):
        moose.connect(comp, 'channel', chan, 'channel')
    moose.connect(cal, 'IkOut', capool, 'current')
    moose.connect(capool, 'concOut', kca, 'concen')
    # These values are taken from OSB/NeuroML2 version of the model
    cal.Gbar = 0  # 9.084216 * sarea
    chan_h.Gbar = 0  # 0.3090506 * sarea
    ka.Gbar = 0  # 11.4567 * sarea
    kca.Gbar = 0  # 179.811 * sarea
    kdr.Gbar = 88.9691 * sarea
    naf.Gbar = 557.227 * sarea
    comp.Rm = 1/(0.330033 * sarea)
    comp.initVm = -65e-3

    return {'compartment': comp,
            'channels': [ka, kdr, kca, chan_h, cal, naf],
            'ca': capool}


def setup_recording(model_dict):
    data = moose.Neutral('data')
    vmtab = moose.Table(f'{data.path}/Vm')
    catab = moose.Table(f'{data.path}/conc_Ca')
    comp = model_dict['compartment']
    moose.connect(vmtab, 'requestOut', comp, 'getVm')
    gk_tabs = []
    for chan in model_dict['channels']:
        gk_tabs.append(moose.Table(f'{data.path}/g_{chan.name}'))
        moose.connect(gk_tabs[-1], 'requestOut', chan, 'getGk')
    moose.connect(catab, 'requestOut', model_dict['ca'], 'getCa')
    return {'Vm': vmtab, 'gk': gk_tabs, 'ca': catab}




if __name__ == '__main__':
    model_dict = make_Granule_98()
    data = setup_recording(model_dict)
    fig, axes = plt.subplots(nrows=3, ncols=1)
    prestim = 50e-3
    stimtime = 100e-3
    moose.reinit()
    moose.start(prestim)
    model_dict['compartment'].inject = 10e-12
    moose.start(stimtime)
    t = data['Vm'].dt * np.arange(len(data['Vm'].vector))
    axes[0].plot(t, data['Vm'].vector)
    for tab in data['gk']:
        axes[1].plot(t, tab.vector, label=tab.name)
    axes[1].legend()
    axes[2].plot(t, data['ca'].vector)
    fig.tight_layout()
    plt.show()
    
# 
# granule_deschutter.py ends here
