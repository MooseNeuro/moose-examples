# squid_demo.py --- 
# 
# Filename: squid_demo.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Feb 22 23:24:21 2012 (+0530)
# Version: 
# Last-Updated: Wed Sep 24 21:53:47 2025 (+0530)
#           By: Subhasis Ray
#     Update #: 178
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

# Code:

import moose

from squid import SquidAxon
from electronics import ClampCircuit

class SquidSetup(object):
    def __init__(self):
        self.scheduled = False        
        self.model_container = moose.Neutral('/model')
        self.data_container = moose.Neutral('/data')
        self.axon = SquidAxon('/model/axon')
        self.clamp_ckt = ClampCircuit('/model/electronics', self.axon)
        self.simdt = 0.0
        self.plotdt = 0.0
        self.setup_recording()

    def setup_recording(self):
        # Setup data collection
        self.vm_table = moose.Table('/data/Vm')
        moose.connect(self.vm_table, 'requestOut', self.axon.C, 'getVm')
        self.cmd_table = moose.Table('/data/command')
        moose.connect(self.cmd_table, 'requestOut', self.clamp_ckt.vclamp, 'getOutputValue')
        self.iclamp_table = moose.Table('/data/Iclamp')
        moose.connect(self.iclamp_table, 'requestOut', self.clamp_ckt.iclamp, 'getOutputValue')
        self.vclamp_table = moose.Table('/data/Vclamp')
        moose.connect(self.vclamp_table, 'requestOut', self.clamp_ckt.pid, 'getOutputValue')
        self.m_table = moose.Table('/data/m')
        moose.connect(self.m_table, 'requestOut', self.axon.Na_channel.chan, 'getX')
        self.h_table = moose.Table('/data/h')
        moose.connect(self.h_table, 'requestOut', self.axon.Na_channel.chan, 'getY')
        self.n_table = moose.Table('/data/n')
        moose.connect(self.n_table, 'requestOut', self.axon.K_channel.chan, 'getX')
        self.ina_table = moose.Table('/data/INa')
        moose.connect(self.ina_table, 'requestOut', self.axon.Na_channel.chan, 'getIk')
        self.ik_table = moose.Table('/data/IK')
        moose.connect(self.ik_table, 'requestOut', self.axon.K_channel.chan, 'getIk')
        self.gna_table = moose.Table('/data/GNa')
        moose.connect(self.gna_table, 'requestOut', self.axon.Na_channel.chan, 'getGk')
        self.gk_table = moose.Table('/data/GK')
        moose.connect(self.gk_table, 'requestOut', self.axon.K_channel.chan, 'getGk')
        
    def schedule(self, clampmode):
        if clampmode == 'vclamp':
            self.clamp_ckt.do_voltage_clamp()
        else:
            self.clamp_ckt.do_current_clamp()
        self.simdt = self.axon.dt
        self.plotdt = self.vm_table.dt
        moose.reinit()
        
        
    def run(self, runtime):
        moose.start(runtime)

    def save_data(self):
        for child in self.data_container.children:
            tab = moose.Table(moose.element(child))
            tab.xplot('%s.dat' % (tab.name), tab.name)

import sys            
clamp_mode = 'vclamp'
if __name__ == '__main__':
    demo = SquidSetup()
    if len(sys.argv) > 1:
        clamp_mode = sys.argv[1]
    # The pulsegen object needs different values for different clamp modes
    if clamp_mode == 'iclamp':
        demo.clamp_ckt.configure_pulses()
    else:
        demo.clamp_ckt.configure_pulses(baseLevel=0.0, firstDelay=10.0, firstLevel=SquidAxon.EREST_ACT, firstWidth=0.0, secondDelay=0.0, secondLevel=50.0+SquidAxon.EREST_ACT, secondWidth=20.0)
    demo.schedule(clamp_mode)
    
    demo.run(50.0)
    demo.save_data()

# 
# squid_demo.py ends here
