# compartment_net.py ---
#
# Filename: compartment_net.py
# Description:
# Author:Subhasis Ray
# Maintainer:
# Created: Sat Aug 11 14:30:21 2012 (+0530)
# Version:
# Last-Updated: Wed Jan 15 16:41:33 2025 (+0530)
#           By: Subhasis Ray
#     Update #: 634
# URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
# A demo to create a network of single compartmental neurons connected
# via alpha synapses.
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
#
#

# Code:

import numpy as np
import matplotlib.pyplot as plt

import moose
from ionchannel import create_active_compartment


def create_population(container, size, gbarSyn=1e-10):
    """Create a population of `size` single compartmental neurons with Na+
    and K+ channels and attached synapse and spike generators.

    The SpikeGen and SynChan objects connected to these can act as
    plug points for setting up synapses later.

    Parameters
    ----------
    container: moose element
        Container of the population. A compartment vector called `neuron` will
        be created inside this container.
    size: int
        This is the number of elements in the compartment vector
    gbarSyn: float or sequence of floats
        Peak synaptic conductance. If gbarSyn is a float, every synapse will
        have this peak conductance. If gbarSyn is a sequence of floats, it
        must be of `size` length, and the entries are assigned to corresponding
        elements in the synaptic channel vector.

    Returns
    -------
    dict: dict of model components
    """
    comps = create_active_compartment(f'{container.path}/neuron', number=size)
    synchan = moose.SynChan(f'{container.path}/synchan', ndata=size)
    synchan.vec.Gbar = gbarSyn
    synchan.vec.tau1 = 2e-3
    synchan.vec.tau2 = 2e-3
    _ = moose.connect(comps, 'channel', synchan, 'channel', 'OneToOne')

    # This was old style, now you can create vector objects directly
    # synhandler = moose.vec('{}/synhandler'.format(path), n=size,
    #                        dtype='SimpleSynHandler')
    synhandler = moose.SimpleSynHandler(f'{container.path}/synh', ndata=size)
    moose.connect(
        synhandler, 'activationOut', synchan, 'activation', 'OneToOne'
    )
    spikegen = moose.SpikeGen(f'{container.path}/spikegen', ndata=size)
    spikegen.vec.threshold = 0.0
    _ = moose.connect(comps, 'VmOut', spikegen, 'Vm', 'OneToOne')
    return {
        'compartment': comps,
        'spikegen': spikegen,
        'synchan': synchan,
        'synhandler': synhandler,
    }


def make_synapses(spikegen, synhandler, connprob=1.0, delay=5e-3):
    """
    Create synapses from spikegen array to synchan array.

    Parameters
    ----------
    spikegen: vec of SpikGen elements
        Spike generators from neurons.
    synhandler: vec of SynHandler elements
        Handles presynaptic spike event inputs to synchans.
    connprob: float in range (0, 1]
        connection probability between any two neurons.
    delay: float (mean delay of synaptic transmission)
        Individual delays are normally distributed with sd=0.1*mean.
    """
    for sh in synhandler.vec:
        scount = spikegen.numData
        sh.numSynapses = scount
        sh.synapse.vec.delay = 5e-3
        for ii, syn in enumerate(sh.synapse.vec):
            _ = moose.connect(spikegen.vec[ii], 'spikeOut', syn, 'addSpike')
            print(
                'Connected',
                spikegen.vec[ii].path,
                'to',
                syn.path,
                'on',
                sh.path,
            )


def create_network(size=2):
    """
    Create a network containing two neuronal populations, popA and
    popB and connect them up.

    Parameters
    ----------
    size: int
        size of each population
    """
    net = moose.Neutral('network')
    pop_a = create_population(
        moose.Neutral(f'{net.path}/popA'),
        size,
        gbarSyn=np.random.randint(1, 10, size) * 1e-10,
    )
    pop_b = create_population(
        moose.Neutral(f'{net.path}/popB'),
        size,
        gbarSyn=np.random.randint(1, 100, size) * 1e-10,
    )
    make_synapses(pop_a['spikegen'], pop_b['synhandler'])
    pulse = moose.PulseGen('pulse')
    pulse.level[0] = 1e-9
    pulse.delay[0] = 0.02  # disable the pulsegen
    pulse.width[0] = 40e-3
    pulse.delay[1] = 1e9
    data = moose.Neutral('/data')
    vm_a = moose.Table(f'{data.path}/Vm_A', ndata=size)
    moose.connect(
        pulse, 'output', pop_a['compartment'], 'injectMsg', 'OneToAll'
    )
    moose.connect(
        vm_a, 'requestOut', pop_a['compartment'], 'getVm', 'OneToOne'
    )
    vm_b = moose.Table(f'{data.path}/Vm_B', ndata=size)
    moose.connect(
        vm_b, 'requestOut', pop_b['compartment'], 'getVm', 'OneToOne'
    )
    gksyn_b = moose.Table(f'{data.path}/Gk_syn_b', ndata=size)
    moose.connect(gksyn_b, 'requestOut', pop_b['synchan'], 'getGk', 'OneToOne')
    pulsetable = moose.Table(f'{data.path}/pulse')
    pulsetable.connect('requestOut', pulse, 'getOutputValue')
    return {
        'A': pop_a,
        'B': pop_b,
        'Vm_A': vm_a,
        'Vm_B': vm_b,
        'Gsyn_B': gksyn_b,
    }


def main(simtime=0.1):
    """
    This example illustrates how to create a network of single compartmental neurons
    connected via alpha synapses. It also shows the use of SynChan class.
    """
    netinfo = create_network(size=2)
    vm_a = netinfo['Vm_A']
    vm_b = netinfo['Vm_B']
    gksyn_b = netinfo['Gsyn_B']
    # for ii in range(10):
    #     moose.setClock(ii, simdt)
    # moose.setClock(18, plotdt)
    moose.reinit()
    moose.start(simtime)
    t = np.arange(len(vm_a.vec[0].vector)) * vm_a.dt
    plt.subplot(221)
    for oid in vm_a.vec:
        plt.plot(t, oid.vector, label=f'{oid.name}[{oid.dataIndex}]')
    plt.legend()
    plt.subplot(223)
    for oid in vm_b.vec:
        plt.plot(t, oid.vector, label=f'{oid.name}[{oid.dataIndex}]')
    plt.legend()
    plt.subplot(224)
    title = [
        f'{synchan.parent.name}[{synchan.dataIndex}].Gbar={synchan.Gbar:0.3g}'
        for synchan in netinfo['B']['synchan'].vec
    ]
    for oid in gksyn_b.vec:
        plt.plot(t, oid.vector, label=f'{oid.name}[{oid.dataIndex}]')
    plt.title('\n'.join(title))
    plt.legend()
    plt.show()


#
# compartment_net.py ends here
if __name__ == '__main__':
    main()
