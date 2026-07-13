# run_traubnet.py ---
#
# Filename: run_traubnet.py
# Description:
# Author: Subhasis Ray
# Created: Sun Jul 12 22:59:28 2026 (+0530)
#

# Code:

import sys
import time
import numpy as np
from collections import defaultdict
import h5py
import matplotlib.pyplot as plt
import moose
import cells
import cortical_column as cort


def setup_data_recording(model_root, vm_frac=0.1):
    """Setup data recording: spike times for all neurons, somatic Vm
    for a subset of `vm_frac` fraction of each celltype"""
    spike_dict = defaultdict(dict)
    Vm_dict = defaultdict(dict)

    data = moose.Neutral('/data')
    Vm = moose.Neutral('/data/Vm')
    spikes = moose.Neutral('/data/spikes')
    # Go through soma (comp_1) and spikegens
    for celltype in cells.cell_spec:
        nrns = moose.wildcardFind(f'{model_root}/{celltype}#')
        if len(nrns) == 0:
            print('No neuron of type', celltype, 'skipping ...')
            continue

        print('Celltype', celltype, ', neurons:', len(nrns))
        for cell in nrns:
            sg = moose.wildcardFind(f'{cell.path}/##[ISA=SpikeGen]')
            if len(sg) != 1:
                raise Exception(f'{cell.path} has {len(sg)} spikegens')
            sg = sg[0]
            tab = moose.Table(f'{spikes.path}/spikes_{cell.name}')
            moose.connect(sg, 'spikeOut', tab, 'input')
            spike_dict[celltype][cell.name] = tab

        count = int(len(nrns) * vm_frac)
        if count == 0:
            count = 1
        selected = np.random.choice(nrns, size=count)
        for cell in selected:
            tab = moose.Table(f'{Vm.path}/{cell.name}')
            moose.connect(tab, 'requestOut', moose.element(f'{cell.path}/comp_1'), 'getVm')
            Vm_dict[celltype][cell.name] = tab
    return (spike_dict, Vm_dict)


def dump_data(filename, spike_dict, Vm_dict):
    """Write the recorded data to `filename` in NSDF (HDF5) layout so
    the visualization code (`vis.display_data` via
    `adapter.get_data`/`get_spike_vm`) can replay it.

    Somatic Vm is written as uniformly-sampled data under
    ``/data/uniform/<celltype>/Vm`` (one row per recorded cell) with the
    corresponding cell names in ``/map/uniform/<celltype>/Vm``. Spike
    trains are written as event data under
    ``/data/event/<celltype>/spike/<cell name>``.
    """
    str_dt = h5py.string_dtype(encoding='utf-8')
    with h5py.File(filename, 'w') as fd:
        # Uniformly sampled somatic Vm
        for celltype, tables in Vm_dict.items():
            if len(tables) == 0:
                continue
            names = list(tables.keys())
            traces = [np.asarray(tables[name].vector) for name in names]
            # guard against off-by-one differences in trace length
            length = min(len(tr) for tr in traces)
            data = np.vstack([tr[:length] for tr in traces])
            dt = tables[names[0]].dt
            dset = fd.create_dataset(
                f'/data/uniform/{celltype}/Vm', data=data
            )
            dset.attrs['dt'] = dt
            dset.attrs['field'] = 'Vm'
            fd.create_dataset(
                f'/map/uniform/{celltype}/Vm',
                data=np.array(names, dtype=str_dt),
            )
            fd.attrs['dt'] = dt
        # Event (spike) data
        for celltype, tables in spike_dict.items():
            for name, table in tables.items():
                fd.create_dataset(
                    f'/data/event/{celltype}/spike/{name}',
                    data=np.asarray(table.vector),
                )
    print(f'Wrote recorded data to {filename}')


if __name__ == '__main__':
    runtime = 200e-3
    scale = 1.0
    outfile = 'traubnet_data.h5'
    if len(sys.argv) > 1:
        runtime = float(sys.argv[1])
    if len(sys.argv) > 2:
        scale = float(sys.argv[2])
    if len(sys.argv) > 3:
        outfile = sys.argv[3]
    model_root = cort.make_net(cort.cell_counts, cort.connection_spec, '/model', scale=scale)
    spike_dict, Vm_dict = setup_data_recording(model_root.path, vm_frac=0.1)
    moose.reinit()
    ts = time.perf_counter()
    moose.start(runtime)
    te = time.perf_counter()
    print(f'Completed {runtime} s of simulation in {(te - ts)} s')
    dump_data(outfile, spike_dict, Vm_dict)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all')
    for celltype in Vm_dict:
        for cell, Vm in Vm_dict[celltype].items():
            v = Vm.vector
            t = np.arange(len(v)) * Vm.dt
            axes[0].plot(t, v)
    cell_no = 1
    for celltype, sd in spike_dict.items():
        for cell, spikes in sd.items():
            st = spikes.vector
            if len(st) > 0:
                axes[1].plot(st, cell_no * np.ones(len(st)), '|')
            cell_no += 1
    plt.show()
    print('Exiting')



#
# run_traubnet.py ends here
