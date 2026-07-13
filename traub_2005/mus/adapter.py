# adapter.py ---
#
# Filename: adapter.py
# Author: Subhasis Ray
#
# Code:
"""Adapters between the MOOSE model / recorded data and the
visualization code in `vis.py`.

Two kinds of input are supported:

1. A live MOOSE model built by `cortical_column.make_net`. Use
   `model_to_graph` to turn it into an :class:`igraph.Graph` whose
   vertices are cells and whose (directed) edges are synaptic
   connections (presynaptic -> postsynaptic).

2. A recorded simulation in NSDF (HDF5) format. Use `get_data` to
   iterate over the recorded time series of a field (e.g. ``Vm``) one
   timestep at a time.
"""
import re
import numpy as np
import igraph as ig
import moose


#: Compartment-level synapses are named ``syn_<presynaptic cell name>``
SYN_PREFIX = 'syn_'


def celltype_of(cell_name):
    """Return the cell type of a cell named ``<celltype>_<index>``.

    The Traub cell-type names contain no underscore, so the type is
    everything before the last underscore.
    """
    return cell_name.rsplit('_', 1)[0]


def model_to_graph(model_root='/model'):
    """Build a directed graph of the network under `model_root`.

    Vertices correspond to the cells (``Neuron`` elements directly
    under `model_root`) and carry ``name`` and ``celltype``
    attributes. A directed edge ``pre -> post`` is added for every
    synaptic channel found on a postsynaptic compartment; the
    presynaptic cell is recovered from the SynChan name
    (``syn_<pre cell name>``).
    """
    root = moose.element(model_root)
    graph = ig.Graph(directed=True)

    names = []
    celltypes = []
    for child in root.children:
        elem = moose.element(child)
        if elem.className not in ('Neuron', 'Neutral'):
            continue
        names.append(elem.name)
        celltypes.append(celltype_of(elem.name))
    graph.add_vertices(names)
    graph.vs['celltype'] = celltypes

    known = set(names)
    edges = []
    for synchan in moose.wildcardFind(f'{model_root}/##[ISA=SynChan]'):
        pre_name = synchan.name[len(SYN_PREFIX):]
        # postsynaptic cell is two levels up: syn -> compartment -> cell
        post_name = synchan.parent.parent.name
        if pre_name in known and post_name in known:
            edges.append((pre_name, post_name))
    if edges:
        graph.add_edges(edges)
    return graph


def _infer_dt(dset, h5file):
    """Best-effort extraction of the sampling interval for an NSDF
    uniform dataset."""
    for key in ('dt', 'DT'):
        if key in dset.attrs:
            return float(dset.attrs[key])
    for key in ('dt', 'DT'):
        if key in h5file.attrs:
            return float(h5file.attrs[key])
    return 1.0


def _cell_name_from_source(source):
    """Extract a ``<celltype>_<index>`` cell name from an NSDF source
    string (which is usually a full element path such as
    ``/model/SupPyrRS_003/comp_1``)."""
    if isinstance(source, bytes):
        source = source.decode()
    match = re.search(r'([A-Za-z]+_\d+)', source)
    if match:
        return match.group(1)
    # fall back to the path component that looks like a cell
    for part in reversed(source.strip('/').split('/')):
        if '_' in part:
            return part
    return source


def get_data(datafile, field='Vm'):
    """Generator yielding ``(time, {cell_name: value})`` for each
    recorded timestep of `field` in the NSDF file `datafile`.

    The NSDF layout written by MOOSE is
    ``/data/uniform/<population>/<field>`` (a 2D array, one row per
    source) with the source list in
    ``/map/uniform/<population>/<field>``.
    """
    import h5py

    with h5py.File(datafile, 'r') as h5file:
        uniform = h5file['/data/uniform']
        # Collect (cell_name, dataset-row) for every population that
        # recorded `field`, plus a common time base.
        series = []  # list of (cell_name, 1D array)
        n_steps = 0
        dt = 1.0
        for pop in uniform:
            grp = uniform[pop]
            if field not in grp:
                continue
            dset = grp[field]
            dt = _infer_dt(dset, h5file)
            sources = None
            try:
                sources = h5file[f'/map/uniform/{pop}/{field}'][()]
            except KeyError:
                sources = [f'{pop}_{i}' for i in range(dset.shape[0])]
            data = dset[()]
            n_steps = max(n_steps, data.shape[1])
            for row, src in enumerate(sources):
                series.append((_cell_name_from_source(src), data[row]))

        for step in range(n_steps):
            t = step * dt
            snapshot = {
                name: float(row[step])
                for name, row in series
                if step < len(row)
            }
            yield t, snapshot


def get_spike_vm(datafile, amp=1000e-3, baseline=-65e-3):
    """Generator reproducing membrane-potential-like traces from
    recorded spike times in an NSDF file.

    For each timestep it yields ``(time, {cell_name: value})`` where a
    cell is set to `amp` at the steps where it spiked and `baseline`
    otherwise. Spike events are stored under
    ``/data/event/<population>/spike`` in NSDF.
    """
    import h5py

    with h5py.File(datafile, 'r') as h5file:
        event = h5file['/data/event']
        spike_times = {}  # cell_name -> sorted array of spike times
        tmax = 0.0
        dt = 1e-4
        for pop in event:
            for field in event[pop]:
                grp = event[pop][field]
                for src in grp:
                    times = np.asarray(grp[src][()], dtype=float)
                    if len(times):
                        tmax = max(tmax, float(times.max()))
                    spike_times[_cell_name_from_source(src)] = np.sort(times)

        n_steps = int(tmax / dt) + 1
        for step in range(n_steps):
            t = step * dt
            snapshot = {}
            for name, times in spike_times.items():
                spiking = np.any(np.abs(times - t) < (dt / 2.0))
                snapshot[name] = amp if spiking else baseline
            yield t, snapshot


#
# adapter.py ends here
