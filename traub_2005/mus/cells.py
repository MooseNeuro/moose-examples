# cells.py ---
#
# Filename: cells.py
# Description:
# Author: Subhasis Ray
# Created: Wed Apr 16 14:49:08 2025 (+0530)
#

# Code:
"""Cell prototype loaders for Traub 2005 model"""

import os
import numpy as np
import pandas as pd
import time
import moose
from config import logger


from channels import get_proto, init_channels

cell_spec = {
    'DeepAxoaxonic': {
        'proto': 'DeepAxoaxonic.p',
        'levels': 'DeepAxoaxonic.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -100e-3,
            'Ca': 125e-3,
            'AR': -40e-3,
            'GABA': -75e-3,
        },
        'tauCa': {'all': 20e-3, 'comp_1': 50e-3},
    },
    'DeepBasket': {
        'proto': 'DeepBasket.p',
        'levels': 'DeepBasket.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -100e-3,
            'AR': -40e-3,
            'Ca': 125e-3,
            'GABA': -75e-3,  # Sanchez-Vives et al. 1997
        },
        'tauCa': {'all': 20e-3, 'comp_1': 50e-3},
        'X_AR': 0.25,
    },
    'DeepLTS': {
        'proto': 'DeepLTS.p',
        'levels': 'DeepLTS.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -100e-3,
            'AR': -40e-3,
            'Ca': 125e-3,
            'GABA': -75e-3,  # Sanchez-Vives et al. 1997
        },
        'tauCa': {'all': 20e-3, 'comp_1': 50e-3},
        'X_AR': 0.25,
    },
    'NontuftedRS': {
        'proto': 'NontuftedRS.p',
        'depths': 'NontuftedRS.depths',
        'levels': 'NontuftedRS.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -95e-3,
            'AR': -35e-3,
            'Ca': 125e-3,
            'GABA': -75e-3,  # Sanchez-Vives et al. 1997
        },
        'tauCa': {'all': 20e-3, 'comp_1': 100e-3},
        'X_AR': 0.25,
    },
    'nRT': {
        'proto': 'nRT.p',
        'levels': 'nRT.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -95e-3,
            'AR': -35e-3,
            'Ca': 125e-3,
            'GABA': -75e-3,  # Sanchez-Vives et al. 1997
        },
        'tauCa': {
            'all': 1e-3 / 0.075,
            'comp_1': 100e-3,
        },
        'X_AR': 0.25,
    },
    'SpinyStellate': {
        'proto': 'SpinyStellate.p',
        'levels': 'SpinyStellate.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -100e-3,
            'AR': -40e-3,
            'Ca': 125e-3,
            'GABA': -75e-3,
        },
        'X_AR': 0.0,
        'tauCa': {'all': 20e-3, 'comp_1': 50e-3},
    },
    'SupAxoaxonic': {
        'proto': 'SupAxoaxonic.p',
        'levels': 'SupAxoaxonic.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -100e-3,
            'Ca': 125e-3,
            'AR': -40e-3,
            'GABA': -75e-3,
        },
        'tauCa': {'all': 20e-3, 'comp_1': 50e-3},
        'X_AR': 0.0,
    },
    'SupBasket': {
        'proto': 'SupBasket.p',
        'levels': 'SupBasket.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -100e-3,
            'AR': -40e-3,
            'Ca': 125e-3,
            'GABA': -75e-3,  # Sanchez-Vives et al. 1997
        },
        'tauCa': {'all': 20e-3, 'comp_1': 50e-3},
        'X_AR': 0.0,
    },
    'SupLTS': {
        'proto': 'SupLTS.p',
        'levels': 'SupLTS.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -100e-3,
            'Ca': 125e-3,
            'AR': -40e-3,  # dummy to set things back to original
            'GABA': -75e-3,  # Sanchez-Vives et al. 1997
        },
        'tauCa': {'all': 20e-3, 'comp_1': 50e-3},
        'X_AR': 0.25,
    },
    'SupPyrFRB': {
        'proto': 'SupPyrFRB.p',
        'depths': 'SupPyrFRB.depths',
        'levels': 'SupPyrFRB.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -95e-3,
            'AR': -35e-3,
            'Ca': 125e-3,
            'GABA': -81e-3,
        },
        'tauCa': {'all': 20e-3, 'comp_1': 100e-3},
    },
    'SupPyrRS': {
        'proto': 'SupPyrRS.p',
        'levels': 'SupPyrRS.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -95e-3,
            'Ca': 125e-3,
            'AR': -35e-3,
            'GABA': -81e-3,
        },
        'tauCa': {'all': 20e-3, 'comp_1': 100e-3},
    },
    'TCR': {
        'proto': 'TCR.p',
        'levels': 'TCR.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -95e-3,
            'AR': -35e-3,
            'Ca': 125e-3,
            'GABA': -81e-3,
        },
        'tauCa': {'all': 20e-3, 'comp_1': 50e-3},
        'X_AR': 0.25,
    },
    'TuftedIB': {
        'proto': 'TuftedIB.p',
        'levels': 'TuftedIB.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -95e-3,
            'AR': -35e-3,
            'Ca': 125e-3,
            'GABA': -75e-3,  # Sanchez-Vives et al. 1997
        },
        'tauCa': {
            'all': 1e-3 / 0.075,
            'comp_1': 100e-3,
            'comp_2': 1e-3 / 0.02,
            'comp_5': 1e-3 / 0.02,
            'comp_6': 1e-3 / 0.02,
        },
        'X_AR': 0.25,
    },
    'TuftedRS': {
        'proto': 'TuftedRS.p',
        'levels': 'TuftedRS.levels',
        'Ek': {
            'Na': 50e-3,
            'K': -95e-3,
            'AR': -35e-3,
            'Ca': 125e-3,
            'GABA': -75e-3,  # Sanchez-Vives et al. 1997
        },
        'tauCa': {
            'all': 1e-3 / 0.075,
            'comp_1': 100e-3,
            'comp_2': 1e-3 / 0.02,
            'comp_5': 1e-3 / 0.02,
            'comp_6': 1e-3 / 0.02,
        },
        'X_AR': 0.25,
    },
}


def assign_depths(proto, cell_spec, protodir='proto'):
    """Assign depth information to compartments"""
    fdepth = cell_spec.get('depth')
    flevels = cell_spec.get('levels')
    if fdepth is None:
        return
    dfile = cell_spec.get('depths')
    lfile = cell_spec.get('levels')
    if (dfile is not None) and (lfile is not None):
        depths = pd.read_csv(
            os.path.join(protodir, dfile),
            sep='\s+',
            names=['level', 'depth'],
            dtype={'level': np.int32, 'depth': np.float64},
        ).to_dict()
        levels = pd.read_csv(
            os.path.join(protodir, lfile),
            sep='\s+',
            names=['num', 'level'],
            dtype={'num': np.int32, 'level': np.int32},
        ).to_dict()
        for num, level in levels.items():
            depth = depths[level]
            comp = moose.element(f'{proto.path}/comp_{num}')
            comp.z = depth


def update_ek(proto, cell_spec):
    """Update Ek for channels based on that specified in the cell spec"""
    for ion, Ek in cell_spec['Ek'].items():
        channels = moose.wildcardFind(f'{proto.path}/#/{ion}#')
        for ch in channels:
            if not isinstance(ch, moose.CaConc):
                ch.Ek = Ek


def update_tau_ca(proto, cell_spec):
    """Update tau for Ca pool from cell spec"""
    tau_dict = cell_spec['tauCa']
    tau_all = tau_dict.pop('all')
    for ca_pool in moose.wildcardFind(f'{proto.path}/##[ISA=CaConc]'):
        ca_pool.tau = tau_all
        print(proto.name, 'Set tau Ca for all comp to', ca_pool.tau)
    for comp, tau in tau_dict.items():
        ca_pool = moose.element(f'{proto.path}/{comp}/CaPool')
        ca_pool.tau = tau
        print(f'{proto.name}: Set tau Ca for {comp} to {ca_pool.tau}')
    
    # to keep the cell_spec unaltered, put back the popped entry
    tau_dict['all'] = tau_all


def update_ar(cell, cell_spec):
    """Initial value of gating parameter is force-set for AR (as
    opposed to lookup value according to the voltage)"""
    if 'X_AR' in cell_spec:
        ar_chans = moose.wildcardFind(f'{cell.path}/##[FIELD(name)=AR]')
        for ch in ar_chans:
            ch.X = cell_spec['X_AR']
    

def get_cell(name, spec, parent='/library', protodir='proto'):
    """Returns a prototype cell with name `name` under `parent`,
    creating it if it does not exist."""
    cell = get_proto(name, parent)
    if cell:
        return cell

    init_channels()
    cell_path = f'{parent}/{name}'
    file_path = os.path.join(protodir, spec['proto'])
    proto = moose.loadModel(file_path, cell_path)
    assign_depths(proto, spec, protodir)
    update_ek(proto, spec)
    update_tau_ca(proto, spec)
    return proto


def init_cells():
    init_channels()
    tstart = time.perf_counter()
    cells = {}
    for name, spec in cell_spec.items():
        cells[name] = get_cell(name, spec=spec)
    tend = time.perf_counter()
    logger.debug(f'Created {len(cells)} prototype neurons in {tend - tstart} s')
    return cells


if __name__ == '__main__':
    init_cells()

#
# cells.py ends here
