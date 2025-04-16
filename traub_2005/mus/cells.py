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
import moose

from channels import get_proto, init_channels

cell_spec = {
    'DeepAxoaxonic': {
        'proto': 'DeepAxoaxonic.p',
    },
    'DeepBasket': {'proto': 'DeepBasket.p'},
    'DeepLTS': {'proto': 'DeepLTS.p', 'levels': 'DeepLTS.levels'},
    'NontuftedRS': {
        'proto': 'NontuftedRS.p',
        'depths': 'NontuftedRS.depths',
        'levels': 'NontuftedRS.levels',
    },
    'nRT': {'proto': 'nRT.p'},
    'SpinyStellate': {
        'proto': 'SpinyStellate.p',
        'levels': 'SpinyStellate.levels',
    },
    'SupAxoaxonic': {
        'proto': 'SupAxoaxonic.p',
        'levels': 'SupAxoaxonic.levels',
    },
    'SupBasket': {'proto': 'SupBasket.p'},
    'SupLTS': {'proto': 'SupLTS.p'},
    'SupPyrFRB': {
        'proto': 'SupPyrFRB.p',
        'depths': 'SupPyrFRB.depths',
        'levels': 'SupPyrFRB.levels',
    },
    'SupPyrRS': {'proto': 'SupPyrRS.p'},
    'TCR': {
        'proto': 'TCR.p',
        'levels': 'TCR.levels',
    },
    'TuftedIB': {
        'proto': 'TuftedIB.p',
        'levels': 'TuftedIB.levels',
    },
    'TuftedRS': {'proto': 'TuftedRS.p'},
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


def get_cell(name, spec, parent='/library', protodir='proto'):
    """Returns a prototype cell with name `name` under `parent`, creating it if it does not exist."""
    cell = get_proto(name, parent)
    if cell:
        return cell

    init_channels()
    cell_path = f'{parent}/{name}'
    file_path = os.path.join(protodir, spec['proto'])
    proto = moose.loadModel(file_path, cell_path)
    assign_depths(proto, spec, protodir)


def init_cells():
    init_channels()
    cells = {}
    for name, spec in cell_spec.items():
        cells[name] = get_cell(name, spec=spec)
    return cells



#
# cells.py ends here
