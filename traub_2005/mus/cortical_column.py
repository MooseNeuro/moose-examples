# cortical_column.py ---
#
# Filename: cortical_column.py
# Description:
# Author: Subhasis Ray
# Created: Sat Apr 26 09:01:43 2025 (+0530)
#

# Code:
"""Implements the network described in Traub et al., 2005"""
import time
import numpy as np
import moose
import cells
from config import logger


#: number of cells of each type in original model
orig_cell_counts = {
    'SupPyrRS': 1000,
    'SupPyrFRB': 50,
    'SupBasket': 90,
    'SupAxoaxonic': 90,
    'SupLTS': 90,
    'SpinyStellate': 240,
    'TuftedIB': 800,
    'TuftedRS': 200,
    'DeepBasket': 100,
    'DeepAxoaxonic': 100,
    'DeepLTS': 100,
    'NontuftedRS': 500,
    'TCR': 100,
    'nRT': 100,
}


#: number of cells of each type
#: modify this to reduce model size
cell_counts = {
    'SupPyrRS': 1000,
    'SupPyrFRB': 50,
    'SupBasket': 90,
    'SupAxoaxonic': 90,
    'SupLTS': 90,
    'SpinyStellate': 240,
    'TuftedIB': 800,
    'TuftedRS': 200,
    'DeepBasket': 100,
    'DeepAxoaxonic': 100,
    'DeepLTS': 100,
    'NontuftedRS': 500,
    'TCR': 100,
    'nRT': 100,
}

# fmt: off
#: The connection spec is
#: presynaptic_celltype: {
#:    postsynpatic_celltype: {
#:        'npre': number of presynaptic cells per postsynpatic neuron,
#:        'comps': allowed compartments on the postsynaptic neuron,
#:    }
#: }
#:

connection_spec = {
    # Keys are presynaptic cell type
    'SupPyrRS': {
        'SupPyrRS': {
            'npre': 50,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27,
                28, 29, 30, 31, 32, 33, 10, 11, 12, 13, 22, 23, 24, 25, 34, 35,
                36, 37
            ]
        },
        'SupPyrFRB': {
            'npre': 50,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27,
                28, 29, 30, 31, 32, 33, 10, 11, 12, 13, 22, 23, 24, 25, 34, 35,
                36, 37
            ]
        },
        'SupBasket': {
            'npre': 90,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupAxoaxonic': {
            'npre': 90,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupLTS': {
            'npre': 90,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SpinyStellate': {
            'npre': 3,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'TuftedIB': {
            'npre': 60,
            'comps': [39, 40, 41, 42, 43, 44, 45, 46]
        },
        'TuftedRS': {
            'npre': 60,
            'comps': [39, 40, 41, 42, 43, 44, 45, 46]
        },
        'DeepBasket': {
            'npre': 30,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepAxoaxonic': {
            'npre': 30,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepLTS': {
            'npre': 30,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'NontuftedRS': {
            'npre': 3,
            'comps': [38, 39, 40, 41, 42, 43, 44]
        },
    },
    'SupPyrFRB': {
        'SupPyrRS': {
            'npre': 5,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27,
                28, 29, 30, 31, 32, 33, 10, 11, 12, 13, 22, 23, 24, 25, 34, 35,
                36, 37
            ]
        },
        'SupPyrFRB': {
            'npre': 5,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27,
                28, 29, 30, 31, 32, 33, 10, 11, 12, 13, 22, 23, 24, 25, 34, 35,
                36, 37
            ]
        },
        'SupBasket': {
            'npre': 5,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupAxoaxonic': {
            'npre': 5,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupLTS': {
            'npre': 5,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SpinyStellate': {
            'npre': 1,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'TuftedIB': {
            'npre': 3,
            'comps': [39, 40, 41, 42, 43, 44, 45, 46]
        },
        'TuftedRS': {
            'npre': 3,
            'comps': [39, 40, 41, 42, 43, 44, 45, 46]
        },
        'DeepBasket': {
            'npre': 3,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepAxoaxonic': {
            'npre': 3,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepLTS': {
            'npre': 3,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'NontuftedRS': {
            'npre': 1,
            'comps': [38, 39, 40, 41, 42, 43, 44]
        },
    },
    'SupBasket': {
        'SupPyrRS': {
            'npre': 20,
            'comps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 38, 39]
        },
        'SupPyrFRB': {
            'npre': 20,
            'comps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 38, 39]
        },
        'SupBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SpinyStellate': {
            'npre': 20,
            'comps': [1, 2, 15, 28, 41]
        },
    },
    'SupAxoaxonic': {
        'SupPyrRS': {
            'npre': 20,
            'comps': [69]
        },
        'SupPyrFRB': {
            'npre': 20,
            'comps': [69]
        },
        'SpinyStellate': {
            'npre': 5,
            'comps': [54]
        },
        'TuftedIB': {
            'npre': 5,
            'comps': [56]
        },
        'TuftedRS': {
            'npre': 5,
            'comps': [56]
        },
        'NontuftedRS': {
            'npre': 5,
            'comps': [45]
        },
    },
    'SupLTS': {
        'SupPyrRS': {
            'npre': 20,
            'comps': [
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 66, 67, 68
            ]
        },
        'SupPyrFRB': {
            'npre': 20,
            'comps': [
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 66, 67, 68
            ]
        },
        'SupBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53
            ]
        },
        'SupAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53
            ]
        },
        'SupLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53
            ]
        },
        'SpinyStellate': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53
            ]
        },
        'TuftedIB': {
            'npre': 20,
            'comps': [
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55
            ]
        },
        'TuftedRS': {
            'npre': 20,
            'comps': [
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55
            ]
        },
        'DeepBasket': {
            'npre': 20,
            'comps': [
                8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 47,
                48, 49, 50, 51
            ]
        },
        'DeepAxoaxonic': {
            'npre': 20,
            'comps': [
                8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 47,
                48, 49, 50, 51
            ]
        },
        'DeepLTS': {
            'npre': 20,
            'comps': [
                8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 47,
                48, 49, 50, 51
            ]
        },
        'NontuftedRS': {
            'npre': 20,
            'comps': [
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44
            ]
        },
    },
    'SpinyStellate': {
        'SupPyrRS': {
            'npre': 20,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27,
                28, 29, 30, 31, 32, 33
            ]
        },
        'SupPyrFRB': {
            'npre': 20,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27,
                28, 29, 30, 31, 32, 33
            ]
        },
        'SupBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SpinyStellate': {
            'npre': 30,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'TuftedIB': {
            'npre': 20,
            'comps': [7, 8, 9, 10, 11, 12, 36, 37, 38, 39, 40, 41]
        },
        'TuftedRS': {
            'npre': 20,
            'comps': [7, 8, 9, 10, 11, 12, 36, 37, 38, 39, 40, 41]
        },
        'DeepBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'NontuftedRS': {
            'npre': 20,
            'comps': [37, 38, 39, 40, 41]
        },
    },
    'TuftedIB': {
        'SupPyrRS': {
            'npre': 2,
            'comps': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        },
        'SupPyrFRB': {
            'npre': 2,
            'comps': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        },
        'SupBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SpinyStellate': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'TuftedIB': {
            'npre': 50,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
            ]
        },
        'TuftedRS': {
            'npre': 20,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
            ]
        },
        'DeepBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'NontuftedRS': {
            'npre': 20,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44
            ]
        },
    },
    'TuftedRS': {
        'SupPyrRS': {
            'npre': 2,
            'comps': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        },
        'SupPyrFRB': {
            'npre': 2,
            'comps': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        },
        'SupBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SpinyStellate': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'TuftedIB': {
            'npre': 20,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
            ]
        },
        'TuftedRS': {
            'npre': 10,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
            ]
        },
        'DeepBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'NontuftedRS': {
            'npre': 20,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44
            ]
        },
    },
    'DeepBasket': {
        'SpinyStellate': {
            'npre': 20,
            'comps': [1, 2, 15, 28, 41]
        },
        'TuftedIB': {
            'npre': 20,
            'comps': [1, 2, 3, 4, 5, 6, 35, 36]
        },
        'TuftedRS': {
            'npre': 20,
            'comps': [1, 2, 3, 4, 5, 6, 35, 36]
        },
        'DeepBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'NontuftedRS': {
            'npre': 20,
            'comps': [1, 2, 3, 4, 5, 6, 35, 36]
        },
    },
    'DeepAxoaxonic': {
        'SupPyrRS': {
            'npre': 5,
            'comps': [69]
        },
        'SupPyrFRB': {
            'npre': 5,
            'comps': [69]
        },
        'SpinyStellate': {
            'npre': 5,
            'comps': [54]
        },
        'TuftedIB': {
            'npre': 5,
            'comps': [56]
        },
        'TuftedRS': {
            'npre': 5,
            'comps': [56]
        },
        'NontuftedRS': {
            'npre': 5,
            'comps': [45]
        },
    },
    'DeepLTS': {
        'SupPyrRS': {
            'npre': 10,
            'comps': [
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 66, 67, 68
            ]
        },
        'SupPyrFRB': {
            'npre': 10,
            'comps': [
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 66, 67, 68
            ]
        },
        'SupBasket': {
            'npre': 10,
            'comps': [
                8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 47,
                48, 49, 50, 51
            ]
        },
        'SupAxoaxonic': {
            'npre': 10,
            'comps': [
                8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 47,
                48, 49, 50, 51
            ]
        },
        'SupLTS': {
            'npre': 10,
            'comps': [
                8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 47,
                48, 49, 50, 51
            ]
        },
        'SpinyStellate': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53
            ]
        },
        'TuftedIB': {
            'npre': 20,
            'comps': [
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55
            ]
        },
        'TuftedRS': {
            'npre': 20,
            'comps': [
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55
            ]
        },
        'DeepBasket': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53
            ]
        },
        'DeepAxoaxonic': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53
            ]
        },
        'DeepLTS': {
            'npre': 20,
            'comps': [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53
            ]
        },
        'NontuftedRS': {
            'npre': 20,
            'comps': [
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44
            ]
        },
    },
    'NontuftedRS': {
        'SupPyrRS': {
            'npre': 10,
            'comps': [41, 42, 43, 44]
        },
        'SupPyrFRB': {
            'npre': 10,
            'comps': [41, 42, 43, 44]
        },
        'SupBasket': {
            'npre': 10,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupAxoaxonic': {
            'npre': 10,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SupLTS': {
            'npre': 10,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'SpinyStellate': {
            'npre': 10,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'TuftedIB': {
            'npre': 10,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
            ]
        },
        'TuftedRS': {
            'npre': 10,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
            ]
        },
        'DeepBasket': {
            'npre': 10,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepAxoaxonic': {
            'npre': 10,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'DeepLTS': {
            'npre': 10,
            'comps': [
                5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35,
                36, 44, 45, 46, 47, 48, 49
            ]
        },
        'NontuftedRS': {
            'npre': 20,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44
            ]
        },
        'TCR': {
            'npre': 20,
            'comps': [
                6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26,
                27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 58, 59, 60, 61, 62, 63, 64, 65, 66, 71, 72, 73, 74,
                75, 76, 77, 78, 79, 84, 85, 86, 87, 88, 89, 90, 91, 92, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 110, 111, 112, 113, 114, 115,
                116, 117, 118, 123, 124, 125, 126, 127, 128, 129, 130, 131
            ]
        },
        'nRT': {
            'npre': 20,
            'comps': [2, 3, 4, 15, 16, 17, 28, 29, 30, 41, 42, 43]
        },
    },
    'TCR': {
        'SupPyrRS': {
            'npre': 10,
            'comps': [
                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                61, 62, 63, 64, 65, 66, 67, 68
            ]
        },
        'SupPyrFRB': {
            'npre': 10,
            'comps': [
                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                61, 62, 63, 64, 65, 66, 67, 68
            ]
        },
        'SupBasket': {
            'npre': 10,
            'comps': [2, 3, 4, 15, 16, 17, 28, 29, 30, 41, 42, 43]
        },
        'SupAxoaxonic': {
            'npre': 10,
            'comps': [2, 3, 4, 15, 16, 17, 28, 29, 30, 41, 42, 43]
        },
        'SpinyStellate': {
            'npre': 20,
            'comps': [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                52, 53
            ]
        },
        'TuftedIB': {
            'npre': 10,
            'comps': [47, 48, 49, 50, 51, 52, 53, 54, 55]
        },
        'TuftedRS': {
            'npre': 10,
            'comps': [47, 48, 49, 50, 51, 52, 53, 54, 55]
        },
        'DeepBasket': {
            'npre': 20,
            'comps': [2, 3, 4, 15, 16, 17, 28, 29, 30, 41, 42, 43]
        },
        'DeepAxoaxonic': {
            'npre': 10,
            'comps': [2, 3, 4, 15, 16, 17, 28, 29, 30, 41, 42, 43]
        },
        'NontuftedRS': {
            'npre': 10,
            'comps': [40,41,42,43,44]
        },
        'nRT': {
            'npre': 25,
            'comps': [2, 3, 4, 15, 16, 17, 28, 29, 30, 41, 42, 43]
        },
    },
    'nRT': {
        'TCR': {
            'npre': 15,
            'comps': [1, 2, 15, 28, 41, 54, 67, 80, 93, 106, 119]
        },
        'nRT': {
            'npre': 10,
            'comps': [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53
            ]
        },
    },
}
# fmt: on


rng = np.random.default_rng()


def create_neuron_populations(cell_counts, model_root='/model', scale=0.1):
    """Create cel populations using counts in `popspec` dict.
    Scale the population sizes by `scale`"""
    cells.init_cells()
    tstart = time.perf_counter()
    populations = {}
    if not moose.exists(model_root):
        moose.Neutral(model_root)
    for name, count in cell_counts.items():
        proto = moose.element(f'/library/{name}')
        populations[name] = [
            moose.copy(proto, model_root, f'{name}_{ii:03d}')
            for ii in range(int(scale * count))
        ]
        logger.info(f'{name}: Created {count} copies')
    tend = time.perf_counter()
    logger.info(f'Finished creating neuron populations in {tend - tstart} s')
    return populations


def connect_populations(connspec, population_dict, rng=rng):
    """Connect the neuronal populations using connection specification
    in `connspec`.  `population_dict` maps celltype name to the list
    of cells of theis type

    """
    ttot = 0.0
    for pre_type, specs in connspec.items():
        for post_type, conn_info in specs.items():
            tstart = time.perf_counter()
            assert (
                conn_info['npre'] == 0 and len(conn_info['comps']) == 0
            ) or (
                conn_info['npre'] > 0 and len(conn_info['comps']) > 0
            ), f'0 target comp or precell for {pre_type}->{post_type}'
            conn_prob = float(conn_info['npre'])/orig_cell_counts[pre_type]
            npre = int(len(population_dict[pre_type]) * conn_prob)
            for cell in population_dict[post_type]:
                idx = rng.choice(
                    range(len(population_dict[pre_type])),
                    size=int(npre)
                )
                pre_list = [population_dict[pre_type][ii] for ii in idx]
                comp_list = rng.choice(
                    conn_info['comps'], size=npre
                )
                for pre_cell, comp_name in zip(pre_list, comp_list):
                    post_comp = moose.element(f'{cell.path}/comp_{comp_name}')
                    spikegen = moose.wildcardFind(
                        f'{pre_cell.path}/##[ISA=SpikeGen]'
                    )
                    assert len(spikegen) == 1
                    spikegen = spikegen[0]
                    # Here I am creating an independent synachan for each presynaptic neuron.
                    # Could be a single synchan for one presynaptic population.
                    # TODO: Compare results and performance.
                    synchan_path = f'{post_comp.path}/syn_{pre_cell.name}'
                    try:
                        synhandler = moose.element(f'{synchan_path}/synh')
                        synchan = moose.element(synchan_path)
                    except:
                        synchan = moose.SynChan(synchan_path)
                        # TODO: set synchan properties
                        synhandler = moose.SimpleSynHandler(
                            f'{synchan_path}/synh'
                        )
                        moose.connect(
                            synhandler, 'activationOut', synchan, 'activation'
                        )
                    # numSynapses must be incremented first to
                    # allocate synapse. We could also set the
                    # numSynapses right away at creation of the
                    # synhandler
                    synhandler.numSynapses += 1
                    moose.connect(
                        spikegen,
                        'spikeOut',
                        synhandler.synapse[synhandler.numSynapses - 1],
                        'addSpike',
                    )
            tend = time.perf_counter()
            ttot += tend - tstart
            logger.debug(
                f'Connected {pre_type} population {post_type} in {tend - tstart} s'
            )
    logger.debug('Total time to setup connections', ttot, 's')


def make_net(cell_counts, connection_spec, model_root='/model'):
    populations = create_neuron_populations(cell_counts, model_root=model_root)
    connect_populations(connection_spec, populations)
    return moose.element(model_root)


if __name__ == '__main__':
    model_root = '/model'
    make_net(cell_counts, connection_spec, model_root)

#
# cortical_column.py ends here
