# morphology_registry.py ---
#
# Filename: morphology_registry.py
# Description:
# Author: Subhasis Ray
# Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
# Created: Thu Apr 23 12:12:47 2026 (+0530)
#

# Code:

"""This shows how to use the morphology database bundled with moose."""

# %% Usual imports
import moose
import moose.morphologies as morph

# %% list the available named morphologies
morph.list()

# %% Query the morphology database
# All entries as a list of dicts
all_cells = morph.entries()

print('There are', len(all_cells), 'morphologies in the database')
# Just the names
names = [e['name'] for e in morph.entries()]
print('The names of the cell morphologies', names)

# Filter by field (case-insensitive substring match)
ca1_cells = morph.entries(cell_type='CA1')

print('CA1 neurons:', [info['name'] for info in ca1_cells])

rat_cells = morph.entries(species='rat')
print('Rat cells:', [info['name'] for info in rat_cells])

hippo = morph.entries(region='hippocampus')
print('Hippocampal cells from these papers:', [info['source'] for info in hippo])

# Single entry by exact name
info = morph.get('traub91_CA1')
print(info['description'])  # '19-compartment CA1 pyramidal (Traub et al. 1991)'
print(info['source'])


# %% Load a named model from the registry
cell = morph.load('traub91_CA1', '/neuron')
moose.delete(cell.root.path + '/spine')  # This is needed to avoid segfault at showfield
# %% Display the fields in the compartments
print(cell)

for comp in cell.root.children:
    print(comp.path, ': Rm=', comp.Rm, 'Ra=', comp.Ra)

#
# morphology_registry.py ends here
