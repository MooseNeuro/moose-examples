# icg_channels.py ---
#
# Filename: icg_channel_registry.py
# Description:
# Author: Subhasis Ray
# Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
# Created: Thu Apr 23 12:11:29 2026 (+0530)
#

# Code:
"""This example shows how to use the builtin channel database from Ion Channel Genealogy"""

#%% Usual imports
import moose
import moose.channels as chan

#%% ── Discovery ─────────────────────────────────────────────────────────────────

# What ion classes exist?
chan.list_ion_classes()          # ['Ca', 'IH', 'K', 'KCa', 'Na']

# Search — returns list of dicts; show=False suppresses the printed table
results = chan.search(ion_class='Na', show=False)
results = chan.search(author='Traub', show=False)
results = chan.search(author='Traub', ion_class='K', show=False)
results = chan.search(model_id=45539, show=False)

# What's in a result dict?
r = results[0]
r['model_id']                    # int
r['meta']                        # dict with author, year, paper, url, …
r['channels']                    # dict: {suffix: [rows]}  — one row per gate

# List all suffixes in a result
list(r['channels'].keys())       # e.g. ['naf', 'nap']

#%% ── Inspect a channel without building MOOSE objects ──────────────────────────

chan.info(results[0])            # prints gate/power/expression summary
chan.info(45539, suffix='naf')   # same, by model_id directly

inf_expr, tau_expr = chan.get_expressions(45539, 'naf', 'm')
print(inf_expr)   # e.g. "1 / (1 + exp((v - vh) / k))"
print(tau_expr)

#%% ── Load and insert ───────────────────────────────────────────────────────────
import moose
import moose.morphologies as morph

cell = morph.load('traub91_CA1', '/neuron')

# Build prototype once, insert into all compartments
chans = chan.load(cell.compartments,
                  model_id=45539, suffix='naf',
                  gbar=120e-12, Ek=0.05)

# Distance-dependent gbar in apical dendrites only
chans = chan.load(cell.select('##[TYPE=Compartment]'),
                  model_id=45539, suffix='kdr',
                  Ek=0.05,
                  gbar=lambda c: 40e-12 * morph.surface_area(c))

#%% ── After loading — inspect what's in /library ────────────────────────────────
protos = chan.list_prototypes()
for p in protos:
    print(p)   # dicts with name, model_id, suffix, ion_class, Ek, …

    #
# icg_channel_registry.py ends here
