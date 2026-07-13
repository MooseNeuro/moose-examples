# Traub 2005 model in MOOSE

Redesigned by Subhasis Ray in 2025


The original moose implementation of the Single Column Thalamocortical
Model by Traub et al, 2005 used a lot of metaprogramming with
metafix. While this was clever, the code turned out to be hard to read
and maintain. In the mean time, MOOSE went through many changes and
eveolved to make some things simpler to implement.

This directory contains a reimplementation of the Traub et al 2005
model using modern Python (Python 3) and MOOSE > 4.3.0.

## Network (`cortical_column.py`)
The full network is parameterized and created in the
`cortical_column.py` file. The `orig_cell_conts` dict has the number
of cells of each type in the original model for reference. The actual
numbers used are in the `cell_counts` dict. Modify this to change the
proprtion and number of cells actually simulated. The original cell
counts are:

| serial no | celltype      | count | description                                                         |
|-----------|---------------|-------|---------------------------------------------------------------------|
| 1         | SupPyrRS      | 1000  | Regular-spiking pyramidal neurons in superficial layers             |
| 2         | SupPyrFRB     | 50    | Fast, rhythmically bursting pyramidal neurons in superficial layers |
| 3         | SupBasket     | 90    | Basket cells in superficial layers                                  |
| 4         | SupAxoaxonic  | 90    | Axoaxonic neurons in superficial layers                             |
| 5         | SupLTS        | 90    | Low-threshold spiking neurons in superficial layers                 |
| 6         | SpinyStellate | 240   | Spiny stellate neurons                                              |
| 7         | TuftedIB      | 800   | Tufted intrinsically bursting neurons                               |
| 8         | TuftedRS      | 200   | Tufted regular-spiking neurons                                      |
| 9         | DeepBasket    | 100   | Basket cells in deep layers                                         |
| 10        | DeepAxoaxonic | 100   | Axoaxonic neurons in deep layers                                    |
| 11        | DeepLTS       | 100   | Low-threshold spiking neurons in deep layers                        |
| 12        | NontuftedRS   | 500   | Non-tufted regular-spiking neurons                                  |
| 13        | TCR           | 100   | Thalamocortical relay cells                                         |
| 14        | nRT           | 100   | neurons of the reticulo-thalamic neucleus)                          |

The specified numbers in `cell_count` can be further scaled by passing
the `scale` parameter to the `cortical_column.make_net(...)` function.
### Prototypes of neuron classes
The neuron prototypes (compartmental structure, channel distribution)
are specified in GENESIS prototype files (extension `.p`) in the
folder `proto`. These are loaded and instantiated in the `cells.py`
file. Customization of the ionic reversal potentials and time constant
for Ca2+ decay is specified in the `cell_spec` dict in that file.

## Channels (`channels.py`)
Each celltype in the model has many channels of various types
(voltage-dependent Na+, K+, and Ca2+, Ca2+ dependent K+
channels). These are specified in the `channel_spec` dict in
`channels.py`, and instantiated by utility functions in that file.

## Running the model
The model can be simulated by running the script `run_traubnet.py`. It
takes up to 3 positional parameters: (1) runtime in seconds, (2)
scaling factor for the network (population sizes of all cell types),
(3) fraction of cells of all types to record membrane potential from,
and (4) output filename. For example,

`python run_traubnet.py 1.0 0.1 0.5 traubnet_0.1_data.h5`

will setup a 1/10-th size network and simulate it for 1 second and
dump the data into `traubnet_0.1_data.h5`, recording somatic Vm of
half of the cells of each type.

As of 2026, the full model takes about 2 hours (~7000 seconds) on a
MacBook pro with Apple M4 Pro with 24 GB RAM running Darwin Kernel
Version 25.5.0. The actual simulation takes up over 7 GB of RAM.

## Animated viasualization
The dumped data can be visualized by running another script (this
requires `pyvista`) with the script `display_traubnet.py`.
