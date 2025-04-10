# chemicalReaction.py ---
#
# Filename: chemicalReaction.py
# Description:
# Author: Subhasis Ray
# Created: Wed Jan 15 12:14:23 2025 (+0530)
#

# Code:
"""Simplest chemical kinetics model"""

import numpy as np
import matplotlib.pyplot as plt
import moose


def makeSimpleReactionModel(concA=1.0, concC=0.0, Kf=0.1, Kb=0.05):
    """This function demonstrate a simple reversible chemical reaction model

       Kf
    A <-> C
       Kb

    where A transforms into C with forward rate constant Kf and C
    transforms back to A with backward rate constant Kb.


    Paramters
    ---------
    concA: float
        initial concentration of A
    concC: float
        initial concentration of C
    Kf: float
        forward reaction rate constant
    Kb: float
        backward reaction rate constant

    Returns
    -------
    dict: {'model': the model container object,
           'data': list of data tables}
    """
    model = moose.Neutral('/model')  # model container
    comp = moose.CubeMesh('/model/comp')  # chemical compartment
    comp.volume = 1e-21  # the reaction is in a 0.1 um cube
    a = moose.Pool(f'{comp.path}/A')
    c = moose.Pool(f'{comp.path}/C')
    reac = moose.Reac(f'{comp.path}/reac')
    # Connect the substrate and product to the reaction
    moose.connect(reac, 'sub', a, 'reac')
    moose.connect(reac, 'prd', c, 'reac')

    # Setup solver for stoichiometric calculations
    stoich = moose.Stoich(f'{comp.path}/stoich')
    stoich.compartment = comp
    ksolve = moose.Ksolve(f'{comp.path}/ksolve')
    stoich.ksolve = ksolve
    stoich.reacSystemPath = f'{comp.path}/##'

    # Setup data recording
    data = moose.Neutral('/data')
    tabA = moose.Table2(f'{data.path}/A')
    tabC = moose.Table2(f'{data.path}/C')
    moose.connect(tabA, 'requestOut', a, 'getConc')
    moose.connect(tabC, 'requestOut', c, 'getConc')

    reac.Kf = Kf
    reac.Kb = Kb
    a.concInit = concA
    c.concInit = concC
    return dict(model=model, data=[tabA, tabC])


def main(simtime=100.0):
    """Main function to create the model, run the simulation, and plot
    the data.

    Parameters
    ----------
    simtime: float
        duration (in seconds) to simulate the reaction for.
    """
    modeldict = makeSimpleReactionModel()
    moose.reinit()
    moose.start(simtime)
    fig, ax = plt.subplots()
    t = np.arange(len(modeldict['data'][0].vector)) * modeldict['data'][0].dt
    for tab in modeldict['data']:
        ax.plot(t, tab.vector, label=f'[{tab.name}]')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concentration (mol/m^3)')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main(100)


#
# chemicalReaction.py ends here
