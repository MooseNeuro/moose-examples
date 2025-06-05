# cell_test_util.py ---
#
# Filename: cell_test_util.py
# Description: Utility functions for testing single cells
# Author:
# Maintainer:
# Created: Mon Oct 15 15:03:09 2012 (+0530)
# Version:
# Last-Updated: Tue May 13 16:31:06 2025 (+0530)
#           By: Subhasis Ray
#     Update #: 433
# URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change log:
#
#
#
#

# Code:


from datetime import datetime
import time
import os
import sys
import uuid
import unittest
import numpy as np
from matplotlib import pyplot as plt
import pylab
import moose
import config
import cells
import testutils
from testutils import compare_cell_dump, step_run


def step_run(simtime, steptime, verbose=True, logger=None):
    """Run the simulation in steps of `steptime` for `simtime`."""
    clock = moose.Clock('/clock')
    if verbose:
        msg = 'Starting simulation for %g' % (simtime)
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
    ts = datetime.now()
    while clock.currentTime < simtime - steptime:
        ts1 = datetime.now()
        moose.start(steptime)
        te = datetime.now()
        td = te - ts1
        if verbose:
            msg = (
                'Simulated till %g. Left: %g. %g of simulation took: %g s'
                % (
                    clock.currentTime,
                    simtime - clock.currentTime,
                    steptime,
                    td.days * 86400 + td.seconds + 1e-6 * td.microseconds,
                )
            )
            if logger is None:
                print(msg)
            else:
                logger.info(msg)

    remaining = simtime - clock.currentTime
    if remaining > 0:
        if verbose:
            msg = 'Running the remaining %g.' % (remaining)
            if logger is None:
                print(msg)
            else:
                logger.info(msg)
        moose.start(remaining)
    te = datetime.now()
    td = te - ts
    dt = min([t for t in moose.element('/clock').dts if t > 0.0])
    if verbose:
        msg = 'Finished simulation of %g with minimum dt=%g in %g s' % (
            simtime,
            dt,
            td.days * 86400 + td.seconds + 1e-6 * td.microseconds,
        )
        if logger is None:
            print(msg)
        else:
            logger.info(msg)


def setup_current_step_model(model_path, data_path, celltype, pulse_list):
    """Setup a single cell simulation.

    Parameters
    ----------
    model_path: str
        Path of model container element

    data_path: str
        Path of data container element

    celltype: str
        Cell type

    pulseinfo: list of dicts
        Parameters to set up `PulseGen` for current injection.
        Entries should be: {'delay': delay,
                            'width': width,
                            'level': level}
        for as many pulses as needed.

    """
    classname = f'cells.{celltype}'
    print(
        'mc=',
        model_path,
        'dc=',
        data_path,
        'ct=',
        celltype,
        'pa=',
        pulse_list,
        'classname=',
        classname,
    )
    cell_class = eval(classname)
    cell = cell_class(f'{model_path}/{celltype}')
    pulsegen = moose.PulseGen(f'{model_path}/pulse')
    for ii in range(len(pulse_list)):
        pulsegen.delay[ii] = pulse_list[ii]['delay']
        pulsegen.width[ii] = pulse_list[ii]['width']
        pulsegen.level[ii] = pulse_list[ii]['level']
    moose.connect(pulsegen, 'output', cell.soma, 'injectMsg')
    presyn_vm = moose.Table(f'{data_path}/presynVm')
    soma_vm = moose.Table(f'{data_path}/somaVm')
    moose.connect(presyn_vm, 'requestOut', cell.presynaptic, 'getVm')
    moose.connect(soma_vm, 'requestOut', cell.soma, 'getVm')
    pulse_table = moose.Table(f'{data_path}/injectCurrent')
    moose.connect(pulse_table, 'requestOut', pulsegen, 'getOutputValue')
    return {
        'cell': cell,
        'stimulus': pulsegen,
        'presynVm': presyn_vm,
        'somaVm': soma_vm,
        'injectionCurrent': pulse_table,
    }


class SingleCellCurrentStepTest(unittest.TestCase):
    """Base class for simulating a single cell with step current injection.

    Attributes
    ----------
    pulse_list: list of dicts
        Specification for current pulses. Each entry should be a dict
        of the form {'delay': delay, 'width': width, 'level': level}
        for the corresponding pulse.

    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        # By default give a 1 nA current injection for 100 ms,
        # starting at 100 ms, make the delay for the second pulse very
        # large to avoid a second pulse within realistic simulation
        # runtime
        self.pulse_list = [
            {'delay': 100e-3, 'width': 400e-3, 'level': 1e-9},
            {'delay': 1e9, 'width': 0, 'level': 0},
        ]
        self.solver = config.simulationSettings.method
        self.simdt = None
        self.plotdt = None
        self.tseries = []

    def setUp(self):
        self.test_container = moose.Neutral(f'test_{self.celltype}')
        self.model_container = moose.Neutral(
            f'{self.test_container.path}/model'
        )

        self.data_container = moose.Neutral(f'{self.test_container.path}/data')
        params = setup_current_step_model(
            self.model_container.path,
            self.data_container.path,
            self.celltype,
            self.pulse_list,
        )
        self.cell = params['cell']
        self.cell.dump_cell(f'{self.cell.name}.csv')
        for ch in moose.wildcardFind(
            self.cell.soma.path + '/##[ISA=ChanBase]'
        ):
            config.logger.debug('%s Ek = %g' % (ch.path, ch.Ek))
        for ch in moose.wildcardFind(self.cell.soma.path + '/##[ISA=CaConc]'):
            config.logger.debug('%s tau = %g' % (ch.path, ch.tau))

        self.somaVmTab = params['somaVm']
        self.presynVmTab = params['presynVm']
        self.injectionTab = params['injectionCurrent']
        self.pulsegen = params['stimulus']
        # setup_clocks(self.simdt, self.plotdt)
        # assign_clocks(self.model_container, self.data_container, self.solver)

    def tweak_stimulus(self, pulse_list):
        """Update the pulsegen for this model with new (delay, width,
        level) values specified in `pulse_list` list."""
        self.pulsegen.count = len(pulse_list)
        for ii in range(len(pulse_list)):
            self.pulsegen.delay[ii] = pulse_list[ii]['delay']
            self.pulsegen.width[ii] = pulse_list[ii]['width']
            self.pulsegen.level[ii] = pulse_list[ii]['level']

    def schedule(self, simdt=None, plotdt=None, solver=None):
        config.logger.info(
            f'Scheduling: simdt={simdt}, plotdt={plotdt}, solver={solver}'
        )
        if simdt is not None:
            self.simdt = simdt
            for tick in range(moose.element('/clock').numTicks):
                moose.setClock(tick, simdt)
        if plotdt is not None:
            self.plotdt = plotdt
            tables = moose.wildcardFind(
                f'{self.data_container.path}/#[TYPE=Table]'
            )
            assert len(tables) > 0, 'No data tables found. Abort.'
            moose.setClock(tables[0].tick, plotdt)
        if solver is not None:
            self.solver = solver
        if self.solver == 'hsolve':
            self.hsolve = moose.HSolve(f'{self.cell.path}/solver')
            self.hsolve.dt = simdt
            self.hsolve.target = self.cell.path

    def runsim(self, simtime, stepsize=0.1, pulse_list=None):
        """Run the simulation for `simtime`. Save the data at the
        end."""
        config.logger.info(
            f'running: simtime={simtime}, stepsize={stepsize},'
            f' pulses={pulse_list}'
        )
        self.simtime = simtime
        if pulse_list is not None:
            self.tweak_stimulus(pulse_list)
        for ii in range(self.pulsegen.count):
            config.logger.info(
                'pulse[%d]: delay=%g, width=%g, level=%g'
                % (
                    ii,
                    self.pulsegen.delay[ii],
                    self.pulsegen.width[ii],
                    self.pulsegen.level[ii],
                )
            )
        config.logger.info('Start reinit')
        self.schedule(self.simdt, self.plotdt, self.solver)
        moose.reinit()
        config.logger.info('Finished reinit')
        ts = datetime.now()
        step_run(simtime, simtime / 10.0, verbose=True)
        # The sleep is required to get all threads to end
        while moose.isRunning():
            time.sleep(0.1)
        te = datetime.now()
        td = te - ts
        config.logger.info(
            f'Simulation time of {simtime} s'
            f' at simdt={self.simdt}'
            f' with solver {self.solver}:'
            f'{td.seconds + td.microseconds * 1e-6}'
        )

    def savedata(self):
        # Now save the data
        for table_id in self.data_container.children:
            ts = np.linspace(0, self.simtime, len(table_id[0].vector))
            data = np.vstack((ts, table_id[0].vector))
            fname = os.path.join(
                config.data_dir,
                '%s_%s_%s_%s.dat'
                % (
                    self.celltype,
                    table_id[0].name,
                    self.solver,
                    config.filename_suffix,
                ),
            )
            np.savetxt(fname, np.transpose(data))
            config.logger.info('Saved %s in %s' % (table_id[0].name, fname))

    def plot_vm(self):
        """Plot Vm for presynaptic compartment and soma - along with
        the same in NEURON simulation if possible."""
        pylab.subplot(211)
        pylab.title('Soma Vm')
        self.tseries = np.linspace(0, self.simtime, len(self.somaVmTab.vector))
        pylab.plot(
            self.tseries * 1e3,
            self.somaVmTab.vector * 1e3,
            label='Vm (mV) - moose',
        )
        pylab.plot(
            self.tseries * 1e3,
            self.injectionTab.vector * 1e9,
            label='Stimulus (nA)',
        )
        try:
            nrn_data = np.loadtxt(
                f'../nrn/data/{self.celltype}_soma_Vm.dat'
            )
            nrn_indices = np.nonzero(nrn_data[:, 0] <= self.tseries[-1] * 1e3)[
                0
            ]
            pylab.plot(
                nrn_data[nrn_indices, 0],
                nrn_data[nrn_indices, 1],
                label='Vm (mV) - neuron',
            )
        except IOError:
            print('No neuron data found.')
        pylab.legend()
        pylab.subplot(212)
        pylab.title('Presynaptic Vm')
        pylab.plot(
            self.tseries * 1e3,
            self.presynVmTab.vector * 1e3,
            label='Vm (mV) - moose',
        )
        pylab.plot(
            self.tseries * 1e3,
            self.injectionTab.vector * 1e9,
            label='Stimulus (nA)',
        )
        try:
            fname = os.path.join(
                config.mydir,
                '..',
                'nrn',
                'data',
                '%s_presynaptic_Vm.dat' % (self.celltype),
            )
            nrn_data = np.loadtxt(fname)
            nrn_indices = np.nonzero(nrn_data[:, 0] <= self.tseries[-1] * 1e3)[
                0
            ]
            pylab.plot(
                nrn_data[nrn_indices, 0],
                nrn_data[nrn_indices, 1],
                label='Vm (mV) - neuron',
            )
        except IOError:
            print('No neuron data found.')
        pylab.legend()
        pylab.show()


#
# cell_test_util.py ends here
