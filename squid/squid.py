# Filename: squid.py
# Description:
# Author: Subhasis Ray
# Maintainer: Dilawar Singh

import numpy as np
import moose

GAS_CONSTANT = 8.314
FARADAY = 9.65e4
CELSIUS_TO_KELVIN = 273.15


class IonChannel(object):
    """Enhanced version of HHChannel with setupAlpha that takes a dict
    of parameters."""

    def __init__(
        self, name, axon, specific_gbar, e_rev, Xpower, Ypower=0.0, Zpower=0.0
    ):
        """Instantuate an ion channel.

        name -- name of the channel.

        axon -- SquidAxon object that contains the channel.

        specific_gbar -- specific value of maximum conductance.

        e_rev -- reversal potential of the channel.

        Xpower -- exponent for the first gating parameter.

        Ypower -- exponent for the second gatinmg component.
        """
        self.path = f'{axon.path}/{name}'
        self.chan = moose.HHChannel(self.path)
        self.chan.Gbar = specific_gbar * axon.area
        self.chan.Ek = e_rev
        self.chan.Xpower = Xpower
        self.chan.Ypower = Ypower
        self.chan.Zpower = Zpower
        moose.connect(self.chan, 'channel', axon.compartment, 'channel')

    def setupAlpha(self, gate, params, vdivs, vmin, vmax):
        """Setup alpha and beta parameters of specified gate.

        gate -- 'X'/'Y'/'Z' string initial of the gate.

        params -- dict of parameters to compute alpha and beta, the rate constants for gates.

        vdivs -- number of divisions in the interpolation tables for alpha and beta parameters.

        vmin -- minimum voltage value for the alpha/beta lookup tables.

        vmax -- maximum voltage value for the alpha/beta lookup tables.
        """
        if gate == 'X' and self.chan.Xpower > 0:
            gate = moose.element(self.path + '/gateX')
        elif gate == 'Y' and self.chan.Ypower > 0:
            gate = moose.element(self.path + '/gateY')
        else:
            return False
        gate.setupAlpha(
            [
                params['A_A'],
                params['A_B'],
                params['A_C'],
                params['A_D'],
                params['A_F'],
                params['B_A'],
                params['B_B'],
                params['B_C'],
                params['B_D'],
                params['B_F'],
                vdivs,
                vmin,
                vmax,
            ]
        )
        return True

    @property
    def alpha_m(self):
        if self.chan.Xpower == 0:
            return np.array([])
        return np.array(moose.element(f'{self.path}/gateX').tableA)

    @property
    def beta_m(self):
        if self.chan.Xpower == 0:
            return np.array([])
        return np.array(moose.element(f'{self.path}/gateX').tableB) - np.array(
            moose.element(f'{self.path}/gateX').tableA
        )

    @property
    def alpha_h(self):
        if self.chan.Ypower == 0:
            return np.array([])
        return np.array(moose.element(f'{self.path}/gateY').tableA)

    @property
    def beta_h(self):
        if self.chan.Ypower == 0:
            return np.array([])
        return np.array(moose.element(f'{self.path}/gateY').tableB) - np.array(
            moose.element(f'{self.path}/gateY').tableA
        )


class SquidAxon(object):
    # can be -70 mV if not following original HH convention
    EREST_ACT = 0.0
    VMIN = -30.0
    VMAX = 120.0
    VDIVS = 150
    defaults = {
        'temperature': CELSIUS_TO_KELVIN + 6.3,
        'K_out': 10.0,
        'Na_out': 460.0,
        'K_in': 301.4,
        'Na_in': 70.97,
        'Cl_out': 540.0,
        'Cl_in': 100.0,
        'length': 500.0,  # um
        'diameter': 500.0,  # um
        'Em': EREST_ACT + 10.613,
        'initVm': EREST_ACT,
        'specific_cm': 1.0,  # uF/cm^2
        'specific_gl': 0.3,  # mmho/cm^2
        'specific_ra': 0.030,  # kohm-cm
        'specific_gNa': 120.0,  # mmho/cm^2
        'specific_gK': 36.0,  # mmho/cm^2
    }

    Na_m_params = {
        'A_A': 0.1 * (25.0 + EREST_ACT),
        'A_B': -0.1,
        'A_C': -1.0,
        'A_D': -25.0 - EREST_ACT,
        'A_F': -10.0,
        'B_A': 4.0,
        'B_B': 0.0,
        'B_C': 0.0,
        'B_D': 0.0 - EREST_ACT,
        'B_F': 18.0,
    }
    Na_h_params = {
        'A_A': 0.07,
        'A_B': 0.0,
        'A_C': 0.0,
        'A_D': 0.0 - EREST_ACT,
        'A_F': 20.0,
        'B_A': 1.0,
        'B_B': 0.0,
        'B_C': 1.0,
        'B_D': -30.0 - EREST_ACT,
        'B_F': -10.0,
    }
    K_n_params = {
        'A_A': 0.01 * (10.0 + EREST_ACT),
        'A_B': -0.01,
        'A_C': -1.0,
        'A_D': -10.0 - EREST_ACT,
        'A_F': -10.0,
        'B_A': 0.125,
        'B_B': 0.0,
        'B_C': 0.0,
        'B_D': 0.0 - EREST_ACT,
        'B_F': 80.0,
    }
    """Compartment class enhanced with specific values of passive
    electrical properties set and calculated using dimensions."""

    def __init__(self, path):
        #  moose.Compartment.__init__(self, path)
        self.path = path
        self.compartment = moose.Compartment(self.path)
        self.dt = self.compartment.dt
        self.temperature = SquidAxon.defaults['temperature']
        self.K_out = SquidAxon.defaults['K_out']
        self.Na_out = SquidAxon.defaults['Na_out']
        # Modified internal concentrations used to give HH values of
        # equilibrium constants from the Nernst equation at 6.3 deg C.
        # HH 1952a, p. 455
        self.K_in = SquidAxon.defaults['K_in']
        self.Na_in = SquidAxon.defaults['Na_in']
        self.Cl_out = SquidAxon.defaults['Cl_out']
        self.Cl_in = SquidAxon.defaults['Cl_in']

        self.compartment.length = SquidAxon.defaults['length']
        self.compartment.diameter = SquidAxon.defaults['diameter']
        self.compartment.Em = SquidAxon.defaults['Em']
        self.compartment.initVm = SquidAxon.defaults['initVm']

        self.specific_cm = SquidAxon.defaults['specific_cm']
        self.specific_gl = SquidAxon.defaults['specific_gl']
        self.specific_ra = SquidAxon.defaults['specific_ra']

        self.Na_channel = IonChannel('Na', self, 0.0, self.VNa, Xpower=3.0, Ypower=1.0)

        self.Na_channel.setupAlpha(
            'X', SquidAxon.Na_m_params, SquidAxon.VDIVS, SquidAxon.VMIN, SquidAxon.VMAX
        )

        self.Na_channel.setupAlpha(
            'Y', SquidAxon.Na_h_params, SquidAxon.VDIVS, SquidAxon.VMIN, SquidAxon.VMAX
        )

        self.K_channel = IonChannel('K', self, 0.0, self.VK, Xpower=4.0)

        self.K_channel.setupAlpha(
            'X', SquidAxon.K_n_params, SquidAxon.VDIVS, SquidAxon.VMIN, SquidAxon.VMAX
        )

        self.specific_gNa = SquidAxon.defaults['specific_gNa']
        self.specific_gK = SquidAxon.defaults['specific_gK']

    @classmethod
    def reversal_potential(cls, temp, c_out, c_in):
        """Compute the reversal potential based on Nernst equation."""
        # NOTE the 70 mV added for compatibility with original HH
        v = (
            (GAS_CONSTANT * temp / FARADAY) * 1000.0 * np.log(c_out / c_in)
            + 70.0
            + cls.EREST_ACT
        )
        return v

    @property
    def xarea(self):
        """Area of cross section in cm^2 when length and diameter are in um"""
        return (
            1e-8 * np.pi * self.compartment.diameter * self.compartment.diameter / 4.0
        )  # cm^2

    @property
    def area(self):
        """Area in cm^2 when length and diameter are in um"""
        return (
            1e-8 * self.compartment.length * np.pi * self.compartment.diameter
        )  # cm^2

    @property
    def specific_ra(self):
        return self.compartment.Ra * self.xarea / self.compartment.length

    @specific_ra.setter
    def specific_ra(self, value):
        self.compartment.Ra = value * self.compartment.length / self.xarea

    @property
    def specific_cm(self):
        return self.compartment.Cm / self.area

    @specific_cm.setter
    def specific_cm(self, value):
        self.compartment.Cm = value * self.area

    @property
    def specific_gl(self):
        return 1.0 / (self.compartment.Rm * self.area)

    @specific_gl.setter
    def specific_gl(self, value):
        self.compartment.Rm = 1.0 / (value * self.area)

    @property
    def specific_rm(self):
        return self.compartment.Rm * self.area

    @specific_rm.setter
    def specific_rm(self, value):
        self.compartment.Rm = value / self.area

    @property
    def specific_gNa(self):
        return self.Na_channel.chan.Gbar / self.area

    @specific_gNa.setter
    def specific_gNa(self, value):
        self.Na_channel.chan.Gbar = value * self.area

    @property
    def specific_gK(self):
        return self.K_channel.chan.Gbar / self.area

    @specific_gK.setter
    def specific_gK(self, value):
        self.K_channel.chan.Gbar = value * self.area

    @property
    def VK(self):
        """Reversal potential of K+ channels"""
        return SquidAxon.reversal_potential(self.temperature, self.K_out, self.K_in)

    @property
    def VNa(self):
        """Reversal potential of Na+ channels"""
        return SquidAxon.reversal_potential(self.temperature, self.Na_out, self.Na_in)

    def updateEk(self):
        """Update the channels' Ek"""
        self.Na_channel.chan.Ek = self.VNa
        self.K_channel.chan.Ek = self.VK
        # Special case for both channels blocked
        if self.K_channel.chan.Gbar == 0 and self.Na_channel.chan.Gbar == 0:
            self.compartment.Em = 0.0

    def get_celsius(self):
        return self.temperature - CELSIUS_TO_KELVIN

    def set_celsius(self, celsius):
        self.temperature = celsius + CELSIUS_TO_KELVIN

    celsius = property(get_celsius, set_celsius)
