# channels.py ---
#
# Filename: channels.py
# Description:
# Author: Subhasis Ray
# Created: Thu Apr 10 14:18:58 2025 (+0530)
#

# Code:
"""Channels for Traub 2005 model"""
import time
import moose
from config import logger

# Voltage range for HH gate interpolation tables
VMIN = -0.12
VMAX = 0.04
VDIVS = 640

CMIN = 0
CMAX = 1e3
CDIVS = 3000

E_AR = -35e03
E_Ca = 125e-3
E_K = -95e-3
E_K_FS = -100e-3
E_Na = 50e-3


channel_spec = {
    'AR': {
        'Ek': E_AR,
        'Xpower': 1,
        'X': 0.25,
        'gateX': {
            'infExpr': '1 / ( 1 + exp( ( v * 1e3 + 75 ) / 5.5 ) )',
            'tauExpr': (
                '1e-3 / ( exp( -14.6 - 0.086 * v * 1e3) +'
                ' exp( -1.87 + 0.07 * v * 1e3))'
            ),
        },
    },
    # =====================================================================
    # Ca channels
    # =====================================================================
    'CaL': {
        'Ek': E_Ca,
        'Xpower': 2,
        'X': 0.0,
        # Mstring (MOOSE-string) is used by the prototype reader to
        # create custom connections
        'Mstring': ('addmsg1', '.	IkOut	../CaPool	current'),
        'gateX': {
            'alphaExpr': '1.6e3 / (1.0 + exp(-0.072 * (v * 1e3 - 5)))',
            'betaExpr': (
                'if (abs(v + 8.9e-3) < 1e-9,'
                '1e3 * 0.1 * exp(-(v + 8.9e-3) / 5e-3),'
                '1e3 * 0.02 * v * 1e3 / (exp(v / 5e-3) - 1))'
            ),
        },
    },
    'CaT': {
        'Ek': E_Ca,
        'Xpower': 2,
        'Ypower': 1,
        'X': 0.0,
        'gateX': {
            'infExpr': '1/(1 + exp(-v - 56e-3)/6.2e-3)',
            'tauExpr': (
                '1e-3 * (0.204 + 0.333 /'
                ' (exp((v + 15.8e-3) / 18.2e-3 ) +'
                ' exp((- v - 131e-3) / 16.7e-3)))'
            ),
        },
        'gateY': {
            'infExpr': '1 / (1 + exp((v + 80e-3 ) / 4e-3))',
            'tauExpr': (
                'if(v < -81e-3,'
                ' 1e-3 * 0.333 * exp( ( v + 466e-3 ) / 66.6e-3 ),'
                ' 1e-3 * (9.32 + 0.333 * exp( ( -v - 21e-3 ) / 10.5e-3 )))'
            ),
        },
    },
    'CaT_A': {
        'Ek': E_Ca,
        'Xpower': 2,
        'Ypower': 1,
        'X': 0.0,
        'Y': 0.0,
        'gateX': {
            'infExpr': '1.0 / ( 1 + exp( ( - v * 1e3 - 52 ) / 7.4 ) )',
            'tauExpr': (
                '1e-3 * (1 + .33 /'
                ' ( exp( ( v * 1e3 + 27.0 ) / 10.0 ) +'
                ' exp( ( - v * 1e3 - 102 ) / 15.0 )))'
            ),
        },
        'gateY': {
            'infExpr': '1 / ( 1 + exp( ( v * 1e3 + 80 ) / 5 ) )',
            'tauExpr': (
                '1e-3 * (28.30 + 0.33 /'
                ' (exp(( v * 1e3 + 48.0)/ 4.0) +'
                ' exp( ( -v * 1e3 - 407.0) / 50.0 ) ))'
            ),
        },
    },
    # =====================================================================
    # K channels
    # =====================================================================
    'KDR': {
        'Ek': E_K,
        'Xpower': 4,
        'X': 0.0,
        'gateX': {
            'infExpr': '1.0 / (1.0 + exp((- v - 29.5e-3) / 10e-3))',
            'tauExpr': (
                'if (v < -10e-3,'
                ' 1e-3 * (0.25 + 4.35 * exp((v + 10.0e-3) / 10.0e-3)),'
                ' 1e-3 * (0.25 + 4.35 * exp((- v - 10.0e-3) / 10.0e-3)))'
            ),
        },
    },
    'KDR_FS': {
        'Ek': E_K_FS,
        'Xpower': 4,
        'X': 0.0,
        'gateX': {
            'infExpr': '1.0 / (1.0 + exp((- v - 27e-3) / 11.5e-3))',
            'tauExpr': (
                'if(v < -10e-3,'
                ' 1e-3 * (0.25 + 4.35 * exp((v + 10.0e-3) / 10.0e-3)),'
                ' 1e-3 * (0.25 + 4.35 * exp((- v - 10.0e-3) / 10.0e-3)))'
            ),
        },
    },
    'KA': {
        'Ek': E_K,
        'Xpower': 4,
        'X': 0.0,
        'Ypower': 1.0,
        'gateX': {
            'infExpr': '1 / (1 + exp((- v - 60e-3) / 8.5e-3))',
            'tauExpr': (
                '1e-3 * (0.185 + 0.5 / '
                '(exp( ( v + 35.8e-3 ) / 19.7e-3) +'
                ' exp((-v - 79.7e-3 ) / 12.7e-3 )))'
            ),
        },
        'gateY': {
            'infExpr': '1 / (1 + exp(( v + 78e-3 ) / 6e-3 ))',
            'tauExpr': (
                'if(v <= -63e-3,'
                ' 1e-3 * 0.5 / ( exp( ( v + 46e-3 ) / 5e-3) +'
                '     exp(( - v - 238e-3 ) / 37.5e-3 )),'
                ' 9.5e-3)'
            ),
        },
    },
    'KA_IB': {
        'Ek': E_K,
        'Xpower': 4,
        'X': 0.0,
        'Ypower': 1.0,
        'gateX': {
            'infExpr': '1 / (1 + exp((- v - 60e-3) / 8.5e-3))',
            'tauExpr': (
                '1e-3 * (0.185 + 0.5 / '
                '  (exp( ( v + 35.8e-3 ) / 19.7e-3) +'
                '   exp((-v - 79.7e-3 ) / 12.7e-3 )))'
            ),
        },
        'gateY': {
            'infExpr': '1 / (1 + exp((v + 78e-3) / 6e-3))',
            'tauExpr': (
                '2.6 *'
                ' if(v <= -63e-3,'
                ' 1e-3 * 0.5 / ( exp( ( v + 46e-3 ) / 5e-3) +'
                '     exp(( - v - 238e-3 ) / 37.5e-3 )),'
                ' 9.5e-3)'
            ),
        },
    },
    'K2': {
        'Ek': E_K,
        'Xpower': 1,
        'X': 0.0,
        'Ypower': 1,
        'Y': 0.0,
        'gateX': {
            'infExpr': '1 / (1 + exp((-v * 1e3 - 10) / 17))',
            'tauExpr': (
                '1e-3 * (4.95 + 0.5 /'
                ' (exp((v * 1e3 - 81) / 25.6) +'
                '  exp((-v * 1e3 - 132) / 18)))'
            ),
        },
        'gateY': {
            'infExpr': '1 / (1 + exp((v * 1e3 + 58) / 10.6))',
            'tauExpr': (
                '1e-3 * (60 + 0.5 /'
                '   (exp((v * 1e3 - 1.33) / 200) +'
                '    exp((-v * 1e3 - 130) / 7.1)))'
            ),
        },
    },
    'KM': {
        'Ek': E_K,
        'Xpower': 1,
        'X': 0.0,
        'gateX': {
            'alphaExpr': '1e3 * 0.02 / ( 1 + exp((-v - 20e-3 ) / 5e-3))',
            'betaExpr': '1e3 * 0.01 * exp((-v - 43e-3) / 18e-3)',
        },
    },
    'KAHP': {
        'Ek': E_K,
        'Zpower': 1,
        'Z': 0.0,
        'mstring': ('addmsg1', '../CaPool concOut . concen'),
        'gateZ': {
            'alphaExpr': 'if (c < 100.0, 0.1 * c, 10.0)',
            'betaExpr': '10.0',
        },
    },
    'KAHP_SLOWER': {
        'Ek': E_K,
        'Zpower': 1,
        'Z': 0.0,
        'mstring': ('addmsg1', '../CaPool concOut . concen'),
        'gateZ': {
            'alphaExpr': 'if (c < 500.0, 1e3 * c / 50000, 10.0)',
            'betaExpr': '1.0',
        },
    },
    'KAHP_DP': {
        'Ek': E_K,
        'Zpower': 1,
        'Z': 0.0,
        'mstring': ('addmsg1', '../CaPool concOut . concen'),
        'gateZ': {
            'alphaExpr': 'if (c < 100.0, 0.1 * c, 10.0)',
            'betaExpr': '1.0',
        },
    },
    'KC': {
        'Ek': E_K,
        'Xpower': 1,
        'X': 0.0,
        'Zpower': 1,
        'Z': 0.0,
        'gateZ': {
            'alphaExpr': 'if (c < 250.0, c / 250.0, 1.0)',
            'betaExpr': 'if (c < 250.0, 1 - c / 250.0, 0.0)',
        },
        'gateX': {
            'alphaExpr': (
                'if (v < -10e-3,'
                '2e3 / 37.95 * (exp('
                '    (v * 1e3 + 50 ) / 11 -'
                '    (v * 1e3 + 53.5 ) / 27 )),'
                ' 2e3 * exp(( - v * 1e3 - 53.5) / 27))'
            ),
            'betaExpr': (
                'if (v < -10e-3,'
                '    2e3 * exp(( - v * 1e3 - 53.5) / 27) -'
                # beta  = 2 * exp( ( - v - 53.5 ) / 27 ) - alpha
                '        2e3 / 37.95 * (exp('
                '            (v * 1e3 + 50 ) / 11 - '
                '            ( v * 1e3 + 53.5 ) / 27 )),'
                ' 0.0)'
            ),
        },
        'instant': 4,
    },
    'KC_FAST': {
        'Ek': E_K,
        'Xpower': 1,
        'X': 0.0,
        'Zpower': 1,
        'Z': 0.0,
        'gateZ': {
            'alphaExpr': 'if (c < 250.0, c / 250.0, 1.0)',
            'betaExpr': 'if (c < 250.0, 1 - c / 250.0, 0.0)',
        },
        'gateX': {
            'alphaExpr': (
                '2 * (if(v < -10e-3,'
                '    2e3 / 37.95 * ( exp('
                '        (v * 1e3 + 50 ) / 11 -'
                '        ( v * 1e3 + 53.5 ) / 27)),'
                '    2e3 * exp(( - v * 1e3 - 53.5) / 27)))'
            ),
            'betaExpr': (
                '2 * (if (v < -10e-3,'
                '   2e3 * exp(( - v * 1e3 - 53.5) / 27) - '
                # beta  = 2 * exp( ( - v - 53.5 ) / 27 ) - alpha
                '       2e3 / 37.95 * ( exp('
                '         (v * 1e3 + 50) / 11 -'
                '         (v * 1e3 + 53.5) / 27)),'
                ' 0.0))'
            ),
        },
        'instant': 4,
    },
    # =====================================================================
    # Na channels
    # =====================================================================
    'NaF': {
        'Ek': E_Na,
        'Xpower': 3,
        'Ypower': 1,
        'X': 0.0,
        # 'Y':   0.54876953,
        'gateX': {
            'tauExpr': (
                'if ((v + (-3.5e-3)) < -30e-3,'
                '1.0e-3 * (0.025 + 0.14 * '
                '    exp((v + (-3.5e-3) + 30.0e-3) / 10.0e-3)),'
                '1.0e-3 * (0.02 + 0.145 * '
                '    exp(( - v - (-3.5e-3) - 30.0e-3) / 10.0e-3)))'
            ),
            'infExpr': (
                '1.0 / (1.0 + ' '    exp(( - v - (-3.5e-3) - 38e-3) / 10e-3))'
            ),
        },
        'gateY': {
            'tauExpr': (
                '1.0e-3 * (0.15 + 1.15 / '
                '    ( 1.0 + exp(( v + 37.0e-3) / 15.0e-3)))'
            ),
            'infExpr': '1.0 / (1.0 + exp((v + 62.9e-3) / 10.7e-3))',
        },
    },
    'NaF_TCR': {
        'Ek': E_Na,
        'Xpower': 3,
        'Ypower': 1,
        'X': 0.0,
        'gateY': {
            'tauExpr': (
                '1.0e-3 * (0.15 + 1.15 / '
                '( 1.0 + exp(( v + 37.0e-3) / 15.0e-3)))'
            ),
            'infExpr': '1.0 / (1.0 + exp((v + (-7e-3) + 62.9e-3) / 10.7e-3))',
        },
        'gateX': {
            'tauExpr': (
                'if ((v + (-5.5e-3)) < -30e-3,'
                '1.0e-3 * (0.025 + 0.14 * '
                '    exp((v + (-5.5e-3) + 30.0e-3) / 10.0e-3)),'
                '1.0e-3 * (0.02 + 0.145 * '
                '    exp(( - v - (-5.5e-3) - 30.0e-3) / 10.0e-3)))'
            ),
            'infExpr': '1.0 / (1.0 + exp(( - v - (-5.5e-3) - 38e-3) / 10e-3))',
        },
    },
    'NaF2': {
        'Ek': E_Na,
        'Xpower': 3,
        'Ypower': 1,
        'X': 0.0,
        # -2.5 mV is fastNa_shift used for tweaking the channel for
        # some neuron classes
        'gateX': {
            'tauExpr': (
                'if ((v + (-2.5e-3)) < -30e-3,'
                ' 1.0e-3 * (0.0125 + 0.1525 * '
                '    exp ((v + (-2.5e-3) + 30e-3) / 10e-3)),'
                ' 1.0e-3 * (0.02 + 0.145 * '
                '    exp((- v - (-2.5e-3) - 30e-3) / 10e-3)))'
            ),
            'infExpr': (
                '1.0 / (1.0 + exp(( - v - (-2.5e-3) - 38e-3) / 10e-3))'
            ),
        },
        'gateY': {
            'tauExpr': (
                '1e-3 * (0.225 + 1.125 /'
                '   ( 1 + exp( (  v + 37e-3 ) / 15e-3 ) ))'
            ),
            'infExpr': '1.0 / (1.0 + exp((v + 58.3e-3) / 6.7e-3))',
        },
    },
    'NaF2_nRT': {
        'Ek': E_Na,
        'Xpower': 3,
        'Ypower': 1,
        'X': 0.0,
        'gateX': {
            'tauExpr': (
                'if (v < -30e-3,'
                ' 1.0e-3 * (0.0125 + 0.1525 *'
                '     exp ((v + 30e-3) / 10e-3)),'
                ' 1.0e-3 * (0.02 + 0.145 * '
                '     exp((-v - 30e-3) / 10e-3)))'
            ),
            'infExpr': '1.0 / (1.0 + exp(( - v - 38e-3) / 10e-3))',
        },
        'gateY': {
            'tauExpr': '1e-3 * (0.225 + 1.125 /'
            ' (1 + exp( (  v + 37e-3 ) / 15e-3)))',
            'infExpr': '1.0 / (1.0 + exp((v + 58.3e-3) / 6.7e-3))',
        },
    },
    'NaP': {
        'Ek': E_Na,
        'Xpower': 1,
        'gateX': {
            'tauExpr': (
                'if (v < -40e-3,'
                '  1.0e-3 * (0.025 + 0.14 * '
                '    exp((v + 40e-3) / 10e-3)),'
                '  1.0e-3 * (0.02 + 0.145 * '
                '    exp((-v - 40e-3) / 10e-3)))'
            ),
            'infExpr': '1.0 / (1.0 + exp((-v - 48e-3) / 10e-3))',
        },
    },
    'NaPF': {
        'Ek': E_Na,
        'Xpower': 3,
        'gateX': {
            'tauExpr': (
                'if (v < -30e-3,'
                ' 1.0e-3 * (0.025 + 0.14 * '
                '     exp((v  + 30.0e-3) / 10.0e-3)),'
                ' 1.0e-3 * (0.02 + 0.145 * '
                '     exp((- v - 30.0e-3) / 10.0e-3)))'
            ),
            'infExpr': '1.0 / (1.0 + exp((-v - 38e-3) / 10e-3))',
        },
    },
    'NaPF_SS': {
        'Ek': E_Na,
        'Xpower': 3,
        'X': 0.0,
        # For spiny stellate, a fastNa_shift of -2.5 mV is added
        'gateX': {
            'tauExpr': (
                'if ((v + (-2.5e-3)) < -30e-3,'
                ' 1.0e-3 * (0.025 + 0.14 *'
                '    exp(((v + (-2.5e-3))  + 30.0e-3) / 10.0e-3)),'
                ' 1.0e-3 * (0.02 + 0.145 * '
                '    exp((- (v + (-2.5e-3)) - 30.0e-3) / 10.0e-3)))'
            ),
            'infExpr': (
                '1.0 / (1.0 + exp(' '   (- (v + (-2.5e-3)) - 38e-3) / 10e-3))'
            ),
        },
    },
    'NaPF_TCR': {
        'Ek': E_Na,
        'Xpower': 1,
        'X': 0.0,
        'gateX': {
            'tauExpr': (
                'if ((v + (7e-3)) < -30e-3,'
                '  1.0e-3 * (0.025 + 0.14 * exp('
                '      ((v + (7e-3))  + 30.0e-3) / 10.0e-3)),'
                '  1.0e-3 * (0.02 + 0.145 * exp('
                '      (- (v + (7e-3)) - 30.0e-3) / 10.0e-3)))'
            ),
            'infExpr': (
                '1.0 / (1.0 + exp(' '    (-(v + (7e-3)) - 38e-3) / 10e-3))'
            ),
        },
    },
}


def get_proto(name, parent='/library'):
    """Returns prototype element `name` under `parent`, `None` if the
    element does not exist."""
    if not moose.exists(parent):
        _ = moose.Neutral(parent)
    path = f'{parent}/{name}'
    if moose.exists(path):
        return moose.element(path)

    return None


def get_channel(name, spec, parent='/library'):
    """Returns a prototype HH channel with name `name` under `parent`,
    creating it if it does not exits

    Parameters
    ----------
    name: str
        Name of the channel. There must be a corresponding entry in the `spec`.
    spec: dict (`channel_spec`)
        Dictionary of channel specification.
    parent: str
        Path of the parent element of the channel object
    """
    if not spec:
        raise ValueError(f'Unknown channel: {name}')
    chan = get_proto(name, parent)
    if chan:
        return chan

    path = f'{parent}/{name}'
    chan = moose.HHChannel(path)
    # Populate the gate attributes if corresponding power is positive
    for key, attr in {'X': 'Xpower', 'Y': 'Ypower', 'Z': 'Zpower'}.items():
        power = spec.get(attr, 0)
        if power <= 0:
            continue
        setattr(chan, attr, power)
        setattr(chan, key, spec.get(key, 0.0))
        logger.debug(f'Set {path}.{key} = {getattr(chan, key)}')
        gate_name = f'gate{key}'
        gate = moose.element(f'{path}/{gate_name}')
        gate_spec = spec.get(gate_name)
        for gate_attr, val in gate_spec.items():
            logger.debug(f'Setting {gate.path}.{gate_attr} = {val}')
            setattr(gate, gate_attr, val)
            logger.debug(
                f'OK Set {gate.path}.{gate_attr} = {getattr(gate, gate_attr)}'
            )
        if key == 'Z':
            gate.min = CMIN
            gate.max = CMAX
            gate.divs = CDIVS
        else:
            gate.min = VMIN
            gate.max = VMAX
            gate.divs = VDIVS
        gate.fillFromExpr()
    chan.Ek = spec.get('Ek')
    mstring = spec.get('Mstring')
    if mstring:
        ms = moose.Mstring(f'{chan.path}/{mstring[0]}')
        ms.value = mstring[1]
    return chan


def get_capool(parent='/library'):
    name = 'CaPool'
    capool = get_proto(name, parent)
    if capool is None:
        path = f'{parent}/{name}'
        capool = moose.CaConc(path)
        capool.CaBasal = 0.0
        capool.ceiling = 1e6
        capool.floor = 0.0
    return capool


def init_channels(libpath='/library'):
    channels = {}
    logger.debug('Start initializing channels')
    ts = time.perf_counter()
    for name, spec in channel_spec.items():
        if moose.exists(f'{libpath}/{name}'):
            channels[name] = moose.element(f'{libpath}/{name}')
            continue

        try:
            logger.debug(f'   ... Creating prototype for {name}')
            channels[name] = get_channel(name, spec, parent=libpath)
            logger.debug(f'OK ... Created prototype for {name}')
        except Exception:
            logger.error(f'EE .. Could not create prototype for {name}')
            raise

    channels['CaPool'] = get_capool(parent=libpath)
    channels['spike'] = moose.SpikeGen(f'{libpath}/spike')
    te = time.perf_counter()
    logger.debug(f'Finished initializing channels in {te - ts} seconds.')
    return channels


#
# channels.py ends here
