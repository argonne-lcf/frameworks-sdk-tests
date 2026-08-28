"""
The parse function first loads the default input file then loads the given
input file, overwriting the default value of any of the parameters given.

Input files must have all variables in the [params] section.
"""
import configparser
import re


def isinteger(val):
    """
    Returns True if the string given as input represents an integer.
    Does not check for leading or training spaces.
    """
    return val.isdigit() or (val[0] == '-' and val[1:].isdigit())


def interp(val):
    """
    Interpret the value of the given string as an int, a float, or a boolean if
    possible. When none of these interpretations are possible it will return
    the input string.
    """
    if len(val) == 0:
        return val

    if isinteger(val):
        return int(val)

    if isinteger(val.replace('.', '', 1)):
        return float(val)

    if val == 'True' or val == 'False':
        return val == 'True'

    return val


name_pattern = r'(\d*)N\+(\d*)Z'
def name_parse(name):
    """
    Given name of nuclei return the corresponding information which it implies.
    Currently the relevant information is:
    number of particles, number of protons, number of spin up nucleons, and ground state parity

    Example:
    if name is '2H' then this will return (2, 1, 2, 1)
    """
    # The values of the nuclei_dictionary are (npart, nprot, nup, ground state parity)
    nuclei_dictionary = {'2H': (2, 1, 2, 1),
                         '3He': (3, 2, 2, 1),
                         '3H': (3, 1, 2, 1),
                         '4He': (4, 2, 2, 1),
                         '6Li': (6, 3, 4, 1),
                         '6He': (6, 2, 3, 1),
                         '7Li': (7, 3, 4, -1),
                         '7Be': (7, 4, 3, -1),
                         '8Be': (8, 4, 4, 1),
                         '8He': (8, 2, 4, 1),
                         '9Be': (9, 4, 5, -1),
                         '9Li': (9, 3, 5, -1),
                         '10Be': (10, 4, 5, 1),
                         '11C': (11, 6, 5, -1),
                         '12C': (12, 6, 6, 1),
                         '13C': (13, 6, 7, -1),
                         '14C': (14, 6, 7, 1),
                         '14N': (14, 7, 8, 1),
                         '15N': (15, 7, 8, -1),
                         '15O': (15, 8, 8, -1),
                         '16O': (16, 8, 8, 1),
                         '17O': (17, 8, 9, 1),
                         '18O': (18, 8, 9, 1),
                         '20O': (20, 8, 10, 1),
                         '20Ne': (20, 10, 10, 1),
                         '21Ne': (21, 10, 11, 1),
                         '17F': (17, 9, 9, 1),
                         '24Mg': (24, 12, 12, 1),
                         '28Si': (28, 14, 14, 1),
                         '34Si': (34, 14, 17, 1),
                         '40Ca': (40, 20, 20, 1),
                         '40Ar': (40, 18, 20, 1),
                         '2N': (2, 0, 1, 1)}

    if name in nuclei_dictionary:
        return nuclei_dictionary[name]

    P = re.findall(name_pattern, name)[0]
    if len(P) == 2:
        N, Z = [int(x) for x in P]
        npart = N+Z
        nprot = Z
        nup = (N+Z)//2
        return (npart, nprot, nup, 1)

    raise NameError('Unknown nuc_name value.')


class Config(object):
    def __init__(self, config_reader):
        self.config_reader = config_reader
        self.set_name_info()

        # Derived values
        if self.rho == 0:
            self.periodic = False
            self.remove_cm = True  # remove center of mass
            self.L = None
        else:
            self.periodic = True
            self.remove_cm = False
            self.L = (self.npart/self.rho)**(1.0/3)  # fm

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            config_reader = object.__getattribute__(self, 'config_reader')
            return interp(config_reader['params'][name])

    def set_name_info(self):
        self.npart, self.nprot, self.nup, parity_GS = name_parse(self.nuc_name)
        self.ndown = self.npart - self.nup

        # Set parity to ground state value if none is specified.
        # Print error if specified value is not 1 or -1
        try:
            assert self.parity == 1 or self.parity == -1
        except KeyError:
            self.parity = parity_GS
        except AssertionError:
            raise AssertionError('Parity value not 1 or -1. Stopping execution.')


def parse(filename):
    config_reader = configparser.ConfigParser(inline_comment_prefixes="#")
    if filename != 'DEFAULT.ini':
        config_reader.read('DEFAULT.ini')
    config_reader.read(filename)
    return Config(config_reader)
