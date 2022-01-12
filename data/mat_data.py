"""Material properties data
    abar   : avarage atomic mass number
    zvals  : atomic charge numbers
    fracs  : atomic number fractions
"""

import numpy as np
from yt import physical_constants as p
from yt import units as u
from math import pi

mat_data = {
    "C": {
        "avals"  : [12.011,],
        "zvals"  : [6.0,],
        "fracs"  : [1.0,],
        "a_cold" : 9.441289368348627,
    },
    "CH": {
        "avals"  : [1.008, 12.011],
        "zvals"  : [1.0, 6.0],
        "fracs"  : [0.5, 0.5],
        "a_cold" : 9.672776314704233,
    },
    "He": {
        "avals"  : [4.003,], 
        "zvals"  : [2.0,], 
        "fracs"  : [1.0,], 
        "a_cold" : 10.11960957043243,
    },
    "Be": {
        "avals"  : [9.012,],
        "zvals"  : [4.0,],
        "fracs"  : [1.0,],
        "a_cold" : 9.66535935281768,
    },
    "Au": {
        "avals"  : [196.97,],
        "zvals"  : [79.0,],
        "fracs"  : [1.0,],
        "a_cold" : 6.989546948126616,
    }, 
    "Cu": {
        "avals"  : [63.546,],
        "zvals"  : [29.0,],
        "fracs"  : [1.0,],
        "a_cold" : 7.886949762356227,
    }, 
    "Al": {
        "avals"  : [26.982,],
        "zvals"  : [13.0,],
        "fracs"  : [1.0,],
        "a_cold" : 8.661272119472274,
    }, 
    "Mg": {
        "avals"  : [24.305,],
        "zvals"  : [12.0,],
        "fracs"  : [1.0,],
        "a_cold" : 8.71134977735726,
    }, 
    "Ta": {
        "avals"  : [180.95,],
        "zvals"  : [73.0,],
        "fracs"  : [1.0,],
        "a_cold" : 7.053984855805477, 
    }, 
}

stop_coef1 = float((4*np.pi* p.elementary_charge**4 * u.g/u.cm**3 \
        / p.amu / p.mass_electron / p.speed_of_light**2 / u.MeV * u.cm).in_cgs())
stop_coef2 = float(np.log((1.123/np.sqrt(2.0*np.pi) * p.mass_electron**1.5 * p.amu**0.5 * p.speed_of_light**2 \
        / p.hbar / p.elementary_charge / np.sqrt(u.g/u.cm**3)).in_cgs()))
coef_shield = (p.mass_electron * p.elementary_charge * p.kb**0.5 / 0.885 / (4.0*np.pi)**0.5 / p.hbar**2).in_cgs()
coef_shield = coef_shield**2
wavelength0 = (p.hbar*p.speed_of_light/u.MeV).in_cgs()
a0 = (p.hbar**2/ p.mass_electron / p.elementary_charge**2).in_cgs()

coef_straggling = float((4.0 * p.pi * p.elementary_charge**4 * u.g \
        / u.cm**3 / u.amu / u.MeV**2 * u.cm).in_cgs())


for mat, data in mat_data.items():
    fracs = np.array(data['fracs'])
    zvals = np.array(data['zvals'])
    avals = np.array(data['avals'])
    z1bar = np.dot(fracs, zvals)
    z2bar = np.dot(fracs, zvals**2)
    logzbar = np.dot(fracs, np.log(zvals))
    abar = np.dot(fracs, avals)

    data['abar'] = abar
    data['zbar'] = z1bar
    data['z2bar'] = z2bar
    data['logzbar'] = logzbar
    data['stop_b'] = stop_coef1 * z1bar / abar
    data['a_plasma'] = stop_coef2 + 0.5 * np.dot(fracs, zvals * np.log(avals / zvals)) / z1bar
    
    data['msc_a22']  = float((4*p.pi* p.elementary_charge**4 / p.amu / abar * (z1bar+z2bar) / u.MeV**2).in_cgs())
    data['msc_a1']   = float((2.0*np.log((1.86296189*wavelength0/a0).in_cgs()) + 2.0/3.0*logzbar)/abar)
mat_data['coef_shield'] = coef_shield
mat_data['coef_straggling'] = coef_straggling
