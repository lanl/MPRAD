"""An example of input data for proton radiography simulation """

from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_sim
from matplotlib import pyplot as plt
import os

geometryCommon = {
  'protonAngle'  : 16.0,
  'protonAngle'  : 0.0,
  'plateRange'   : np.array([[-2.0,2.0],[-2.0,2.0]]),
  'binX'         : 100,
  'binY'         : 100,
  'dtLimitLow'   : 0.00001,
  'dtLimitHigh'  : 0.0001,
  'dvLimit'      : 0.001,
  'protonNumber' : 500000,
  'countAllProcs': 0,
}

theta = 0.0
phi = 0.0
h1 = np.array([0.0, 0.0, 0.0])
L1 = 1.0
D1 = 27.0
rhat = np.array([cos(phi), sin(phi), 0.0])
xhat = np.array([sin(phi),-cos(phi), 0.0])
yhat = np.array([sin(theta)*cos(phi), \
                 sin(theta)*sin(phi), \
                 cos(theta)])
size1 = 00.0e-04

erange = np.array([[0.05+0.1*i, 0.15+0.1*i] for i in range(160)])
print(erange)

protonEnergy1 = {
  'protonEnergyInit'       : 15.3, 
  'protonEnergyWidth'      : 0.00,
  'protonSourceType'       : 0,
  'protonEnergyEnd'        : erange,
}

protonEnergy2 = {
  'protonEnergyInit'       : 40.0, 
  'protonEnergyWidth'      : 0.00,
  'protonSourceType'       : 0,
  'protonEnergyEnd'        : np.array([[10.0, 40.0],])
}

sourceAndTarget1 = {
  'protonCenter' : h1 - L1*rhat,
  'plateCenter'  : h1 + D1*rhat,
  'protonSize'   : size1
}

plateOrientation1 = {
  'plateX' : xhat,
  'plateY' : yhat,
}

pars = [
    [ 0.1, 'CH', 1.0e-20 ],
    [ 0.1, 'CH', 100.0 ],
    [ 0.1, 'CH', 300.0 ],
    [ 0.1, 'CH', 1000.0 ],
    [ 1.0, 'CH', 100.0 ],
    [ 1.0, 'CH', 300.0 ],
    [ 10.0, 'CH', 100.0 ],
    [ 1.85, 'Be', 30.0 ],
    [ 1.85, 'Be', 100.0 ],
    [ 1.85, 'Be', 300.0 ],
    [ 10.0, 'Be', 30.0 ],
    [ 2.7, 'Al', 10.0 ],
    [ 2.7, 'Al', 30.0 ],
    [ 2.7, 'Al', 100.0 ],
    [ 10.0, 'Al', 10.0 ],
]

data_all = {}

for dens, mat_name, length in pars[:1]:
    onecell = {
    'dens' : dens,
    'mat'  : mat_name,
    'leftEdge': -np.array([length/2.0, 1000.0, 1000.0])*1e-4,
    'rightEdge': np.array([length/2.0, 1000.0, 1000.0])*1e-4,
    'usePlasma' : False,
    }

    fn = "{0}_{1}gpcc_{2}um".format(mat_name, dens, length)

    args_input = {
    'onecell' : onecell,
    }
    
    input_list = [
            [fn, "{0}.npz".format(fn), 
            [args_input, ],
            [geometryCommon, protonEnergy1, sourceAndTarget1, plateOrientation1]
    ],
    ]
    proton_sim(input_list)

    data = np.load("{0}.npz".format(fn))['data'][:].copy()
    data_all[fn] = data

np.savez_compressed("results_new.npz", **data_all)


