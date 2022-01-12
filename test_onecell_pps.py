"""An example of input data for proton radiography simulation """

from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_sim
from matplotlib import pyplot as plt
import os

pps = dict(
  pps_center = np.array([0.0,0.0,0.0]),
  pps_x = np.array([0.0, -1.0, 0.0]),
  pps_y = np.array([0.0, 0.0, 1.0]),
  pps_list_circ = np.array([[-0.02, 0.02, 0.01], [0.0, -0.01, 0.01]]),
  pps_list1_poly = np.array([[-0.05, -0.03], [-0.05, 0.05], [0.03, 0.06], [0.06,-0.03]]), 
  pps_list2_poly = np.array([[-0.06, -0.01], [-0.06, 0.01], [0.06, 0.01], [0.06,-0.01]]), 
  pps_dE1 = 1e10,
)
#pps = {}
pps_radius = 100.0
pps = dict(
  pps_center = np.array([-0.13, 0.0,0.0]),
  pps_x = np.array([0.0, 0.0, 1.0]),
  pps_y = np.array([0.0, 1.0, 0.0]),
  pps_list_circ = np.array([np.array([0.0, 0.0, pps_radius])*1.0e-4, 
                            np.array([0.0,  450.0, pps_radius])*1.0e-4, 
                            np.array([0.0, -450.0, pps_radius])*1.0e-4, 
                            np.array([ 450.0, 0.0, pps_radius])*1.0e-4, 
                            np.array([-450.0, 0.0, pps_radius])*1.0e-4, 
                            np.array([ 450.0,  450.0, 37.5])*1.0e-4, 
                            np.array([-450.0,  450.0, 25.0])*1.0e-4, 
                            np.array([-450.0, -450.0, 10.0])*1.0e-4, 
                            np.array([ 450.0, -450.0, 50.0])*1.0e-4, 
                                ]),
  pps_list1_poly = np.array([[-1450.0, -250.0], [-1450.0, 250.0], [1450.0, 250.0], [1450.0, -250.0]])*1.0e-4, 
  pps_list2_poly = np.array([[-500.0, -750.0], [-500.0, 750.0], [500.0, 750.0], [500.0, -750.0]])*1.0e-4, 
  pps_dE1 = 0.9958,
  pps_dE2 = 0.0669,
  pps_ang_strag = 34.901e-3,
  protonNumber = int(4E7),
  progress_display = 1
)

geometryCommon = {
  'protonAngle'  : 16.0,
  'plateRange'   : np.array([[-5.0,5.0],[-5.0,5.0]]),
  'binX'         : 500,
  'binY'         : 500,
  'dtLimitLow'   : 0.00001,
  'dtLimitHigh'  : 0.0001,
  'dvLimit'      : 0.001,
  'countAllProcs': 0,
}

theta = 0.0
phi = 0.0
h1 = np.array([0.0, 0.0, 0.0])
L1 = 1.0
D1 = 20.0
rhat = np.array([cos(phi), sin(phi), 0.0])
xhat = np.array([sin(phi),-cos(phi), 0.0])
yhat = np.array([sin(theta)*cos(phi), \
                 sin(theta)*sin(phi), \
                 cos(theta)])
size1 = 00.0e-04
size1 = 20.0e-04

erange = np.array([[0.05+0.1*i, 0.15+0.1*i] for i in range(160)])
ener_t = np.array(
[2. , 11.0 , 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.7,
           12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8,
           13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 16.0])
erange = np.array([[ener_t[i], ener_t[i+1]] for i in range(len(ener_t)-1)])
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
protonEnergy1 = {
  'protonSize'             : 20.0e-04,
  'protonEnergyInit'       : 14.80, 
  'protonEnergyWidth'      : 0.250,
  'protonSourceType'       : 2,
  'protonEnergyEnd'        : erange,
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
            [geometryCommon, protonEnergy1, sourceAndTarget1, plateOrientation1, pps]
    ],
    ]
    proton_sim(input_list)

    data = np.load("{0}.npz".format(fn))['data'][:].copy()
    data_all[fn] = data

np.savez_compressed("results_new.npz", **data_all)


