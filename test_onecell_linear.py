"""An example of input data for proton radiography simulation """

from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_sim
from matplotlib import pyplot as plt
import os
from pprint import pprint

shape = (100,1,1)
E0 = 14.8
geometryCommon = dict(
  protonEnergyInit = E0,
  msc_w = np.full(shape, 1.0),
  #@magx = np.array([[[1.0e3]]]),
  #@magy = np.array([[[2.0e3]]]),
  #@magz = np.array([[[3.0e3]]]),
  axis = '+x',
)

pars = [
    [ 19.32, 'Au', 25.0],
    [ 16.69, 'Ta', 40.0],
    [ 16.69, 'Ta', 5.0],
    [ 2.0, 'Al', 100.0 ],
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
    'leftEdge': -np.array([length/2.0, length/2.0*2.0, length/2.0*3.0])*1e-4,
    'rightEdge': np.array([length/2.0, length/2.0*2.0, length/2.0*3.0])*1e-4,
    'usePlasma' : False,
    'shape': shape,
    }

    fn = "{0}_{1}gpcc_{2}um".format(mat_name, dens, length)

    args_input = {
    'onecell' : onecell,
    }
    
    input_list = [
            [fn, "{0}.npz".format(fn), 
            [args_input, ],
            [geometryCommon]
    ],
    ]
    proton_sim(input_list, linear=True)
    data = np.load("{0}.npz".format(fn))
    for key in ['thetax', 'thetay', 'msc_rms_chi', 'msc_B', 'msc_chi', 'msc_chi_min', 'energy', 'delta_energy']:
        print(key, data[key][0,0])
    print('dE1 = {0:6.6f}'.format(E0-data['energy'][0,0]))
    print('dE2 = {0:6.6f}'.format(data['delta_energy'][0,0]))
    print('msc_chi = {0:6.6f}'.format(data['msc_chi'][0,0]))
    print('msc_chi_min = {0:6.6f}'.format(data['msc_chi_min'][0,0]))
    print('{0:6.6f}, {1:6.6f}, {2:6.6f}, {3:6.6f}'.\
            format(E0-data['energy'][0,0], \
                      data['delta_energy'][0,0], \
                      data['msc_chi'][0,0], \
                      data['msc_chi_min'][0,0]))


