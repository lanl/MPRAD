"""An example of input data for proton radiography simulation """

from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_sim

input_list = []

geometryCommon = {
  'protonAngle'  : 15.0,
  'plateRange'   : np.array([[-2.0,2.0],[-2.0,2.0]]),
  'binX'         : 200,
  'binY'         : 200,
  'dtLimitLow'   : 0.00001,
  'dtLimitHigh'  : 0.0001,
  'dvLimit'      : 0.001,
  'protonNumber' : int(1E5),
  'countAllProcs': 0,
  'progress_display': 1,
  'msc_pmax'     : 0.05,
  'track_n' : 384,
  'track_leap_n' : 16,
  'track_max_step': 4096,
  'hit_3d_n' : 4096,
  'protonSourceLinear' : 1,
  'protonSourceLinearDirection' : np.array([0.0,0.0,1.0]), 
}


protonEnergy1 = {
'protonSize'             : 00.0e-04,
'protonEnergyInit'       : 15.3, 
'protonEnergyWidth'      : 0.67,
'protonSourceType'       : 0,
##'protonEnergyEnd'        : np.array([[1.7, 13.5], [14.35, 16.0]]),
'protonEnergyEnd'        : np.array([[14.35, 16.0]]),
}

protonEnergy2 = {
'protonSize'             : 2.0e-04,
'protonEnergyInit'       : 40.0, 
'protonEnergyWidth'      : 5.60,
'protonSourceType'       : 1,
'protonEnergyEnd'        : np.array([[40.0, 100.0], ]),
}

sourceAndTarget1 = {
  'protonCenter' : -np.array([-1.0,-0.0,0.0]),
  'plateCenter'  : -np.array([27.0,-0.0,0.0]),
  'plateX' : np.array([0.0, 0.0, 1.0]),
  'plateY' : -np.array([0.0, -1.0, 0.0]),
}

species1 = {
'push' : 'CH',
'foam' : 'CH',
'cham' : 'He',
'wall' : 'Be',
'sphr' : 'Be',
}

species2 = {
'push' : 'CH',
'foam' : 'CH',
'cham' : 'He',
'wall' : 'Be',
'sphr' : 'CH',
'gcap' : 'Al',
}

species3 = {
'push' : 'CH',
'foam' : 'CH',
'cham' : 'He',
'wall' : 'Be',
'sphr' : 'Be',
'gcap' : 'Cu',
}


args_input = {
'species': species3,
}


input_list = []

name = 'run2'
root_path = "/net/scratch3/yclu/HEDB/" +  name + "/omega2017_hdf5_plt_cnt_"

for i in [150, ]:
    input_list.append(
            ["{0}{1:04d}".format(root_path, i), "_shear_{0}_bp1_{1:04d}.npz".format(name, i),
            [args_input, ],
                [geometryCommon, protonEnergy1, sourceAndTarget1, 
                ],]
    )
    input_list.append(
            ["{0}{1:04d}".format(root_path, i), "_shear_{0}_bp0_{1:04d}.npz".format(name, i),
            [args_input, ],
                [geometryCommon, protonEnergy1, sourceAndTarget1, 
                    {'multimag': np.array([0.0,0.0,0.0])}, 
                ],]
    )
proton_sim(input_list)
