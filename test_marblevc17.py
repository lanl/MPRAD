"""An example of input data for proton radiography simulation """

from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_sim


input_list = []

geometryCommon = {
  'protonAngle'  : 14.2,
  'plateRange'   : np.array([[-5.0,5.0],[-5.0,5.0]]),
  'binX'         : 500,
  'binY'         : 500,
  'dtLimitLow'   : 0.00001,
  'dtLimitHigh'  : 0.0002,
  'dvLimit'      : 0.001,
  'protonNumber' : int(1E7),
  'countAllProcs': 0,
  'progress_display': 0,
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

erange = [[1.8, 13.0],]
for i in range(17):
    erange.append([13.0+0.1*i, 13.1+0.1*i])
erange.append([14.7, 16.0])
erange = np.array(erange)

protonEnergy1 = {
'protonSize'             : 20.0e-04,
'protonEnergyInit'       : 15.3, 
'protonEnergyWidth'      : 0.67,
'protonSourceType'       : 0,
'protonEnergyEnd'        : erange,
}

sourceAndTarget1 = {
  'protonCenter' : h1 - L1*rhat,
  'plateCenter'  : h1 + D1*rhat,
}

plateOrientation1 = {
  'plateX' : xhat,
  'plateY' : yhat,
}

species= {
'cham' : 'He',
'sphr' : 'CH', 
'push' : 'CH',
'wall' : 'Be',
'foam' : 'CH', 
}

args_input = {
'species': species,
'usePlasma' : True,
}

root_dir = "/net/scratch4/yclu/Marbel17/Blue_H_ed/omega2017_hdf5_chk"

input_list = []
for i in range(0, 171, 5):
    fn_in = "{0}_{1:04d}".format(root_dir, i)
    fn_out = "_img_MVC_Blue_b0_{0:04d}".format(i)
    input_list.append([
        fn_in, fn_out, 
        [args_input, ],
        [geometryCommon, protonEnergy1, sourceAndTarget1, 
            {'multimag': np.array([0.0,0.0,0.0])}, plateOrientation1],
    ])
    fn_out = "_img_MVC_Blue_b1_{0:04d}".format(i)
    input_list.append([
        fn_in, fn_out, 
        [args_input, ],
        [geometryCommon, protonEnergy1, sourceAndTarget1, 
            {'multimag': np.array([1.0,1.0,1.0])}, plateOrientation1],
    ])

proton_sim(input_list)
