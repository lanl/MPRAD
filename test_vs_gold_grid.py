"""An example of input data for proton radiography simulation """

from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_sim


input_list = []

geometryCommon = {
  'protonAngle'  : 13.0,
  'plateRange'   : np.array([[-5.0,5.0],[-5.0,5.0]]),
  'binX'         : 250,
  'binY'         : 250,
  'dtLimitLow'   : 0.00001,
  'dtLimitHigh'  : 0.0001,
  'dvLimit'      : 0.001,
  'protonNumber' : 1000000,
  'countAllProcs': 0,
  'msc_pmax'     : 0.01,
  'plateMatrixInput' : np.load("shape/cone/cone.npz")['data'],
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
size1 = 26.5e-04

protonEnergy1 = {
  'protonEnergyInit'       : 15.3, 
  'protonEnergyFWHM'       : 0.67,
  'protonEnergyEnd'        : np.array([[14.35, 17.0],[1.7, 13.6]])
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

args_input = {
'specialGrid': 'vc_gold_grid',
}


from glob import glob
import os
input_list=[]
name_list=glob("/home/yingchao/Desktop/VC/Targets/TIM6_Green/*.tif")
for useMSC in (True, False):
    for in_fn in name_list:
        out_fn = os.path.splitext(in_fn)[0] + \
                 ('-msc' if useMSC else '-nomsc') + \
                 '.npz'
        input_list.append([
        in_fn, out_fn, 
        [args_input, {"useMSC" : useMSC}],
        [geometryCommon, protonEnergy1, sourceAndTarget1, plateOrientation1],
        ])

input_list

proton_sim(input_list)
