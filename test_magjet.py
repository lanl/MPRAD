"""An example of input data for proton radiography simulation """

from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_sim


input_list = []

geometryCommon = {
  'protonAngle'  : 23,
  'plateRange'   : np.array([[-5.0,5.0],[-5.0,5.0]]),
  'binX'         : 250,
  'binY'         : 250,
  'dtLimitLow'   : 0.00001,
  'dtLimitHigh'  : 0.0001,
  'dvLimit'      : 0.001,
  'protonNumber' : int(5E6),
  ##'protonNumber' : int(5E1),
  'countAllProcs': 0,
  'msc_pmax'     : 0.5,
  'progress_display': 1
}

theta = 10.81/180.0*pi
phi = 9.0/180.0*pi
h1 = np.array([0.0, 0.0, 0.25])
h2 = np.array([0.0, 0.0, 0.15])
L1 = 1.0
L2 = 0.8
D1 = 17.0
D2 = 16.5
rhat = np.array([cos(phi), sin(phi), 0.0])
xhat = np.array([sin(phi),-cos(phi), 0.0])
yhat = np.array([sin(theta)*cos(phi), \
                 sin(theta)*sin(phi), \
                 cos(theta)])
#size1 = 00.0e-04
size1 = 20.0e-04
size2 = 5.0e-04

protonEnergy1 = {
  'protonEnergyInit'       : 3.0
}

protonEnergy2 = {
  'protonEnergyInit'       : 14.7,
  'protonEnergyWidth'      : 0.0,
  'protonEnergyEnd'        : np.array([[13.0, 15.0],]),
  'protonSourceType'      : 0
}

protonEnergy3 = {
  'protonEnergyInit'       : 10.2,
  'protonEnergyEnd'        : np.array([[9.0, 11.0],]),
}

sourceAndTarget15 = {
  'protonCenter' : h1 - L1*rhat,
  'plateCenter'  : h1 + D1*rhat,
  'protonSize'   : size1
}

sourceAndTarget16 = {
  'protonCenter' : h2 - L1*rhat,
  'plateCenter'  : h2 + D1*rhat,
  'protonSize'   : size1
}

sourceAndTarget16EP = {
  'protonCenter' : h2 - L2*rhat,
  'plateCenter'  : h2 + D2*rhat,
  'protonSize'   : size2
}


plateOrientation1 = {
  'plateX' : xhat,
  'plateY' : yhat,
}

plateOrientation2 = {
  'plateX' : (xhat+yhat)*sqrt(0.5),
  'plateY' : (yhat-xhat)*sqrt(0.5),
}

args_data = {
'species' : {
    'cham': 'He',
    'targ': 'CH',
    },
'level_reduction' : 0,
#'level_reduction' : 3,
}

run="1.0_shok"

input_list = []
for run in ["1.0_shok", ]:
##for run in ["1.0_shok", "0.1_shok", "0.1",]:
    root="/net/scratch1/yclu/MagJet_production_OMEGA_4/RUN0_"+run+"/MagJet3D_hdf5_plt_cnt_00"
    for t in ["16", ]:
        input_list.append(
                [root+t, "_"+run+"_"+t+".npz", 
                    [args_data, ],
                    [geometryCommon, protonEnergy2, sourceAndTarget15, plateOrientation1],],
        
            )

proton_sim(input_list)
