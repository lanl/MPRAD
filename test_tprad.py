"""An example of input data for proton radiography simulation """

from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_core

geometryRange= {
  'leftEdge'     : np.array([-0.1, -0.04, -0.04]),
  'rightEdge'    : np.array([ 0.1,  0.04,  0.04]),
}
geometryArray = dict([(var, np.zeros((1,1,1), np.float64)) for var in \
       ["magx", "magy", "magz", "msc_zbar", "msc_a1", "msc_a22", "stop_b"]])
geometryArray['stop_a'] = np.array([[[0.0]]])

geometryCommon = {
  'binX'         : 250,
  'binY'         : 250,
  ##'dtLimitLow'   : 1e-7,
  'dtLimitLow'   : 2e-6,
  'dtLimitHigh'  : 1e-4,
  'dvLimit'      : 1e-3,
  ##'protonNumber' : 384,
  'protonNumber' : int(1E5),
  'countAllProcs': 0,
  'progress_display': 1,
  'em_analysical' : 1,
  ##'em_parset' : np.array([[1.0, \
  ##        00.0e6/sqrt(4*pi), -0.0e12/3e4/sqrt(4*pi), 0.004, 0.02, \
  ##        0.1e12/3e4/sqrt(4*pi), 0.02, 1e-10, 0.02]]),
  'em_parset' : np.array([[1.0, \
          60.0e6/sqrt(4*pi), -0.1e12/3e4/sqrt(4*pi), 0.004, 0.02, \
          0.1e12/3e4/sqrt(4*pi), 0.02, 1e-10, 0.02]]),
  'track_n' : 384,
  'track_leap_n' : 512,
  'track_max_step': 1024,
  'hit_3d_n' : 384,
  'protonSourceLinear' : 1,
}


protonEnergy1 = {
  'protonCenter'      : np.array([0.0, 0.0, 0.255]),
  'protonAngle'       : 5.0,
  'protonSize'        : 0.0,
  'protonEnergyInit'  : 10.0, 
  'protonEnergyWidth' : 0.00,
  'protonSourceType'  : 0,
  'protonEnergyEnd'   : np.array([[1.0, 100.0],])
}

plateOrientation1 = {
  'plateX'      : np.array([1.0,0.0,0.0]),
  'plateY'      : np.array([0.0,1.0,0.0]),
  'plateCenter' : np.array([0.0, 0.0, -6.5]),
  'plateRange'  : np.array([[-2.5,2.5],[-2.5,2.5]]),
  ##'plateRange'  : np.array([[-10.0,10.0],[-10.0,10.0]]),
  #'plateCenter' : np.array([0.0, 0.0, -0.041]),
  #'plateRange'  : np.array([[-0.1,0.1],[-0.1,0.1]]),
}

paraset = {}
for d in [geometryRange, geometryArray, geometryCommon, protonEnergy1, plateOrientation1]:
    paraset.update(d)
image=proton_core(**paraset)
np.savez_compressed("_a.npz",  **image)

