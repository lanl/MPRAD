"""An example of input data for proton radiography simulation """

from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_sim
from matplotlib import pyplot as plt
import os

geometryCommon = dict(
  protonAngle = 10.0,
  plateRange = np.array([[-1.0,1.0],[-1.0,1.0]]),
  binX = 500,
  binY = 500,
  dtLimitLow = 0.00001,
  dtLimitHigh = 0.0001,
  dvLimit = 0.001,
  countAllProcs = 0,
  protonNumber = int(4E7),
  progress_display = -1,
)

theta = 0.0
phi = 0.0
h1 = np.array([0.0, 0.0, 0.0])
rhat = np.array([cos(phi), sin(phi), 0.0])
xhat = np.array([sin(phi),-cos(phi), 0.0])
yhat = np.array([sin(theta)*cos(phi), \
                 sin(theta)*sin(phi), \
                 cos(theta)])

plateOrientation1 = {
  'plateX' : xhat,
  'plateY' : yhat,
}


par_list = [
    ['AU_TMD', 50.0, 0.288288, 0.056621, 0.005059, 0.000357, 27.7, 0.4],
    ['AU_TMD', 40.0, 0.338000, 0.056321, 0.006293, 0.000440, 27.7, 0.4],
    ['AU_TMD', 30.0, 0.415289, 0.056022, 0.008348, 0.000579, 27.7, 0.4],
    ['AU_TMD', 20.0, 0.554348, 0.055724, 0.012458, 0.000857, 27.7, 0.4],
    ['AU_TMD', 10.0, 0.894862, 0.055428, 0.024787, 0.001691, 27.7, 0.4],
    ['AU_HEDB', 14.8, 0.684684, 0.055570, 0.016790, 0.001150, 20.13, 0.62],
]
for name, E0, dE1, dE2, msc_chi, msc_chi_min, D1, L1 in par_list:
    #erange = np.linspace(E0-dE1-10.0*dE2, E0, 10)
    #erange = np.array([[erange[i], erange[i+1]] for i in range(len(erange)-1)], dtype=np.float64)
    #print(erange)
    erange = np.array([[E0-10.0,E0]])
    grid = dict(
      use_grid = 1, 
      grid_lambda_x = 62.0e-4, grid_lambda_y = 62.0e-4, 
      grid_xbar = 25.0e-4, grid_ybar = 25.0e-4, 
      grid_center = np.array([0.0,0.0,0.0]),
      grid_x = np.array([0.0, -1.0, 0.0]),
      grid_y = np.array([0.0, 0.0, 1.0]),
      grid_dE1 = dE1,
      grid_dE2 = dE2,
      grid_chi = msc_chi,
      grid_chi_min = msc_chi_min,
    )
    protonEnergy1 = dict(
      protonEnergyInit = E0,
      protonEnergyWidth = 0.00,
      protonSourceType = 0,
      protonEnergyEnd = erange,
    )
    sourceAndTarget1 = dict(
      protonCenter = h1 - L1*rhat,
      plateCenter = h1 + D1*rhat,
      protonSize = 0.0e-4
    )

    onecell = dict(
      dens = 1.0e-20,
      mat = 'Au',
      leftEdge = -np.array([1.0, 1000.0, 1000.0])*1e-4,
      rightEdge = np.array([1.0, 1000.0, 1000.0])*1e-4,
      usePlasma = False,
    )

    fn = "{0}_E{1}_D{2}_L{3}".format(name, E0, D1, L1)

    args_input = dict(
      onecell = onecell,
    )
    
    input_list = [
            [fn, "{0}.npz".format(fn), 
            [args_input, ],
            [geometryCommon, protonEnergy1, sourceAndTarget1, plateOrientation1, grid]
    ],
    ]
    proton_sim(input_list)

