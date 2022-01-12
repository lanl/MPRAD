"""An example of input data for proton radiography simulation """

import argparse
import pprint
from math import pi, sqrt, sin, cos
import numpy as np
from proton import proton_sim, comm

# Unique Choice
species_table = {
'coaxis': ['Coaxis Target', 
    {'cham' : 'He', 
     'sphr' : 'CH', 
     'push' : 'CH', 
     'wall' : 'Be', 
     'foam' : 'CH', 
     }],
'slots_ch' : ['Slots Target', 
    {'cham' : 'He',
     'sphr' : 'CH', 
     'push' : 'CH',
     'wall' : 'Be',
     'foam' : 'CH', 
     'gcap' : 'Cu',
    }],
'slots_mg' : ['Slots Target', 
    {'cham' : 'He',
     'sphr' : 'Mg', 
     'push' : 'CH',
     'wall' : 'Be',
     'foam' : 'CH', 
     'gcap' : 'Cu',
    }],
'slots_cu' : ['Slots Target', 
    {'cham' : 'He',
     'sphr' : 'Cu', 
     'push' : 'CH',
     'wall' : 'Be',
     'foam' : 'CH', 
     'gcap' : 'Cu',
    }],
'marblevc' : ['Marble VC Target', 
    {'cham' : 'He',
     'sphr' : 'CH', 
     'push' : 'CH',
     'wall' : 'Be',
     'foam' : 'CH', 
    }],
}

# Unique Choice
shot_table = {
"HEDBDAY0": [
    [00000, 14.8, 0.250, 0.0, "HEDB1Red", ],
    ],
"HEDBDAY1": [
    [92075, 14.87, 0.276, 8.0, "HEDB1Red", ],
    [92085, 14.70, 0.244, 13.0, "HEDB1Red", ],
    ],
}

# Unique Choice
file_table = {
"oldres":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa/omega2017_hdf5_plt_cnt"},
"newres":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/test_slotsa_test8/omega2018_hdf5_plt_cnt"},
"newrescut":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin/cut_7.5um/omega2017_hdf5_plt_cnt"},
"newresnocut":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin/nocut_7.5um/omega2017_hdf5_plt_cnt"},
"newrescu":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin_cu/omega2017_hdf5_plt_cnt"},
"newres_thin_CH_cut":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin_all/omega2017CH_cut_hdf5_plt_cnt"},
"newres_thin_CH_nocut":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin_all/omega2017CH_nocut_hdf5_plt_cnt"},
"newres_thin_Cu_cut":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin_all/omega2017Cu_cut_hdf5_plt_cnt"},
"newres_thin_Cu_nocut":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin_all/omega2017Cu_nocut_hdf5_plt_cnt"},
"newres_thin_Cu_cut_foam30":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin_all/omega2017Cu_cut_foam30_hdf5_plt_cnt"},
"newres_thin_CH_cut_foam30":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin_all/omega2017CH_cut_foam30_hdf5_plt_cnt"},
"newres_thin_Cu_nocut_foam30":
    {"HEDB1Red" : "/net/scratch4/yclu/HEDB19/slotsa_newres_thin_all/CU_NOCUT_FOAM30/omega2017_hdf5_plt_cnt"},
}

# Unique Choice
ener_table = {
"pack1" : [2. , 11.0 , 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.7,
           12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8,
           13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 16.0]
}

opta_table = {
"optaL": 0.0,
"optaR": 180.0,
}

optb_table = {
"optb20" : 20.0,
"optb27" : 27.0,
}

optc_table = {
"optc0" : 0.0,
"optc1" : 1.0,
"optc3" : 3.0,
}

optd_table = {
"optd0" : {"split_x_type": 0, },
"optd1" : {"split_x_type": 1, 'split_x1_value': -1.0e99, 'split_x2_value': 0.0},
"optd2" : {"split_x_type": 1, 'split_x1_value': 0.0, 'split_x2_value':  1.0e99},
"optd3" : {"split_r_type": 2, 'split_r_value': 0.018}, # No field out, r<..
"optd4" : {"split_r_type": 1, 'split_r_value': 0.018}, # No field in, r>..
}



pps_radius = 100.0
pps0 = dict(
  protonNumber = int(4E7),
  pps_list_circ = np.array([[]]),
  pps_list1_poly = np.array([[]]),
  pps_list2_poly = np.array([[]]),
)
pps1 = dict(
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
)
pps2 = dict(
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
  pps_dE1 = 1e99,
  pps_dE2 = 0.0669,
  pps_ang_strag = 34.901e-3,
  protonNumber = int(3.216E7),
)

opte_table = {
"opte0" : pps0,
"opte1" : pps1,
"opte2" : pps2,
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--species_table',  type=str, help='Species table, options: ' + \
        ', '.join(['{0} for {1}'.format(key, species_table[key][0]) for key in species_table]))
parser.add_argument('--shot_table', type=str, help='Shot table, options: ' + \
        ', '.join(['{0}'.format(key) for key in shot_table]))
parser.add_argument('--file_table', type=str, help='Input file table, options: ' + \
        ', '.join(['{0}'.format(key) for key in file_table]))
parser.add_argument('--ener_table',  type=str, help='Target energy table, options: ' + \
        ', '.join(['{0}'.format(key) for key in ener_table]))
parser.add_argument('--output_base', '-o', type=str, help='Output base name')
parser.add_argument('--index', '-id', nargs=3, type=int, help='Index, w.r.t. the time in shot table')

parser.add_argument('--optaL', action="store_true", help='Protons go from -x to +x')
parser.add_argument('--optaR', action="store_true", help='Protons go from +x to -x')

parser.add_argument('--optb20', action='store_true', help="CR-39 is 20cm away")
parser.add_argument('--optb27', action='store_true', help="CR-39 is 27cm away")

parser.add_argument('--optc0', action='store_true', help='B field Magnification: 0')
parser.add_argument('--optc1', action='store_true', help='B field Magnification: 1')
parser.add_argument('--optc3', action='store_true', help='B field Magnification: 3')

parser.add_argument('--optd0', action='store_true', help='Always turn on B field')
parser.add_argument('--optd1', action='store_true', help='Turn on B field if x<0')
parser.add_argument('--optd2', action='store_true', help='Turn on B field if x>0')
parser.add_argument('--optd3', action='store_true', help='Turn on B field if r<0.015')
parser.add_argument('--optd4', action='store_true', help='Turn on B field if r<0.015')

parser.add_argument('--opte0', action='store_true', help='Solid angle is 14.2')
parser.add_argument('--opte1', action='store_true', help='Solid angle is 0.43')
parser.add_argument('--opte2', action='store_true', help='Solid angle is 0.57')
parser.add_argument('--opte3', action='store_true', help='Solid angle is 0.71')

args = parser.parse_args()

i0, i1, ii = args.index
is_root = True if comm.rank == 0 else False
species = species_table[args.species_table][1]
shot_t = shot_table[args.shot_table]
file_t = file_table[args.file_table]
ener_t = ener_table[args.ener_table]
erange = np.array([[ener_t[i], ener_t[i+1]] for i in range(len(ener_t)-1)])
fn_out_base = args.output_base
if is_root:
    print('---------- BEGIN ARGS WITH UNIQUE CHOICE ----------')
    print('Species:')
    pprint.pprint(species)
    print('Shots:')
    pprint.pprint(shot_t)
    print('Files:')
    pprint.pprint(file_t)
    print('Energy Range:')
    pprint.pprint(erange)
    print('Output base name: {0}'.format(fn_out_base))
    print('File number range w.r.t the central frame:')
    pprint.pprint(list(range(i0, i1, ii)))
if is_root: print('---------- END ARGS WITH UNIQUE CHOICE----------')



erange = np.array(erange)


args_input = {
'species': species,
'usePlasma' : True,
}

input_list = []
if is_root: print('---------- BEGIN INPUT AND OUTPUT LIST ----------')
for shot_no, ener_init, ener_sigma, t0, target in shot_t:
    for i in range(int(t0*10)+i0, int(t0*10)+i1, ii):
        fn_in = "{0}_{1:04d}".format(file_t[target], i)
        if is_root: print("fn_in : "+ fn_in)
        for opta in opta_table:
            if getattr(args, opta): phi = opta_table[opta]
            else: continue
            for optb in optb_table:
                if getattr(args, optb): D1 = optb_table[optb]
                else: continue
                for optc in optc_table:
                    if getattr(args, optc):  bmag = optc_table[optc]
                    else: continue
                    for optd in optd_table:
                        if getattr(args, optd): split_x = optd_table[optd]
                        else: continue
                        for opte in opte_table:
                            if getattr(args, opte): pps = opte_table[opte]
                            else: continue
                            fn_out = "{0}_SHOT{1}_{2}_{3}_{4}_{5}_{6}_{7:04d}.npz".format(fn_out_base, shot_no, opta, optb, optc, optd, opte, i)
                            theta = 0.0
                            h1 = np.array([0.0, 0.0, 0.0])
                            L1 = 1.0
                            rhat = np.array([cos(phi), sin(phi), 0.0])
                            xhat = np.array([sin(phi),-cos(phi), 0.0])
                            yhat = np.array([sin(theta)*cos(phi), \
                                 sin(theta)*sin(phi), \
                                 cos(theta)])
                            geometryCommon = {
                              'protonAngle'  : 14.2,
                              'plateRange'   : np.array([[-5.0,5.0],[-5.0,5.0]]),
                              'binX'         : 500,
                              'binY'         : 500,
                              'dtLimitLow'   : 0.00001,
                              'dtLimitHigh'  : 0.0002,
                              'dvLimit'      : 0.001,
                              'countAllProcs': 0,
                              'progress_display': 0,
                            }

                            sourceAndTarget1 = {
                              'protonCenter' : h1 - L1*rhat,
                              'plateCenter'  : h1 + D1*rhat,
                            }
                            
                            plateOrientation1 = {
                              'plateX' : xhat,
                              'plateY' : yhat,
                            }

                            protonEnergy1 = {
                              'protonSize'             : 20.0e-04,
                              'protonEnergyInit'       : ener_init, 
                              'protonEnergyWidth'      : ener_sigma,
                              'protonSourceType'       : 2,
                              'protonEnergyEnd'        : erange,
                            }
                            if is_root: print("fn_out : " + fn_out)
                            input_list.append([
                                fn_in, fn_out, 
                                [args_input, ],
                                [geometryCommon, protonEnergy1, sourceAndTarget1, split_x, 
                                    {'multimag': np.array([bmag]*3)}, plateOrientation1, pps],
                            ])
if is_root: print('---------- END INPUT AND OUTPUT LIST ----------')

proton_sim(input_list)
