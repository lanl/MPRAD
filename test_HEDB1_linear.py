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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--species_table',  type=str, help='Species table, options: ' + \
        ', '.join(['{0} for {1}'.format(key, species_table[key][0]) for key in species_table]))
parser.add_argument('--shot_table', type=str, help='Shot table, options: ' + \
        ', '.join(['{0}'.format(key) for key in shot_table]))
parser.add_argument('--file_table', type=str, help='Input file table, options: ' + \
        ', '.join(['{0}'.format(key) for key in file_table]))
parser.add_argument('--output_base', '-o', type=str, help='Output base name')
parser.add_argument('--index', '-id', nargs=3, type=int, help='Index, w.r.t. the time in shot table')

args = parser.parse_args()

i0, i1, ii = args.index
is_root = True if comm.rank == 0 else False
species = species_table[args.species_table][1]
shot_t = shot_table[args.shot_table]
file_t = file_table[args.file_table]
fn_out_base = args.output_base
if is_root:
    print('---------- BEGIN ARGS WITH UNIQUE CHOICE ----------')
    print('Species:')
    pprint.pprint(species)
    print('Shots:')
    pprint.pprint(shot_t)
    print('Files:')
    pprint.pprint(file_t)
    print('Output base name: {0}'.format(fn_out_base))
    print('File number range w.r.t the central frame:')
    pprint.pprint(list(range(i0, i1, ii)))
if is_root: print('---------- END ARGS WITH UNIQUE CHOICE----------')

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
        fn_out = "{0}_SHOT{1}_{2:04d}.npz".format(fn_out_base, shot_no, i)
        geometryCommon = {
          'protonEnergyInit'       : ener_init, 
          'protonEnergyWidth'      : ener_sigma,
          'axis'                   : '+x',
        }
        if is_root: print("fn_out : " + fn_out)
        input_list.append([
            fn_in, fn_out, 
            [args_input, ],
            [geometryCommon, ],
        ])
if is_root: print('---------- END INPUT AND OUTPUT LIST ----------')

proton_sim(input_list, linear=True)
