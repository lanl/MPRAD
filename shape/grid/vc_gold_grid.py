import tifffile as tiff
import numpy as np

def vc_gold_grid(fn, mat_data, useMSC=True):
    print "Import TIFF image {0}".format(fn)
    output_data = {}
    data = tiff.imread(fn)
    j_list_for_bar = np.where(data[200:980,:350,1]>254)[1]
    delta_j_for_bar = j_list_for_bar.max() - j_list_for_bar.min()
    um_per_px = 250.0/delta_j_for_bar
    zmax = 501*um_per_px
    zmin = -zmax
    ymin = -zmax
    ymax = -0.4*zmax
    xmin = -25.0
    xmax = 25.0
    output_data['leftEdge'] = np.array([xmin, ymin, zmin])*1e-4
    output_data['rightEdge'] = np.array([xmax, ymax, zmax])*1e-4
    x_number_px = int(50.0/um_per_px)
    x_gold_px1 = int(16/um_per_px)
    x_gold_px2 = int(34/um_per_px)
    grid_data = np.zeros((x_number_px, 300, 1002),dtype=bool)
    grid_data[x_gold_px1:x_gold_px2] = (data[::-1,300:0:-1,0].T<16)
    
    for var in ('magx', 'magy', 'magz'): output_data[var] = np.zeros(grid_data.shape)
        
    msc_zbar = np.full(grid_data.shape, mat_data['CH']['zbar'])
    msc_zbar[grid_data] = mat_data['Au']['zbar']
    output_data['msc_zbar'] = msc_zbar
    msc_a1 = np.full(grid_data.shape, np.exp(mat_data['CH']['msc_a1']*mat_data['CH']['abar']))
    msc_a1[grid_data] = np.exp(mat_data['Au']['msc_a1']*mat_data['Au']['abar'])
    output_data['msc_a1'] = msc_a1
    
    for var in ('msc_a22', 'stop_a', 'stop_b'):
        output_data[var] = np.full(grid_data.shape,mat_data['CH'][var])*1.04
        output_data[var][grid_data] = mat_data['Au'][var]*19.32
    
    if not useMSC:
        output_data['msc_a1'] = 0.0
        output_data['msc_a22'] = 0.0
    
    return output_data