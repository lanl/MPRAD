import numpy as np
from yt import units as u
import pprint
tiny = np.finfo(np.float64).tiny

def onecell(mat_data, onecell):
    output_data = {}
    if 'print' in onecell:
        pprint.pprint(onecell)

    usePlasma = onecell['usePlasma']

    mat = onecell['mat']
    rho = onecell['dens']
    if 'mag_npz' in onecell:
        mag_npz = np.load(onecell['mag_npz'])
        shape = mag_npz['magx'].shape
    elif 'shape' in onecell:
        shape = onecell['shape']
    else:
        shape = (1,1,1)

    if usePlasma: 
        tele = onecell['tele'] * u.K
        nele = onecell['nele'] * u.cm**-3
        cfrac = 1.0/(1.0 + mat_data['coef_shield'] * np.exp(mat_data[mat]['logzbar']*2.0/3.0+tiny) * tele / nele)
        pfrac = 1.0 - cfrac
    else:
        cfrac = 1.0
        pfrac = 0.0


    if 'mag_npz' in onecell:
        output_data['leftEdge'] = mag_npz['leftEdge']
        output_data['rightEdge'] = mag_npz['rightEdge']
    else:
        output_data['leftEdge'] = onecell['leftEdge']
        output_data['rightEdge'] = onecell['rightEdge']
    
    if 'mag_npz' in onecell:
        output_data['magx'] = mag_npz['magx']
        output_data['magy'] = mag_npz['magy']
        output_data['magz'] = mag_npz['magz']
    else:
        for var in ('magx', 'magy', 'magz'): output_data[var] = np.zeros(shape)
        
    msc_a1 = np.full(shape, np.exp(mat_data[mat]['msc_a1']*mat_data[mat]['abar']))
    if usePlasma:
        msc_a1 = msc_a1 * cfrac
    output_data['msc_a1'] = msc_a1
    output_data['msc_a22'] = np.full(shape,mat_data[mat]['msc_a22'])*rho
    output_data['msc_zbar'] = np.full(shape, np.sqrt(mat_data[mat]['z2bar']))

    output_data['stop_a'] = np.full(shape, mat_data[mat]['a_cold'] * cfrac + (mat_data[mat]['a_plasma']-0.5*np.log(rho)) * pfrac)
    output_data['stop_b'] = np.full(shape, mat_data[mat]['stop_b'] * rho)
    output_data['stop_strag'] = np.full(shape, mat_data['coef_straggling']*rho*mat_data[mat]['zbar']/mat_data[mat]['abar'])

    if 'print' in onecell:
        #print("cfrac = {0}, pfrac = {1}".format(cfrac, pfrac))
        pprint.pprint(mat_data[mat])
        for key in output_data:
            pprint.pprint(key)
            pprint.pprint(output_data[key][0])
    
    return output_data
