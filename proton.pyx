#!python
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=True
### cython: wraparound=False

cimport cython, openmp
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.math cimport sqrt, floor, cos, sin, pi, fmin, fmax, trunc, log, fabs, exp, copysign
from cython.parallel cimport prange, parallel
from scipy.linalg.cython_lapack cimport dgesv

from time import time
import numpy as np
from mpi4py import MPI
import yt
from yt import derived_field
from yt import physical_constants as p
from yt import units as u

from data.mat_data import mat_data

from yt.utilities.physical_constants import Na, kb

def getnumthreads():
    """Get the total number of threads from enviornment variables"""
    cdef int num_threads
    with nogil, parallel():
        num_threads = openmp.omp_get_num_threads()
        with gil: return num_threads

class comm_class:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ntasks = comm.Get_size()
    num_threads = getnumthreads()
comm=comm_class()

def import_seeds(fn="data/mcg59seed_3000000019_10000.npz"):
    if comm.rank == 0:
        rng_seeds = np.load(fn)['data']
        ##rng_seeds = np.arange(2**32, 2**32+10000, dtype=np.uint64)
    else:
        rng_seeds = None
    rng_seeds = comm.comm.bcast(rng_seeds, root=0)
    
    max_n = rng_seeds.shape[0]
    omp_x_np = comm.ntasks*comm.num_threads
    stride = int(max_n/omp_x_np)
    if stride==0:
        if comm.rank==0: print("Max OMP_NUM_THREADS * np is {0}!".format(max_n))
        comm.comm.Abort()
    rng_seeds = rng_seeds[0:stride*omp_x_np:stride]
    return rng_seeds

def import_msc_dist(fn1="data/msc_dist_data1.csv", fn2="data/msc_dist_data2.csv"):
    if comm.rank==0:
        data1 = np.loadtxt(fn1)
        data2 = np.loadtxt(fn2).reshape((5,101))
    else:
        data1 = None
        data2 = None
    data1 = comm.comm.bcast(data1, root=0)
    data2 = comm.comm.bcast(data2, root=0)
    return 100, data1, data2

yt.enable_parallelism(communicator=comm.comm)

########## BEGIN BASIC OPERATORS ##########
cdef void vec_norm_perp(double *vn, double *vp1, double * vp2) nogil:
    cdef double inorm
    inorm = 1.0/ sqrt(vn[0]**2+vn[1]**2+vn[2]**2)
    vn[0] = vn[0] * inorm; vn[1] = vn[1] * inorm; vn[2] = vn[2] * inorm
    if 1.0-vn[0]**2 > 0.2: vp1[0] = 0.0; vp1[1] = -vn[2]; vp1[2] = vn[1]
    else: vp1[0] = vn[2]; vp1[1] = 0.0; vp1[2] = -vn[0]
    inorm = 1.0/sqrt(vp1[0]**2+vp1[1]**2+vp1[2]**2)
    vp1[0] = vp1[0] * inorm; vp1[1] = vp1[1] * inorm; vp1[2] = vp1[2] * inorm
    vp2[0] = vn[1]*vp1[2] - vn[2]*vp1[1]
    vp2[1] = vn[2]*vp1[0] - vn[0]*vp1[2]
    vp2[2] = vn[0]*vp1[1] - vn[1]*vp1[0]
########## END BASIC OPERATORS ##########

########## BEGIN RANDOM NUMBER GENERATORS ##########
# Initialize the random number seed
cdef void rand_init(unsigned long *stream, int tid, int ntasks, unsigned long [:] rng_seeds) nogil:
    stream[0] = rng_seeds[tid] + tid

# Generate random number in [0, 1]
cdef double rand(unsigned long *stream) nogil:
    cdef unsigned long a = 13**13
    cdef unsigned long m = 2**59
    cdef double im = 1.734723475976807e-18
    cdef double r0, r
    stream[0] = (a * stream[0]) % m
    r0 = stream[0]*im
    r = r0 - trunc(r0)
    if r<1e-15: return 0.5
    return r

# Generate random number in [a, b]
cdef void rand_real(unsigned long *stream,  double *result, unsigned int n, double a, double b) nogil:
    cdef unsigned int i,j
    cdef double r
    for i in range(n):
        r = rand(stream)
        result[i] = a + r * (b -a)

# Generate uniformly distributed random angle with in 0.0 to thetamax
cdef void rand_angle(unsigned long *stream, double *xyz, double sinHalfTheta) nogil:
    cdef double w, vn, sq, x1, x2
    w = 2.0
    while w > 1.0:
        x1 = 2.0*rand(stream) -1.0
        x2 = 2.0*rand(stream) -1.0
        w = x1**2 + x2**2
    vn = 1.0 - 2.0 * sinHalfTheta**2 * w;
    sq = sqrt(1.0 - sinHalfTheta**2 * w);
    xyz[0] = 2.0 * sinHalfTheta * x1 * sq;
    xyz[1] = 2.0 * sinHalfTheta * x2 * sq;
    xyz[2] = vn

# Generate triangle distributed random number
cdef void rand_trig(unsigned long *stream,  double *result, unsigned int n, double a, double fwhm) nogil:
    cdef unsigned int i,j
    cdef double r
    for i in range(n):
        r = rand(stream)
        if r > 0.5: result[i] = a+fwhm*(1.0-sqrt(2*r-1.0))
        else: result[i] = a-fwhm*(1.0-sqrt(1.0-2*r))

# Generate random numbers with gaussian distribution
cdef void rand_gaussian(unsigned long *stream, double *result, unsigned int n, double sigma) nogil:
    cdef double w, x1, x2
    while n>0:
        w = 2.0
        while w>1.0 or w < 1.0e-6:
            x1 = 2.0*rand(stream) - 1.0
            x2 = 2.0*rand(stream) - 1.0
            w = x1**2 + x2**2
        w = sqrt( (-2.0*log(w) ) / w)
        n = n-1
        result[n] = x1 * w * sigma
        if n!=0: 
            n = n-1
            result[n] = x2 * w *sigma

# Generate random numbers with exponential distribution exp((a-x)/b)
cdef void rand_exp(unsigned long *stream, double *result, unsigned int n, double a, double b) nogil:
    cdef double x
    while n>0:
        n = n - 1
        x = rand(stream)
        result[n] = a - b * log(x)

cdef void rand_coulumb(unsigned long *stream, double *vec, double msc_chi, double msc_chi_min, \
        double msc_p1, double msc_p2, int msc_dist_data_npt, double [:] msc_dist_data1, double [:,:] msc_dist_data2) nogil:
    cdef double x0, x1, x2, x3
    cdef double x, y, z
    cdef double vp1[3]
    cdef double vp2[3]
    cdef double p_left, p_now
    cdef double mb_x1, mb_x2, mb_x3, mb_z1, mb_z0, mb_b, mb_B, mb_frac, mb_iB
    cdef int mb_i, mb_stat

    p_left = (msc_chi/msc_chi_min)**2

    # Multiple scattering
    if p_left > msc_p2:
        mb_b = log(2.32929034*p_left)
        mb_B = mb_b
        for mb_i in range(14): mb_B = mb_b + log(mb_B)
        mb_iB = 1.0/mb_B

        mb_x1 = rand(stream)
        mb_stat = 0
        if mb_x1 < (msc_dist_data1[0]+(msc_dist_data1[1]+msc_dist_data1[2]*mb_iB)*mb_iB):
            while mb_stat==0:
                mb_x2 = rand(stream)
                mb_z1 = sqrt(-log(1.0-msc_dist_data1[0]*mb_x2))
                mb_z0 = mb_z1/1.8
                mb_x3 = rand(stream)
                mb_i = <int> floor(mb_z0 * msc_dist_data_npt)
                mb_frac = mb_z0 * msc_dist_data_npt - mb_i
                if  1.0 + ((msc_dist_data2[0][mb_i]*(1.0-mb_frac)+msc_dist_data2[0][mb_i+1]*mb_frac) \
                         + (msc_dist_data2[1][mb_i]*(1.0-mb_frac)+msc_dist_data2[1][mb_i+1]*mb_frac)*mb_iB)*mb_iB > \
                     mb_x3 * (1.0 + (msc_dist_data2[0][0]+msc_dist_data2[1][0]*mb_iB)*mb_iB):
                    mb_stat = 1
        elif mb_x1 < (msc_dist_data1[3]+(msc_dist_data1[4]+msc_dist_data1[5]*mb_iB)*mb_iB):
            while mb_stat==0:
                mb_x2 = rand(stream)
                mb_i = <int> floor(mb_x2 * msc_dist_data_npt)
                mb_frac = mb_x2 * msc_dist_data_npt - mb_i
                mb_x3 = rand(stream)
                if  (msc_dist_data2[2][mb_i]*(1.0-mb_frac)+msc_dist_data2[2][mb_i+1]*mb_frac) + \
                   ((msc_dist_data2[3][mb_i]*(1.0-mb_frac)+msc_dist_data2[3][mb_i+1]*mb_frac) + \
                    (msc_dist_data2[4][mb_i]*(1.0-mb_frac)+msc_dist_data2[4][mb_i+1]*mb_frac)*mb_iB)*mb_iB > \
                     mb_x3 * (msc_dist_data2[2][0]+(msc_dist_data2[3][0]+msc_dist_data2[4][0]*mb_iB)*mb_iB):
                    mb_stat = 1
                    mb_z1 = 1.8+8.2*mb_x2
        else:
            mb_z1 = 10.0
        x1 = 2.0*pi*rand(stream)
        x3 = msc_chi * sqrt(mb_B) * mb_z1
        vec_norm_perp(vec, vp1, vp2)
        x = sin(x3)*cos(x1)
        y = sin(x3)*sin(x1)
        z = cos(x3)
        vec[0] = vp1[0]*x + vp2[0]*y + vec[0]*z 
        vec[1] = vp1[1]*x + vp2[1]*y + vec[1]*z 
        vec[2] = vp1[2]*x + vp2[2]*y + vec[2]*z 
        p_left = 0.0

    # Single and differential scattering
    while p_left>0.0:
        x0 = rand(stream)
        p_now = - log(1.0-x0)
        if p_now >= p_left:
            p_now = fmin(p_left, msc_p1)
            x0 = rand(stream)
            if x0 >= p_now: 
                p_left = p_left - p_now
                continue
        p_left = p_left - p_now
        x1 = 2.0*pi*rand(stream)
        x2 = rand(stream)
        x3 = msc_chi_min / sqrt(x2)
        vec_norm_perp(vec, vp1, vp2)
        x = sin(x3)*cos(x1)
        y = sin(x3)*sin(x1)
        z = cos(x3)
        vec[0] = vp1[0]*x + vp2[0]*y + vec[0]*z 
        vec[1] = vp1[1]*x + vp2[1]*y + vec[1]*z 
        vec[2] = vp1[2]*x + vp2[2]*y + vec[2]*z 

    vec_norm_perp(vec, vp1, vp2)

########## END RANDOM NUMBER GENERATORS ##########

########## BEGIN EM FORM ##########
cdef void em_form(double *pos, double *e_field, double *b_field, double [:,:] em_parset) nogil:
    cdef double B0, E0, R0, Eth0, dth0, dth1, coef, Rmax, Rmax_th
    cdef double r2, r, cos_phi, sin_phi, B_theta, E_r, E_x
    r = sqrt(pos[2]**2+pos[1]**2)
    if r<1e-15: 
        cos_phi=0.0
        sin_phi=1.0
    else:
        cos_phi = pos[2]/r
        sin_phi = pos[1]/r
    if em_parset[0,0] !=0.0:
        coef = em_parset[0,0]

        B0 = em_parset[0,1]
        E0 = em_parset[0,2]
        R0 = em_parset[0,3]
        Rmax = em_parset[0,4]

        Eth0 = em_parset[0,5]
        dth0 = em_parset[0,6]
        dth1 = em_parset[0,7]
        Rmax_th = em_parset[0,8]

        if r<R0:
            B_theta = B0*(r/R0)
            E_r = E0*(r/R0)
        elif r<Rmax:
            B_theta = B0*(Rmax-r)/(Rmax-R0)
            E_r = E0*(Rmax-r)/(Rmax-R0)
        else:
            B_theta=0.0
            E_r=0.0

        if r<Rmax_th:
            if pos[0] > 0:
                E_x = Eth0*exp(-fabs(pos[0])/dth0)
            else:
                E_x = -Eth0*exp(-fabs(pos[0])/dth1)
        else:
            E_x = 0.0

        if pos[0] < 0:
            B_theta = -B_theta

        e_field[0] += E_x * coef
        e_field[2] += E_r * cos_phi * coef
        e_field[1] += E_r * sin_phi * coef
        b_field[0] += 0.0*coef
        b_field[2] += -B_theta*sin_phi*coef
        b_field[1] += B_theta*cos_phi*coef

cdef void em_form2(double *pos, double *e_field, double *b_field, double [:,:] em_parset) nogil:
    cdef double coef_phi, coef_y
    coef_phi = em_parset[0,0] / em_parset[0,1] / sqrt(4.0*pi) \
            * exp(-(pos[0]**2+pos[1]**2+pos[2]**2) / em_parset[0,1]**2) 
    coef_y = em_parset[1,0] / sqrt(4.0*pi) \
            * exp(-(pos[0]**2+pos[2]**2) / em_parset[1,1]**2) 
    e_field[0] = 0.0
    e_field[1] = 0.0
    e_field[2] = 0.0
    b_field[0] = - coef_phi * pos[1]
    b_field[1] =   coef_phi * pos[0] + coef_y
    b_field[2] = 0.0
########## END EM FORM ##########

def field_2dcyl_to_3dcart_scalor(data_in):
    cdef double r1, x1, y1
    cdef int k, k0, k1, ij1, i_src, i_targ, j_targ
    cdef double [:,:] data_src
    cdef double [:,:,:] data_targ
    cdef double [:,:,:] data_targ_

    data_src = data_in
    shape_src = data_src.shape
    shape_targ = (2*shape_src[0], 2*shape_src[0], shape_src[1])
    data_targ = np.zeros(shape_targ, dtype=np.float64)
    iPerTask = (shape_src[1]+comm.ntasks-1) / comm.ntasks
    k0 = min(iPerTask * comm.rank, shape_src[1])
    k1 = min(k0 + iPerTask, shape_src[1])
    ij1 = shape_src[0]
    with nogil, parallel():
        for k in prange(k0, k1, schedule='guided', chunksize=10):
            for i_targ in range(ij1*2):
                for j_targ in range(ij1*2):
                    x1 =  0.5 + <double>i_targ - <double>ij1
                    y1 =  0.5 + <double>j_targ - <double>ij1
                    r1 = sqrt(x1**2+y1**2)
                    i_src = <int>floor(r1)
                    if i_src < ij1: data_targ[i_targ, j_targ, k] = data_src[i_src,k]
    if comm.ntasks == 1: return data_targ
    else:
        data_targ_ = np.empty(shape_targ, dtype=np.float64)
        comm.comm.Allreduce(data_targ, data_targ_, op=MPI.SUM)
    return data_targ_

def field_2dcart_to_3d(data_in, ratio_z_to_x):
    cdef double [:,:] data_src
    cdef double [:,:,:] data_targ
    cdef int shape_x, shape_y, shape_z
    cdef int i, j, k

    data_src = data_in
    shape_src = data_src.shape
    shape_x = shape_src[0]
    shape_y = shape_src[1]
    #shape_z = int(shape_src[0]*ratio_z_to_x)
    shape_z = 1
    data_targ = np.zeros((shape_x, shape_y, shape_z), dtype=np.float64)
    with nogil, parallel():
        for i in range(shape_x):
            for j in prange(0, shape_y, schedule='guided', chunksize=10):
                for k in range(shape_z):
                    data_targ[i,j,k] = data_src[i,j]
    return data_targ

def field_2dcyl_to_3dcart_vector(data_in):
    cdef double cos1, sin1, r1, x1, y1
    cdef int k, k0, k1, ij1, i_src, i_targ, j_targ
    cdef double [:,:] data_src_x, data_src_y, data_src_z
    cdef double [:,:,:] data_targ_x, data_targ_y, data_targ_z

    data_src_x, data_src_y, data_src_z = data_in
    shape_src = data_src_x.shape
    shape_targ = (2*shape_src[0], 2*shape_src[0], shape_src[1])
    data_targ = [np.zeros(shape_targ, dtype=np.float64) for i in range(3)]
    data_targ_x, data_targ_y, data_targ_z = data_targ
    iPerTask = (shape_src[1]+comm.ntasks-1) / comm.ntasks
    k0 = min(iPerTask * comm.rank, shape_src[1])
    k1 = min(k0 + iPerTask, shape_src[1])
    ij1 = shape_src[0]
    with nogil, parallel():
        for k in prange(k0, k1, schedule='guided', chunksize=10):
            for i_targ in range(ij1*2):
                for j_targ in range(ij1*2):
                    x1 =  0.5 + <double>i_targ - <double>ij1
                    y1 =  0.5 + <double>j_targ - <double>ij1
                    r1 = sqrt(x1**2+y1**2)
                    i_src = <int>floor(r1)
                    if i_src < ij1:
                        cos1=x1/r1
                        sin1=y1/r1
                        data_targ_x[i_targ, j_targ, k] = data_src_x[i_src,k]*cos1 - data_src_z[i_src,k]*sin1
                        data_targ_y[i_targ, j_targ, k] = data_src_x[i_src,k]*sin1 + data_src_z[i_src,k]*cos1
                        data_targ_z[i_targ, j_targ, k] = data_src_y[i_src,k]
    if comm.ntasks == 1:
        return data_targ
    else:
        data_targ_ =[np.empty(shape_targ, dtype=np.float64) for i in range(3)]
        for i in range(3): comm.comm.Allreduce(data_targ[i], data_targ_[i], op=MPI.SUM)
    return data_targ_

def fields_from_flash(filename="", species={}, level_reduction=0, \
        useMSC=True, usePlasma=False, specialGrid='', \
        cart2d_to_3d_length=0.1, onecell=None, msc_w=False):
    """Import magnetic field and density data from FLASH 2D/3D simulation, 
    densities are converted to multiple scattering and stop range data"""

    if specialGrid=='vc_gold_grid':
        import sys
        sys.path.append('shape/grid/')
        from vc_gold_grid import vc_gold_grid
        return vc_gold_grid(filename, mat_data, useMSC)

    if onecell is not None:
        import sys
        sys.path.append('shape/grid/')
        from onecell import onecell as onecell_func
        return onecell_func(mat_data, onecell)

    ds = yt.load(filename)
    geodim = (ds.geometry, ds.dimensionality)
    tiny = np.finfo(np.float64).tiny

    # Calculate missing variables from FLASH data
    if len(species)!=0:
        ##if ('flash','zbar') not in ds.field_list:
        ##    def _zbar(field, data):
        ##        try: return data['flash','ye']/(data['flash', 'sumy']+tiny)
        ##        except: return data['flash','ye  ']/(data['flash', 'sumy']+tiny)
        ##    ds.add_field('zbar', function=_zbar, units="")
        if ('flash', 'abar') not in ds.field_list:
            def _abar(field, data):
                return 1.0/(data['flash', 'sumy']+tiny)
            ds.add_field('abar', function=_abar, units="")

    # For Marbal VC
    if all(x in species for x in ['cham', 'sphr', 'push', 'wall', 'foam']) and 'gcap' not in species:
        if ('flash', 'foam') not in ds.field_list:
            if comm.rank == 0: print("foam = 1.0 - cham - sphr - push - wall")
            def _foam(field, data):
                a = 1.0 - data['flash', 'cham'] - data['flash', 'sphr']\
                        - data['flash', 'push'] - data['flash', 'wall']
                a = np.minimum(1.0,a)
                a = np.maximum(0.0,a)
                return a
            ds.add_field('foam', function=_foam, units="")

    if all(x in species for x in ['cham', 'sphr', 'push', 'wall', 'foam', 'gcap']):
        if ('flash', 'foam') not in ds.field_list:
            if comm.rank == 0: print("foam = 1.0 - cham - sphr - push - wall - gcap")
            def _foam(field, data):
                a = 1.0 - data['flash', 'cham'] - data['flash', 'sphr'] \
                        - data['flash', 'push'] - data['flash', 'wall'] \
                        - data['flash', 'gcap']
                a = np.minimum(1.0,a)
                a = np.maximum(0.0,a)
                return a
            ds.add_field('foam', function=_foam, units="")

    if all(x in species for x in ['cham', 'targ', 'mark', 'ablt', ]):
        if ('flash', 'cham') not in ds.field_list:
            def _cham(field, data):
                a = 1.0 - data['flash', 'targ'] - data['flash', 'mark'] \
                        - data['flash', 'ablt']
                a = np.minimum(1.0,a)
                a = np.maximum(0.0,a)
                return a
            ds.add_field('cham', function=_cham, units="")

    def _z1bar(field, data):
        return data['abar'] * np.sum([data[var]*mat_data[mat]['zbar']/mat_data[mat]['abar'] for var, mat in species.items()], axis=0) 
    ds.add_field('z1bar', function=_z1bar, units="")

    if usePlasma:
        def _stop_nele(field, data):
            try: return np.array(data['flash', 'nele'])*data.ds.quan(1.0, '1/cm**3')
            except: 
                Na_code = data.ds.quan(Na, 'code_length**3/code_mass/cm**3')
                try: return data['flash','dens']*data['flash', 'ye']*Na_code
                except: return data['flash','dens']*data['flash', 'ye  ']*Na_code
        ds.add_field('stop_nele', function=_stop_nele, units='1/cm**3')
        def _cfrac(field, data):
            return 1.0/(1.0 + mat_data['coef_shield'] * \
                    np.exp(np.sum([data[var]*mat_data[mat]['logzbar'] for var, mat in species.items()], axis=0)*2.0/3.0+tiny) * \
                    data['flash', 'tele']/data['stop_nele'])
        ds.add_field('cfrac', function=_cfrac, units='')
        def _pfrac(field, data):
            return 1.0 - data['cfrac']
        ds.add_field('pfrac', function=_pfrac, units='')

    if useMSC:
        def _msc_a1(field, data):
            _data = np.exp(data['abar'] * np.sum([data[var]*mat_data[mat]['msc_a1'] for var, mat in species.items()], axis=0)+tiny)+tiny
            if usePlasma:
                _data = _data * data['cfrac']
            return _data
        def _msc_a22(field, data):
            return data['dens'] * np.sum([data[var]*mat_data[mat]['msc_a22'] for var, mat in species.items()], axis=0)
        def _msc_zbar(field, data):
            return (data['abar'] * np.sum([data[var]*mat_data[mat]['z2bar']/(mat_data[mat]['abar']+tiny)  \
                    for var, mat in species.items()], axis=0))**0.5
    else:
        def _msc_a1(field, data):
            return np.zeros_like(data['dens'])
        def _msc_a22(field, data):
            return np.zeros_like(data['dens'])*data.ds.quan(1.0, 'code_mass/code_length**3')
        def _msc_zbar(field, data):
            return np.zeros_like(data['dens'])
    def _msc_w(field, data):
        return data['dens']/(data['abar']+tiny)
    ds.add_field('msc_zbar', function=_msc_zbar, units='')
    ds.add_field('msc_a1', function=_msc_a1, units='')
    ds.add_field('msc_a22', function=_msc_a22, units='code_mass/code_length**3')
    ds.add_field('msc_w', function=_msc_w, units='code_mass/code_length**3')

    def _stop_a(field, data):
        _data = data['abar']/data['z1bar'] * \
                np.sum([data[var]*mat_data[mat]['a_cold']*mat_data[mat]['zbar']/mat_data[mat]['abar'] for var, mat in species.items()], axis=0)
        if usePlasma:
            _data1 = data['abar']/data['z1bar'] * \
                np.sum([data[var]*mat_data[mat]['a_plasma']*mat_data[mat]['zbar']/mat_data[mat]['abar'] for var, mat in species.items()], axis=0) - \
                0.5*np.log(data['dens'])
            _data = (data['pfrac'] * _data1 + data['cfrac'] * _data)
        return _data
    def _stop_b(field, data):
        _data = data['dens']*np.sum([data[var]*mat_data[mat]['stop_b'] for var, mat in species.items()], axis=0) 
        return _data
    def _stop_strag(field, data):
        return mat_data['coef_straggling']*data['dens']*data['z1bar']/(data['abar']+tiny)
    ds.add_field('stop_a', function=_stop_a, units='')
    ds.add_field('stop_b', function=_stop_b, units='code_mass/code_length**3')
    ds.add_field('stop_strag', function=_stop_strag, units='code_mass/code_length**3')

    try: cg = ds.covering_grid(level=ds.max_level-level_reduction, left_edge=ds.domain_left_edge, \
            fields=['dens', ],
            dims=ds.domain_dimensions*2**(ds.max_level-level_reduction))
    except: cg = ds.covering_grid(level=ds.max_level-level_reduction, left_edge=ds.domain_left_edge, \
               dims=ds.domain_dimensions*2**(ds.max_level-level_reduction)-np.array([1,1,geodim[1]-2]))

    mag_var_list = ['magx', 'magy', 'magz']
    msc_stop_var_list = ['msc_zbar', 'msc_a1', 'msc_a22', 'stop_a', 'stop_b', 'stop_strag']
    if msc_w: msc_stop_var_list.append('msc_w')
    if geodim == ('cartesian', 3):
        data = dict([(var, cg[var].base) for var in mag_var_list + msc_stop_var_list])
        data['leftEdge']  = cg.left_edge
        data['rightEdge'] = cg.right_edge
        return data
    elif geodim == ('cylindrical', 2):
        mag3d = field_2dcyl_to_3dcart_vector([np.array(cg[var][:,:,0]) for var in mag_var_list])
        data = {'magx': mag3d[0], 'magy': mag3d[1], 'magz': mag3d[2]}
        for var in msc_stop_var_list:
            data[var] = field_2dcyl_to_3dcart_scalor(np.array(cg[var][:,:,0]))
        cyl_left = cg.left_edge
        cyl_right = cg.right_edge
        cyl_r = cyl_right[0]
        cyl_zl = cyl_left[1]
        cyl_zr = cyl_right[1]
        data['leftEdge'] =  np.array([-cyl_r, -cyl_r, cyl_zl])
        data['rightEdge'] = np.array([cyl_r, cyl_r, cyl_zr])
        return data
    elif geodim == ('cartesian', 2):
        cart_left = cg.left_edge
        cart_right = cg.right_edge
        cart_xl = cart_left[0]
        cart_xr = cart_right[0]
        cart_yl = cart_left[1]
        cart_yr = cart_right[1]
        cart_zl = -0.5*cart2d_to_3d_length
        cart_zr = 0.5*cart2d_to_3d_length
        data = {'leftEdge': np.array([cart_xl, cart_yl, cart_zl]), 'rightEdge': np.array([cart_xr, cart_yr, cart_zr])}
        ## for var in mag_var_list + msc_stop_var_list:
        ##     data[var] = field_2dcart_to_3d(np.array(cg[var][:,:,0]), cart2d_to_3d_length/(cart_xr-cart_xl))

        for var in ['magz'] + msc_stop_var_list:
             data[var] = field_2dcart_to_3d(np.array(cg[var][:,:,0]), cart2d_to_3d_length/(cart_xr-cart_xl))
        data['magx'] = np.zeros_like(data['magz'])
        data['magy'] = np.zeros_like(data['magz'])

        return data
    else: 
        print("Unsupported geometry, dimensionality!")
        exit(1)

def proton_sim(input_list, linear=False):
    input_fn = ""
    for setup in input_list:

        args_data = {}
        for arg in setup[2]:
            args_data.update(arg)

        if input_fn != setup[0]:
            if comm.rank == 0: tstart = time()
            input_fn = setup[0]
            try: del args_sim
            except: pass

            if linear: args_sim = fields_from_flash(input_fn, **args_data, msc_w=True)
            else: args_sim = fields_from_flash(input_fn, **args_data)

            if comm.rank == 0:
                print("{0} : {1}".format(input_fn, time()-tstart))
    
        output_fn = setup[1]
    
        for arg in setup[3]:
            args_sim.update(arg)
    
        if comm.rank == 0: 
            tstart = time()

        if linear: result = proton_core_linear(**args_sim)
        else: result = proton_core(**args_sim)

        if comm.rank ==0:
            ##np.savez_compressed(output_fn, data=result)
            np.savez_compressed(output_fn, **result)
            tend = time()
            print("{0} > {1} : {2}".format(input_fn, output_fn, tend-tstart))


def proton_core_linear(\
        double [::1] leftEdge, double [::1] rightEdge, \
        double [:,:,:] magx, double [:,:,:] magy, double [:,:,:] magz, \
        double [:,:,:] msc_zbar, double [:,:,:] msc_a1, double [:,:,:] msc_a22, double[:,:,:] msc_w, \
        double [:,:,:] stop_a, double [:,:,:] stop_b, double [:,:,:] stop_strag, \
        double protonEnergyInit, axis, int countAllProcs=0, \
        ):
    # Constants
    cdef double tiny = np.finfo(np.float64).tiny

    # MPI variables
    cdef int rank = comm.rank
    cdef int ntasks = comm.ntasks

    cdef double protonRestEnergy, alpha2, dx, dy, dz, dt
    cdef int nx, ny, nz, axis_id, ix, iy, iz, jx, jy, jz, jz0, jz1, jx_max, jy_max, dir_sign

    cdef double pc_2, pc, pv, pv_2, beta_inv2, pc_by_e_inv, protonEnergyLostRate
    cdef double msc_chi, msc_chi_min, p_left, mb_b, mb_B, msc_rms_chi
    cdef int mb_i
    cdef double mag_coef[6]

    cdef double [:,::1] data_thetax
    cdef double [:,::1] data_thetay
    cdef double [:,::1] data_msc_chi
    cdef double [:,::1] data_msc_chi_min
    cdef double [:,::1] data_msc_log_chi_min
    cdef double [:,::1] data_msc_w
    cdef double [:,::1] data_msc_rms_chi
    cdef double [:,::1] data_msc_B
    cdef double [:,::1] data_energy
    cdef double [:,::1] data_delta_energy

    nx, ny, nz = magx.base.shape
    dx, dy, dz = (rightEdge.base-leftEdge.base)/magx.base.shape
    protonRestEnergy = float((p.mass_hydrogen*p.speed_of_light**2/u.MeV).in_cgs())
    alpha2 = float((p.elementary_charge**2 / p.hbar/ p.speed_of_light).in_cgs())**2

    if axis=='+x':
        axis_id, dir_sign = 0, 1
        jz0, jz1 = 0, nx-1
    if axis=='-x':
        axis_id,dir_sign = 0, -1
        jz0, jz1 = nx-1, 0
    if axis=='+y':
        axis_id, dir_sign = 1, 1
        jz0, jz1 = 0, ny-1
    if axis=='-y':
        axis_id, dir_sign = 1, -1
        jz0, jz1 = ny-1, 0
    if axis=='+z':
        axis_id, dir_sign = 2, 1
        jz0, jz1 = 0, nz-1
    if axis=='-z':
        axis_id, dir_sign = 2, -1
        jz0, jz1 = nz-1, 0
    if axis_id==0:
        data_shape = (ny, nz)
        dt = dx
        mag_coef = (0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
    if axis_id==1:
        data_shape = (nz, nx)
        dt = dy
        mag_coef = (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    if axis_id==2:
        data_shape = (nx, ny)
        dt = dz
        mag_coef = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0)
    jx_max, jy_max = data_shape

    data_thetax = np.zeros(data_shape, dtype=np.float64)
    data_thetay = np.zeros(data_shape, dtype=np.float64)
    data_msc_chi = np.zeros(data_shape, dtype=np.float64)
    data_msc_chi_min = np.zeros(data_shape, dtype=np.float64)
    data_msc_log_chi_min = np.zeros(data_shape, dtype=np.float64)
    data_msc_w = np.zeros(data_shape, dtype=np.float64)
    data_msc_rms_chi = np.zeros(data_shape, dtype=np.float64)
    data_msc_B = np.zeros(data_shape, dtype=np.float64)
    data_energy = np.zeros(data_shape, dtype=np.float64)
    data_delta_energy = np.zeros(data_shape, dtype=np.float64)

    with nogil, parallel():
        for jx in prange(rank, jx_max, ntasks):
            if axis_id == 0: iy = jx
            if axis_id == 1: iz = jx
            if axis_id == 2: ix = jx
            for jy in range(jy_max):
                if axis_id == 0: iz = jy
                if axis_id == 1: ix = jy
                if axis_id == 2: iy = jy
                data_energy[jx, jy] = protonEnergyInit
                data_delta_energy[jx, jy] = 0.0
                jz = jz0
                while jz0<=jz<=jz1 or jz1<=jz<=jz0:
                    if axis_id == 0: ix = jz
                    if axis_id == 1: iy = jz
                    if axis_id == 2: iz = jz
                    if data_energy[jx, jy] <= 0.0: 
                        data_thetax[jx, jy] = 0.0
                        data_thetay[jx, jy] = 0.0
                        data_energy[jx, jy] = 0.0
                        break
                    pc_2 = data_energy[jx, jy]*(data_energy[jx, jy]+2.0*protonRestEnergy)
                    pc = sqrt(pc_2)
                    pv = data_energy[jx, jy]*(data_energy[jx, jy]+2.0*protonRestEnergy)/(data_energy[jx, jy]+protonRestEnergy)
                    pv_2 = pv**2
                    beta_inv2 = pc_2/pv_2
                    pc_by_e_inv = 0.00106273683742/ pc
                    data_thetax[jx, jy] += dir_sign * pc_by_e_inv * dt * \
                            (mag_coef[0]*magx[ix, iy, iz] + mag_coef[1]*magy[ix, iy, iz] + mag_coef[2]*magz[ix, iy, iz])
                    data_thetay[jx, jy] += dir_sign * pc_by_e_inv * dt * \
                            (mag_coef[3]*magx[ix, iy, iz] + mag_coef[4]*magy[ix, iy, iz] + mag_coef[5]*magz[ix, iy, iz])
                    data_msc_chi[jx, jy] += dt*msc_a22[ix,iy,iz]/pv_2
                    data_msc_log_chi_min[jx, jy] += msc_w[ix, iy, iz] * \
                            log(msc_a1[ix, iy, iz]/pc_2*(1.13+3.76*msc_zbar[ix, iy, iz]**2*alpha2*beta_inv2)+tiny)
                    data_msc_w[jx, jy] += msc_w[ix, iy, iz]
                    msc_chi = sqrt(data_msc_chi[jx, jy])
                    msc_chi_min = sqrt(exp(data_msc_log_chi_min[jx, jy] / data_msc_w[jx, jy]))
                    p_left = (msc_chi/msc_chi_min)**2
                    mb_b = log(2.32929034*p_left)
                    if mb_b > 1.0:
                        mb_B = mb_b
                        for mb_i in range(14): mb_B = mb_b + log(mb_B)
                    else: mb_B = 0.0
                    data_msc_rms_chi[jx, jy] = msc_chi * sqrt(mb_B)
                    protonEnergyLostRate = stop_b[ix, iy, iz] * (log(1.0/(beta_inv2-1.0)) + stop_a[ix, iy, iz]) * beta_inv2
                    protonEnergyLostRate = max(0.0, protonEnergyLostRate) * dt
                    data_energy[jx, jy] -= protonEnergyLostRate * (1.0 + 0.5*data_msc_rms_chi[jx, jy]**2)
                    data_delta_energy[jx, jy] += stop_strag[ix,iy,iz]*dt*(beta_inv2-0.5)/(beta_inv2-1.0)
                    jz = jz + dir_sign
                data_msc_B[jx, jy] = mb_B
                data_msc_rms_chi[jx, jy] = data_msc_rms_chi[jx, jy] / sqrt(2.0)
                data_msc_chi[jx, jy] = msc_chi
                data_msc_chi_min[jx, jy] = msc_chi_min
                data_delta_energy[jx, jy] = sqrt(data_delta_energy[jx, jy])

    if countAllProcs != 0:
        total_data_thetax = np.empty(data_shape, dtype=np.float64)
        total_data_thetay = np.empty(data_shape, dtype=np.float64)
        total_data_msc_rms_chi = np.empty(data_shape, dtype=np.float64)
        total_data_msc_chi = np.empty(data_shape, dtype=np.float64)
        total_data_msc_chi_min = np.empty(data_shape, dtype=np.float64)
        total_data_msc_B = np.empty(data_shape, dtype=np.float64)
        total_data_energy = np.empty(data_shape, dtype=np.float64)
        total_data_delta_energy = np.empty(data_shape, dtype=np.float64)
        comm.comm.Allreduce(data_thetax, total_data_thetax, op=MPI.SUM)
        comm.comm.Allreduce(data_thetay, total_data_thetay, op=MPI.SUM)
        comm.comm.Allreduce(data_msc_rms_chi, total_data_msc_rms_chi, op=MPI.SUM)
        comm.comm.Allreduce(data_msc_B, total_data_msc_B, op=MPI.SUM)
        comm.comm.Allreduce(data_msc_chi, total_data_msc_chi, op=MPI.SUM)
        comm.comm.Allreduce(data_msc_chi_min, total_data_msc_chi_min, op=MPI.SUM)
        comm.comm.Allreduce(data_energy, total_data_energy, op=MPI.SUM)
        comm.comm.Allreduce(data_delta_energy, total_data_delta_energy, op=MPI.SUM)
    else:
        total_data_thetax = np.empty(data_shape, dtype=np.float64) if rank==0 else None
        total_data_thetay = np.empty(data_shape, dtype=np.float64) if rank==0 else None
        total_data_msc_rms_chi = np.empty(data_shape, dtype=np.float64) if rank==0 else None
        total_data_msc_chi = np.empty(data_shape, dtype=np.float64) if rank==0 else None
        total_data_msc_chi_min = np.empty(data_shape, dtype=np.float64) if rank==0 else None
        total_data_msc_B = np.empty(data_shape, dtype=np.float64) if rank==0 else None
        total_data_energy = np.empty(data_shape, dtype=np.float64) if rank==0 else None
        total_data_delta_energy = np.empty(data_shape, dtype=np.float64) if rank==0 else None
        comm.comm.Reduce(data_thetax, total_data_thetax, op=MPI.SUM, root=0)
        comm.comm.Reduce(data_thetay, total_data_thetay, op=MPI.SUM, root=0)
        comm.comm.Reduce(data_msc_rms_chi, total_data_msc_rms_chi, op=MPI.SUM, root=0)
        comm.comm.Reduce(data_msc_B, total_data_msc_B, op=MPI.SUM, root=0)
        comm.comm.Reduce(data_msc_chi, total_data_msc_chi, op=MPI.SUM, root=0)
        comm.comm.Reduce(data_msc_chi_min, total_data_msc_chi_min, op=MPI.SUM, root=0)
        comm.comm.Reduce(data_energy, total_data_energy, op=MPI.SUM, root=0)
        comm.comm.Reduce(data_delta_energy, total_data_delta_energy, op=MPI.SUM, root=0)
    return {'thetax': total_data_thetax, 'thetay': total_data_thetay, \
            'msc_rms_chi': total_data_msc_rms_chi, 'msc_B': total_data_msc_B, \
            'msc_chi': total_data_msc_chi, 'msc_chi_min': total_data_msc_chi_min, \
            'energy': total_data_energy, 'delta_energy': total_data_delta_energy}



def proton_core(\
        double [::1] leftEdge, double [::1] rightEdge, \
        double [:,:,:] magx, double [:,:,:] magy, double [:,:,:] magz, \
        double [:,:,:] msc_zbar, double [:,:,:] msc_a1, double [:,:,:] msc_a22, \
        double [:,:,:] stop_a, double [:,:,:] stop_b, double [:,:,:] stop_strag, \
        double protonEnergyInit, double protonEnergyWidth, double [:,:] protonEnergyEnd, int protonNumber, \
        double protonSourceType, \
        # protonEnergyWidth is FWHM for D3He and DD protons
        # protonEnergyWidth is e-fold width for TNSA protons
        # protonSourceType is 0 or default for D3He and DD protons with trigangle distribution
        # protonSourceType is 1 or for TNSA protons on OMEGA EP
        # protonSourceType is 2 for D3He and DD protons with gaussian distribution
        double [::1] protonCenter, double protonAngle, double protonSize, \
        double [::1] plateCenter, double [::1] plateX, double [::1] plateY, \
        double [:,::1] plateRange, int binX, int binY, \
        double dtLimitLow, double dtLimitHigh, double dvLimit, double msc_p1=0.05, double msc_p2=50.0, \
        double [:] multimag=np.array([1.0,1.0,1.0]), double multidens=1.0, \
        int countAllProcs=0, \
        int progress_display=0,\
        plateMatrixInput=None,  rcfDelta0Input=None, rcfDelta1Input=None, \
        int split_r_type=0, double split_r_value=1.0, \
        #if split_r_type=1, no B field in the cylinder, if 2, no B field outside the cylinder
        int split_z_type=0, double split_z1_value=0.0, double split_z2_value=0.0, \
        #if split_z_type=1, no B field for z < split_z1_value or z > split_z2_value
        int split_x_type=0, double split_x1_value=0.0, double split_x2_value=0.0, \
        #if split_x_type=1, no B field for x < split_x1_value or x > split_x2_value
        int split_box_type=0, double [:] split_box_center=np.array([0.0,0.0,0.0]), \
        double [:] split_box_halfwidth=np.array([0.0,0.0,0.0]), \
        #if split_x_type=1, only include field in the box
        int em_analysical=0, double [:,:] em_parset=np.zeros((1,1)), \
        int use_pps=0, \
        double [::1] pps_center=np.array([]), double [::1] pps_x=np.array([]), double [::1] pps_y=np.array([]), \
        double [:,::1] pps_list_circ=np.array([[]]),\
        double [:,::1] pps_list1_poly=np.array([[]]), \
        double [:,::1] pps_list2_poly=np.array([[]]), \
        double [:,::1] pps_list3_poly=np.array([[]]), \
        double pps_dE1=0.0, double pps_dE2=0.0, double pps_chi=0.0, double pps_chi_min=1.0e-6, \
        # use_pps=1 is for use PPS
        # pps_center: center of pps, pps_x: x axis of pps, pps_y: y axis of pps
        # pps_list_circ: list of pps (xi, yi, ri),  pps_list_poly: list of pps (xi, yi)
        # pps_E1, pps_E2: through PPS the energy lost is gussian(E1, E2)
        # pps_chi, pps_chi_min: scattering parameters
        int use_grid1=0, double grid1_lambda_x=1.0, double grid1_lambda_y=1.0, double grid1_xbar=0.5, double grid1_ybar=0.5, \
        double [::1] grid1_center=np.array([]), double [::1] grid1_x=np.array([]), double [::1] grid1_y=np.array([]), \
        double grid1_dE1=0.0, double grid1_dE2=0.0, double grid1_chi=0.0, double grid1_chi_min=1.0e-6, double grid1_radius=1.0e99, \
        double [::1] grid1_extend=np.array([-1e99,1e99,-1e99,1e99]), double [::1] grid1_offset=np.array([0.0, 0.0]), \
        # use_grid1=1 is for use grid1
        # grid1_lambda_x: wavelength of grid1 in x direction, grid1_lambda_y: wavelength of grid1 in y direction
        # grid1_xbar: length of bar in x direction, grid1_ybar: length of bar in y direction
        # grid1_center: center of grid1, grid1_x: x axis of grid1, grid1_y: y axis of grid1
        # grid1_dE1, grid1_dE2: through grid1 the energy lost is gussian(E1, E2)
        # grid1_chi, grid1_chi_min: scattering parameters
        # grid1_radius: radius of the grid1
        int use_grid2=0, double grid2_lambda_x=1.0, double grid2_lambda_y=1.0, double grid2_xbar=0.5, double grid2_ybar=0.5, \
        double [::1] grid2_center=np.array([]), double [::1] grid2_x=np.array([]), double [::1] grid2_y=np.array([]), \
        double grid2_dE1=0.0, double grid2_dE2=0.0, double grid2_chi=0.0, double grid2_chi_min=1.0e-6, double grid2_radius=1.0e99, \
        double [::1] grid2_extend=np.array([-1e99,1e99,-1e99,1e99]), double [::1] grid2_offset=np.array([0.0, 0.0]), \
        # Second grid grid2
        int track_n=0, int track_max_step=1024, int track_leap_n=50, \
        int hit_2d_n=0, int hit_3d_n=0, \
        int protonSourceLinear=0, double [::1] protonSourceLinearDirection=np.array([0.0,1.0,0.0]), \
        # if protonSourceLinear is true, then use protonSourceLinearDirection 
        ):
    # Constants
    cdef double tiny = np.finfo(np.float64).tiny
    cdef int one, three

    # MPI variables
    cdef int rank = comm.rank
    cdef int ntasks = comm.ntasks

    # Random number vairables
    cdef unsigned long *stream

    # Thread variables
    cdef int num_threads = comm.num_threads
    cdef int tid

    # Variables
    ##cdef int n, protonNumberMy, \
    cdef int n, \
            i, j, k, \
            det_i, det_j, det_k, \
            t1Axis, t1Limit, t2Axis, t2Limit, axis, limit, \
            vec_i, \
            positiveT, \
            posTargetOk, protonOk, \
            protonEnergyEnd_n 
    cdef double sinHalfTheta, \
            x1, y1, \
            forceAbs, vel0AbsInv, dt, \
            pc, pc_2, pc_by_e_inv, pv, pv_2, pv_by_e_inv, \
            protonEnergy, protonEnergyEndLowLimit, protonEnergyEndHighLimit, \
            protonEnergyLostRate, protonRestEnergy, \
            protonEnergyLostOmega, \
            msc_chi, msc_chi_min, alpha2, beta_inv2, \
            efield_dot_v, rcf_temp1
    cdef double idxm, idym, idzm, idxp, idyp
    cdef double *rpos 
    cdef double *rvel
    cdef double *rener
    cdef double *pos0
    cdef double *vel0
    cdef double *b_field
    cdef double *e_field
    cdef double *force
    cdef double *posTarget
    cdef double veln[3]
    cdef double velp1[3]
    cdef double velp2[3]
    cdef double **t
    cdef double **matrixA
    cdef int *ipiv
    cdef double [:,:,:,:] protonCount 
    cdef unsigned short [:,:] plateMatrix
    cdef double [:] rcfDelta0
    cdef double [:] rcfDelta1
    cdef int *progress_percent
    cdef unsigned long [:] rng_seeds
    cdef double [:,:,:] track_data
    cdef unsigned short [:] track_data_step
    cdef int track_leap_i
    cdef double [:,:] hit_2d_data
    cdef double [:,:] hit_3d_data
    cdef int msc_dist_data_npt
    cdef double [:] msc_dist_data1
    cdef double [:,:] msc_dist_data2

    cdef int pps_i, pps_ok, pps_in, pps_n1_poly, pps_n2_poly, pps_n3_poly, pps_n_circ
    cdef double pps_cos
    cdef double pps_z[3]

    cdef int grid_ok
    cdef double grid_cos
    cdef double grid1_z[3]
    cdef double grid2_z[3]

    # rng
    rng_seeds = import_seeds()

    # Moliere distribution
    msc_dist_data_npt, msc_dist_data1, msc_dist_data2 = import_msc_dist()

    # Limit of p1, p2
    if msc_p1 > 0.05:
        msc_p1 = 0.05
        if rank==0: print("[Warning] msc_p1 reset to 0.05")
    if msc_p2 < 50.0:
        msc_p2 = 50.0
        if rank==0: print("[Warning] msc_p2 reset to 50.0")

    # Basis of velocity
    _veln = np.subtract(plateCenter, protonCenter)
    np.asarray(<np.double_t [:3]> veln)[:] = _veln
    vec_norm_perp(veln, velp1, velp2)

    # Mininum energy, Constants
    protonEnergyEndLowLimit = np.min(protonEnergyEnd)
    protonEnergyEndHighLimit= np.max(protonEnergyEnd)
    protonRestEnergy = float((p.mass_hydrogen*p.speed_of_light**2/u.MeV).in_cgs())
    alpha2 = float((p.elementary_charge**2 / p.hbar/ p.speed_of_light).in_cgs())**2
    
    # Array init
    protonEnergyEnd_n = protonEnergyEnd.shape[0]
    protonCount = np.zeros((num_threads,binX,binY,protonEnergyEnd_n), dtype=np.float64)
    plateMatrix = plateMatrixInput \
        if plateMatrixInput is not None and plateMatrixInput.shape == (binX, binY) \
        else np.ones((binX,binY), dtype=np.uint16)
    if protonSourceType == 1:
        rcfDelta0 = rcfDelta0Input \
            if rcfDelta0Input is not None and rcfDelta0Input.shape == (protonEnergyEnd_n,) \
            else np.ones((protonEnergyEnd_n,), dtype=np.float64)
        rcfDelta1 = rcfDelta1Input \
            if rcfDelta1Input is not None and rcfDelta1Input.shape == (protonEnergyEnd_n,) \
            else np.ones((protonEnergyEnd_n,), dtype=np.float64)

    # Track init
    track_data_step = np.zeros(track_n, dtype=np.uint16)
    track_data = np.zeros((track_n, track_max_step, 3), dtype=np.float64)

    # Hit init
    hit_2d_data = np.zeros((hit_2d_n,2), dtype=np.float64)
    hit_3d_data = np.zeros((hit_3d_n,3), dtype=np.float64)

    # PPS init
    if use_pps == 1:
        pps_n_circ = pps_list_circ.shape[0]
        pps_n1_poly = pps_list1_poly.shape[0]
        pps_n2_poly = pps_list2_poly.shape[0]
        pps_n3_poly = pps_list3_poly.shape[0]
        if pps_list_circ.size == 0: pps_n_circ = 0
        if pps_list1_poly.size == 0: pps_n1_poly = 0
        if pps_list2_poly.size == 0: pps_n2_poly = 0
        if pps_list3_poly.size == 0: pps_n3_poly = 0
        if pps_n_circ > 0 or pps_n1_poly > 0 or pps_n2_poly > 0 or pps_n3_poly > 0:
            _pps_z = np.cross(pps_x, pps_y)
            np.asarray(<np.double_t [:3]> pps_z)[:] = _pps_z

    # Grid init
    if use_grid1 == 1:
        _grid1_z = np.cross(grid1_x, grid1_y)
        np.asarray(<np.double_t [:3]> grid1_z)[:] = _grid1_z
    if use_grid2 == 1:
        _grid2_z = np.cross(grid2_x, grid2_y)
        np.asarray(<np.double_t [:3]> grid2_z)[:] = _grid2_z

    with nogil, parallel():
        rpos       = <double *>malloc(sizeof(double)*2)
        rvel       = <double *>malloc(sizeof(double)*3)
        rener      = <double *>malloc(sizeof(double)*1)
        pos0       = <double *>malloc(sizeof(double)*3)
        vel0       = <double *>malloc(sizeof(double)*3)
        b_field    = <double *>malloc(sizeof(double)*3)
        e_field    = <double *>malloc(sizeof(double)*3)
        force      = <double *>malloc(sizeof(double)*3)
        posTarget  = <double *>malloc(sizeof(double)*3)
        t          = <double **> malloc(sizeof(double *)*3)
        t[0]       = <double *>malloc(sizeof(double)*3*2)
        t[1]       = t[0] + 2
        t[2]       = t[0] + 4
        matrixA    = <double **> malloc(sizeof(double *)*3)
        matrixA[0] = <double *>malloc(sizeof(double)*3*3)
        matrixA[1] = matrixA[0]+3
        matrixA[2] = matrixA[0]+6
        ipiv       = <int *>malloc(sizeof(int)*3)
        stream     = <unsigned long*>malloc(sizeof(unsigned long))
        progress_percent = <int *>malloc(sizeof(int))

        # Inverse of magnetic field resolution
        with gil: idxm, idym, idzm = 1.0/ (rightEdge.base-leftEdge.base)*magx.base.shape

        sinHalfTheta = sin(0.5*protonAngle/180.0*pi)
        idxp = 1.0/ (plateRange[0][1]-plateRange[0][0])*binX
        idyp = 1.0/ (plateRange[1][1]-plateRange[1][0])*binY
        
        tid = openmp.omp_get_thread_num()
        one = 1
        three = 3

        rand_init(stream, rank*num_threads+tid, num_threads*ntasks, rng_seeds)

        progress_percent[0]=-1

        for n in prange(rank, protonNumber, ntasks, schedule='guided', chunksize=10):
            if progress_display>0 and <int>(n*100.0/protonNumber/progress_display) > progress_percent[0]:
                progress_percent[0]=<int>(n*100.0/protonNumber/progress_display)
                printf("R %d T %d: %d % \n" , rank, tid, progress_percent[0]*progress_display)
            protonOk = 0 # Need to continue
            while protonOk == 0:
                # Energy distribution
                if protonSourceType == 1 or protonSourceType == 3:
                    rand_exp(stream,  rener, 1, protonEnergyInit, protonEnergyWidth)
                elif protonSourceType == 2:
                    rand_gaussian(stream, rener, 1, protonEnergyWidth)
                    rener[0] = rener[0] + protonEnergyInit
                elif protonSourceType == 4:
                    rand_trig(stream, rener, 1, protonEnergyInit, protonEnergyWidth)
                else: 
                    printf("Unsupported proton source type!\n")
                    with gil: exit(1)
                protonEnergy = rener[0]
                if  protonEnergy < protonEnergyEndLowLimit: continue

                # spatial distribution and velocity angle distribution
                rand_angle(stream, rvel,  sinHalfTheta)
                rand_gaussian(stream, rpos, 3, protonSize)
                for vec_i in range(3): pos0[vec_i] = protonCenter[vec_i] + rpos[vec_i]
                if leftEdge[0] < pos0[0] < rightEdge[0] and \
                   leftEdge[1] < pos0[1] < rightEdge[1] and \
                   leftEdge[2] < pos0[2] < rightEdge[2]:
                     continue
                if protonSourceLinear==1:
                    for vec_i in range(3):
                        vel0[vec_i] = copysign(sqrt(rvel[0]**2+rvel[1]**2),rvel[0]) * protonSourceLinearDirection[vec_i] + \
                                rvel[2]*veln[vec_i]
                else:
                    for vec_i in range(3):
                        vel0[vec_i] = rvel[0]*velp1[vec_i] + rvel[1]*velp2[vec_i] + rvel[2]*veln[vec_i]

                # Apply PPS
                if use_pps == 1:
                    for vec_i in range(3): posTarget[vec_i] = pos0[vec_i] - pps_center[vec_i]
                    for vec_i in range(3): matrixA[0][vec_i] = pps_x[vec_i]
                    for vec_i in range(3): matrixA[1][vec_i] = pps_y[vec_i]
                    for vec_i in range(3): matrixA[2][vec_i] = -vel0[vec_i]
                    dgesv(&three, &one, matrixA[0], &three, ipiv, posTarget, &three, &posTargetOk)
                    pps_ok = 0
                    if pps_n1_poly > 0 or pps_n2_poly > 0 or pps_n3_poly > 0:
                        pps_ok = 1
                        if pps_n1_poly > 0:
                            pps_in = 1
                            for pps_i in range(pps_n1_poly):
                                if (pps_list1_poly[pps_i][0]-posTarget[0])*(pps_list1_poly[(pps_i+1)%pps_n1_poly][1]-pps_list1_poly[pps_i][1]) - \
                                   (pps_list1_poly[pps_i][1]-posTarget[1])*(pps_list1_poly[(pps_i+1)%pps_n1_poly][0]-pps_list1_poly[pps_i][0]) > 0:
                                    pps_in = 0
                            if pps_in == 1: pps_ok=0
                        if pps_n2_poly > 0:
                            pps_in = 1
                            for pps_i in range(pps_n2_poly):
                                if (pps_list2_poly[pps_i][0]-posTarget[0])*(pps_list2_poly[(pps_i+1)%pps_n2_poly][1]-pps_list2_poly[pps_i][1]) - \
                                   (pps_list2_poly[pps_i][1]-posTarget[1])*(pps_list2_poly[(pps_i+1)%pps_n2_poly][0]-pps_list2_poly[pps_i][0]) > 0:
                                    pps_in = 0
                            if pps_in == 1: pps_ok=0
                        if pps_n3_poly > 0:
                            pps_in = 1
                            for pps_i in range(pps_n3_poly):
                                if (pps_list3_poly[pps_i][0]-posTarget[0])*(pps_list3_poly[(pps_i+1)%pps_n3_poly][1]-pps_list3_poly[pps_i][1]) - \
                                   (pps_list3_poly[pps_i][1]-posTarget[1])*(pps_list3_poly[(pps_i+1)%pps_n3_poly][0]-pps_list3_poly[pps_i][0]) > 0:
                                    pps_in = 0
                            if pps_in == 1: pps_ok=0
                    for pps_i in range(pps_n_circ):
                        if (posTarget[0]-pps_list_circ[pps_i][0])**2+(posTarget[1]-pps_list_circ[pps_i][1])**2 < \
                                pps_list_circ[pps_i][2]**2:
                            pps_ok = 1
                    if pps_ok == 0: 
                        pps_cos = fabs(vel0[0] * pps_z[0] + vel0[1] * pps_z[1] + vel0[2] * pps_z[2])
                        rand_gaussian(stream, rener, 1, pps_dE2)
                        protonEnergy = protonEnergy - (pps_dE1 + rener[0])/pps_cos
                        if  protonEnergy < protonEnergyEndLowLimit: continue
                        rand_coulumb(stream, vel0, pps_chi/sqrt(pps_cos), pps_chi_min, msc_p1, msc_p2, \
                                msc_dist_data_npt, msc_dist_data1, msc_dist_data2)
                        for vec_i in range(3):
                            pos0[vec_i] = pps_center[vec_i] + pps_x[vec_i] * posTarget[0] \
                                                            + pps_y[vec_i] * posTarget[1] \
                                                            - vel0[vec_i] * posTarget[2] 

                # Apply Grid
                if use_grid1 == 1:
                    for vec_i in range(3): posTarget[vec_i] = pos0[vec_i] - grid1_center[vec_i]
                    for vec_i in range(3): matrixA[0][vec_i] = grid1_x[vec_i]
                    for vec_i in range(3): matrixA[1][vec_i] = grid1_y[vec_i]
                    for vec_i in range(3): matrixA[2][vec_i] = -vel0[vec_i]
                    dgesv(&three, &one, matrixA[0], &three, ipiv, posTarget, &three, &posTargetOk)
                    grid_ok = 1
                    posTarget[0] = posTarget[0] - grid1_offset[0]
                    posTarget[1] = posTarget[0] - grid1_offset[0]
                    if posTarget[0] - grid1_lambda_x * floor(posTarget[0] / grid1_lambda_x) < grid1_xbar: grid_ok = 0
                    if posTarget[1] - grid1_lambda_y * floor(posTarget[1] / grid1_lambda_y) < grid1_ybar: grid_ok = 0
                    posTarget[0] = posTarget[0] + grid1_offset[0]
                    posTarget[1] = posTarget[0] + grid1_offset[0]
                    if posTarget[0]**2 + posTarget[1]**2 > grid1_radius**2: grid_ok = 1
                    if posTarget[0] < grid1_extend[0] or posTarget[0] > grid1_extend[1] or \
                       posTarget[1] < grid1_extend[2] or posTarget[2] > grid1_extend[3]: grid_ok = 1
                    if grid_ok == 0: 
                        grid_cos = fabs(vel0[0] * grid1_z[0] + vel0[1] * grid1_z[1] + vel0[2] * grid1_z[2])
                        rand_gaussian(stream, rener, 1, grid1_dE2)
                        protonEnergy = protonEnergy - (grid1_dE1 + rener[0])/grid_cos
                        if  protonEnergy < protonEnergyEndLowLimit: continue
                        rand_coulumb(stream, vel0, grid1_chi/sqrt(grid_cos), grid1_chi_min, msc_p1, msc_p2, \
                                msc_dist_data_npt, msc_dist_data1, msc_dist_data2)
                        for vec_i in range(3):
                            pos0[vec_i] = grid1_center[vec_i] + grid1_x[vec_i] * posTarget[0] + grid1_y[vec_i] * posTarget[1] \
                                                             - vel0[vec_i] * posTarget[2] 

                # Apply Grid 2
                if use_grid2 == 1:
                    for vec_i in range(3): posTarget[vec_i] = pos0[vec_i] - grid2_center[vec_i]
                    for vec_i in range(3): matrixA[0][vec_i] = grid2_x[vec_i]
                    for vec_i in range(3): matrixA[1][vec_i] = grid2_y[vec_i]
                    for vec_i in range(3): matrixA[2][vec_i] = -vel0[vec_i]
                    dgesv(&three, &one, matrixA[0], &three, ipiv, posTarget, &three, &posTargetOk)
                    grid_ok = 1
                    posTarget[0] = posTarget[0] - grid2_offset[0]
                    posTarget[1] = posTarget[0] - grid2_offset[0]
                    if posTarget[0] - grid2_lambda_x * floor(posTarget[0] / grid2_lambda_x) < grid2_xbar: grid_ok = 0
                    if posTarget[1] - grid2_lambda_y * floor(posTarget[1] / grid2_lambda_y) < grid2_ybar: grid_ok = 0
                    posTarget[0] = posTarget[0] + grid2_offset[0]
                    posTarget[1] = posTarget[0] + grid2_offset[0]
                    if posTarget[0]**2 + posTarget[1]**2 > grid2_radius**2: grid_ok = 1
                    if posTarget[0] < grid2_extend[0] or posTarget[0] > grid2_extend[1] or \
                       posTarget[1] < grid2_extend[2] or posTarget[2] > grid2_extend[3]: grid_ok = 1
                    if grid_ok == 0: 
                        grid_cos = fabs(vel0[0] * grid2_z[0] + vel0[1] * grid2_z[1] + vel0[2] * grid2_z[2])
                        rand_gaussian(stream, rener, 1, grid2_dE2)
                        protonEnergy = protonEnergy - (grid2_dE1 + rener[0])/grid_cos
                        if  protonEnergy < protonEnergyEndLowLimit: continue
                        rand_coulumb(stream, vel0, grid2_chi/sqrt(grid_cos), grid2_chi_min, msc_p1, msc_p2, \
                                msc_dist_data_npt, msc_dist_data1, msc_dist_data2)
                        for vec_i in range(3):
                            pos0[vec_i] = grid2_center[vec_i] + grid2_x[vec_i] * posTarget[0] + grid2_y[vec_i] * posTarget[1] \
                                                             - vel0[vec_i] * posTarget[2] 


                # Track
                if n < track_n:
                    track_data[n,0,0] = pos0[0]
                    track_data[n,0,1] = pos0[1]
                    track_data[n,0,2] = pos0[2]
                    track_data_step[n] = 1
                    track_leap_i = 0

                # Before box
                positiveT = 0
                for axis in range(3):
                    for limit in range(2):
                        t[axis][limit] = (leftEdge[axis]-pos0[axis])/vel0[axis] \
                            if limit == 1 \
                            else (rightEdge[axis]-pos0[axis])/vel0[axis]
                        x1 = pos0[(axis+1)%3] + vel0[(axis+1)%3] * t[axis][limit]
                        y1 = pos0[(axis+2)%3] + vel0[(axis+2)%3] * t[axis][limit]
                        if t[axis][limit]>0.0  and \
                          x1 > leftEdge[(axis+1)%3] and x1 < rightEdge[(axis+1)%3] and \
                          y1 > leftEdge[(axis+2)%3] and y1 < rightEdge[(axis+2)%3] :
                            positiveT = positiveT+1
                            if positiveT == 1:
                                t1Axis = axis
                                t1Limit = limit
                            if positiveT == 2:
                                t2Axis = axis
                                t2Limit = limit
                if positiveT != 2 : protonOk = 1 # Outside the box

                # Motion in the box
                if protonOk == 0 :
                    if t[t1Axis][t1Limit] > t[t2Axis][t2Limit]:
                        t1Axis = t2Axis
                        t1Limit = t2Limit
                    t[t1Axis][t1Limit] = t[t1Axis][t1Limit] + dtLimitLow * 0.5
                    for vec_i in range(3): pos0[vec_i] = pos0[vec_i] + vel0[vec_i] * t[t1Axis][t1Limit] 
                while leftEdge[0] < pos0[0] < rightEdge[0] and \
                      leftEdge[1] < pos0[1] < rightEdge[1] and \
                      leftEdge[2] < pos0[2] < rightEdge[2] and \
                      protonOk == 0 :
                    # Track
                    if n < track_n:
                        if track_data_step[n] < track_max_step:
                            if track_leap_i == 0:
                                track_data[n,track_data_step[n],0] = pos0[0]
                                track_data[n,track_data_step[n],1] = pos0[1]
                                track_data[n,track_data_step[n],2] = pos0[2]
                                track_data_step[n] += 1
                            track_leap_i = (track_leap_i+1)%track_leap_n
                    # Calculate force
                    i = <int>floor((pos0[0]-leftEdge[0])*idxm)
                    j = <int>floor((pos0[1]-leftEdge[1])*idym)
                    k = <int>floor((pos0[2]-leftEdge[2])*idzm)
                    if ((split_r_type==1 and pos0[0]**2+pos0[1]**2<split_r_value**2) \
                            or (split_r_type==2 and pos0[0]**2+pos0[1]**2>split_r_value**2) \
                            or (split_z_type==1 and (pos0[2] < split_z1_value or pos0[2] > split_z2_value)) \
                            or (split_x_type==1 and (pos0[0] < split_x1_value or pos0[0] > split_x2_value)) \
                            or (split_box_type==1 and (fabs(pos0[0]-split_box_center[0]) > split_box_halfwidth[0] \
                                                    or fabs(pos0[1]-split_box_center[1]) > split_box_halfwidth[1] \
                                                    or fabs(pos0[2]-split_box_center[2]) > split_box_halfwidth[2])) \
                            ):
                        b_field[0] = 0.0
                        b_field[1] = 0.0
                        b_field[2] = 0.0
                    else:
                        b_field[0] = multimag[0] * magx[i,j,k]
                        b_field[1] = multimag[1] * magy[i,j,k]
                        b_field[2] = multimag[2] * magz[i,j,k]
                    e_field[0] = 0.0
                    e_field[1] = 0.0
                    e_field[2] = 0.0
                    if em_analysical == 1:
                        em_form(pos0, e_field, b_field, em_parset)
                    if em_analysical == 2:
                        em_form2(pos0, e_field, b_field, em_parset)

                    pc_2 = protonEnergy*(protonEnergy+2.0*protonRestEnergy)
                    pc = sqrt(pc_2)
                    pv = protonEnergy*(protonEnergy+2.0*protonRestEnergy)/(protonEnergy+protonRestEnergy)
                    pv_2 = pv**2
                    beta_inv2 = pc_2/pv_2
                    pc_by_e_inv = 0.00106273683742/ pc
                    pv_by_e_inv = 0.00106273683742/ pv
                    efield_dot_v = e_field[0]*vel0[0] + e_field[1]*vel0[1] + e_field[2]*vel0[2]
                    force[0] = pc_by_e_inv * (b_field[2] * vel0[1] - b_field[1] * vel0[2]) + \
                               pv_by_e_inv * (e_field[0] - efield_dot_v * vel0[0])
                    force[1] = pc_by_e_inv * (b_field[0] * vel0[2] - b_field[2] * vel0[0]) + \
                               pv_by_e_inv * (e_field[1] - efield_dot_v * vel0[1])
                    force[2] = pc_by_e_inv * (b_field[1] * vel0[0] - b_field[0] * vel0[1]) + \
                               pv_by_e_inv * (e_field[2] - efield_dot_v * vel0[2])
                    forceAbs = force[0]**2+force[1]**2+force[2]**2
                    forceAbs = sqrt(forceAbs)
                    # Calculate energy lost rate
                    protonEnergyLostRate = stop_b[i,j,k] * (log(1.0/(beta_inv2-1.0)) + stop_a[i,j,k]) * beta_inv2
                    protonEnergyLostRate = max(0.0, protonEnergyLostRate*multidens)
                    protonEnergyLostRate = protonEnergyLostRate - 0.00106273683742*efield_dot_v
                    # Choose timestep
                    dt = dvLimit/(forceAbs+tiny)
                    dt = fmin(dt, dvLimit/(protonEnergyLostRate+tiny)*protonEnergy)
                    dt = fmin(dt, dtLimitHigh)
                    dt = fmax(dt, dtLimitLow)
                    for vec_i in range(3): vel0[vec_i] = vel0[vec_i] + force[vec_i]*dt
                    # Calculate msc
                    msc_chi = sqrt(dt*msc_a22[i,j,k]*multidens/pv_2)
                    msc_chi_min = sqrt(msc_a1[i,j,k]/pc_2*(1.13+3.76*msc_zbar[i,j,k]**2*alpha2*beta_inv2)+tiny)
                    rand_coulumb(stream, vel0, msc_chi, msc_chi_min, msc_p1, msc_p2, \
                            msc_dist_data_npt, msc_dist_data1, msc_dist_data2)
                    # Calculate straggling
                    protonEnergyLostOmega = sqrt(stop_strag[i,j,k]*dt*(beta_inv2-0.5)/(beta_inv2-1.0))
                    rand_gaussian(stream, rener, 1, protonEnergyLostOmega)
                    # Calculate energy
                    protonEnergy = protonEnergy - protonEnergyLostRate * dt - rener[0]
                    # Particle move
                    for vec_i in range(3): pos0[vec_i] = pos0[vec_i] + vel0[vec_i] * dt
                    if  protonEnergy < protonEnergyEndLowLimit: protonOk = -1
                if protonOk == -1:
                    protonOk = 0
                    continue
                protonOk = 1

                # Motion after leaving box
                if n < track_n:
                    if track_data_step[n] < track_max_step:
                        track_data[n,track_data_step[n],0] = pos0[0]
                        track_data[n,track_data_step[n],1] = pos0[1]
                        track_data[n,track_data_step[n],2] = pos0[2]
                        track_data_step[n] += 1
                for vec_i in range(3): posTarget[vec_i] = pos0[vec_i] - plateCenter[vec_i]
                for vec_i in range(3): matrixA[0][vec_i] = plateX[vec_i]
                for vec_i in range(3): matrixA[1][vec_i] = plateY[vec_i]
                for vec_i in range(3): matrixA[2][vec_i] = -vel0[vec_i]
                dgesv(&three, &one, matrixA[0], &three, ipiv, posTarget, &three, &posTargetOk)

                #! Record the position of proton
                det_i = <int>floor((posTarget[0]-plateRange[0][0])*idxp)
                det_j = <int>floor((posTarget[1]-plateRange[1][0])*idyp)
                protonOk = 0
                protonEnergy = min(protonEnergyEndHighLimit, protonEnergy)
                if det_i<0 or det_j<0 or det_i>=binX or det_j>=binY or posTargetOk!=0 or posTarget[2] < 0:
                    protonOk = 0
                    continue
                if plateMatrix[det_i, det_j] == 0.0:
                    protonOk = 0
                    continue
                else: 
                    for det_k in range(protonEnergyEnd_n):
                        if protonEnergyEnd[det_k,0] < protonEnergy <= protonEnergyEnd[det_k,1]:
                            protonOk = 1
                            if protonSourceType == 1:
                                rcf_temp1 = sqrt((protonEnergy - protonEnergyEnd[det_k,0]) / rcfDelta1[det_k])
                                protonCount[tid, det_i, det_j, det_k] += plateMatrix[det_i, det_j] * \
                                   rcfDelta0[det_k] * (1.0 - exp(-rcf_temp1)) / rcf_temp1
                            elif protonSourceType == 2 or protonSourceType == 3 or protonSourceType == 4:
                                protonCount[tid, det_i, det_j, det_k] += plateMatrix[det_i, det_j] 
                            else: 
                                printf("Unsupported proton source type!\n")
                                with gil: exit(1)
                            # Track
                            if n < track_n:
                                if track_data_step[n] < track_max_step:
                                    track_data[n,track_data_step[n],0] = pos0[0] + vel0[0]*posTarget[2]
                                    track_data[n,track_data_step[n],1] = pos0[1] + vel0[1]*posTarget[2]
                                    track_data[n,track_data_step[n],2] = pos0[2] + vel0[2]*posTarget[2]
                                    track_data_step[n] += 1
                            # Hit
                            if n < hit_2d_n:
                                hit_2d_data[n,0] = posTarget[0]
                                hit_2d_data[n,1] = posTarget[1]
                            if n < hit_3d_n:
                                hit_3d_data[n,0] = pos0[0] + vel0[0]*posTarget[2]
                                hit_3d_data[n,1] = pos0[1] + vel0[1]*posTarget[2]
                                hit_3d_data[n,2] = pos0[2] + vel0[2]*posTarget[2]

        free(stream); free(rpos); free(rvel); free(pos0); free(vel0); 
        free(b_field); free(force); free(posTarget); free(t[0]); free(t); 
        free(matrixA[0]); free(matrixA); free(ipiv); free(rener); free(progress_percent);
        free(e_field)

        for tid in range(1, num_threads):
            for det_i in prange(binX, schedule='dynamic', chunksize=10):
                for det_j in range(binY):
                    for det_k in range(protonEnergyEnd_n):
                        protonCount[0, det_i, det_j, det_k] += protonCount[tid, det_i, det_j, det_k]

    if countAllProcs != 0:
        total = np.empty((binX,binY,protonEnergyEnd_n), dtype=np.float64)
        total_track_data = np.zeros((track_n, track_max_step, 3), dtype=np.float64)
        total_track_data_step = np.zeros(track_n, dtype=np.uint16)
        total_hit_2d_data = np.zeros((hit_2d_n, 2), dtype=np.float64)
        total_hit_3d_data = np.zeros((hit_3d_n, 3), dtype=np.float64)
        comm.comm.Allreduce(protonCount[0,:,:,:], total, op=MPI.SUM)
        comm.comm.Allreduce(track_data, total_track_data, op=MPI.SUM)
        comm.comm.Allreduce(track_data_step, total_track_data_step, op=MPI.SUM)
        comm.comm.Allreduce(hit_2d_data, total_hit_2d_data, op=MPI.SUM)
        comm.comm.Allreduce(hit_3d_data, total_hit_3d_data, op=MPI.SUM)
        ##if total.shape[-1]>1: total = total[:,:,0]
        if protonEnergyEnd_n==1: total = total[:,:,0]
    else:
        total = np.empty((binX,binY,protonEnergyEnd_n), dtype=np.float64) if rank==0 else None
        total_track_data = np.zeros((track_n, track_max_step, 3), dtype=np.float64) if rank==0 else None
        total_track_data_step = np.zeros(track_n, dtype=np.uint16) if rank==0 else None
        total_hit_2d_data = np.zeros((hit_2d_n, 2), dtype=np.float64) if rank==0 else None
        total_hit_3d_data = np.zeros((hit_3d_n, 3), dtype=np.float64) if rank==0 else None
        comm.comm.Reduce(protonCount[0,:,:,:], total, op=MPI.SUM, root=0)
        comm.comm.Reduce(track_data, total_track_data, op=MPI.SUM, root=0)
        comm.comm.Reduce(track_data_step, total_track_data_step, op=MPI.SUM, root=0)
        comm.comm.Reduce(hit_2d_data, total_hit_2d_data, op=MPI.SUM, root=0)
        comm.comm.Reduce(hit_3d_data, total_hit_3d_data, op=MPI.SUM, root=0)
        if rank == 0 and protonEnergyEnd_n==1: total = total[:,:,0]
    if track_n==0 and hit_2d_n==0 and hit_3d_n==0:
        ##return total
        return {'data': total}
    else:
        return {'total': total, \
                'track': total_track_data, 'track_step': total_track_data_step, \
                'hit_2d': total_hit_2d_data, 'hit_3d': total_hit_3d_data, }

def proton_time_integral(sim_list):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        for out_fn, input_list in sim_list:
            image_data_array = np.sum([np.load(u[0])['data']*u[1] for u in input_list], axis=0)
            np.savez_compressed(out_fn, data=image_data_array)
            print("-> {0}".format(out_fn))

