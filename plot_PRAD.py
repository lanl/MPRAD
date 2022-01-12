import numpy as np
from numba import jit
from matplotlib import pyplot as plt
import yt
from mpl_toolkits.axes_grid1 import AxesGrid

@jit(nopython=True, nogil=True, cache=True, parallel=True)
def cal_div(x, y):
    return x / y

@jit(cache=True, parallel=True, nogil=True)
def cal_sum(x, axis):
    return np.sum(x, axis)

@jit(cache=True, parallel=True, nogil=True)
def cal_average(x, weights):
    return np.average(x, weights=weights)

@jit(nopython=True, nogil=True, cache=True, parallel=True)
def cal_square(x):
    return x**2

@jit(nopython=True, nogil=True, cache=True, parallel=True)
def cal_sqrt(x):
    return np.sqrt(x)

@jit(nopython=True, nogil=True, cache=True, parallel=True)
def cal_minus(x, y):
    return x - y

class data_prad():
    def __init__(self, fn, xylim, erange, mag):
        self.data = np.load(fn)['data']
        xmin, xmax = xylim[0]
        ymin, ymax = xylim[1]
        xbin, ybin = self.data.shape[0:2]
        dx, dy = (xmax - xmin) / xbin, (xmax - xmin) / ybin
        if self.data.shape[2] != len(erange)-1:
            raise KeyError('Dimension for energy bin not consistent')
        self.xrange = cal_div(np.linspace(xmin+dx*0.5, xmax-dx*0.5, xbin, endpoint=True), mag/1.0e4)
        self.yrange = cal_div(np.linspace(ymin+dx*0.5, ymax-dx*0.5, ybin, endpoint=True), mag/1.0e4)
        self.erange = np.array(erange)
    def get_data(self, xylim=[[-np.inf,np.inf],[-np.inf,np.inf]], elim=[-np.inf,np.inf], norm=1.0, \
                 vmax_list=None):
        i0_x, i1_x = np.searchsorted(self.xrange, xylim[0][0]), \
                     np.searchsorted(self.xrange, xylim[0][1])
        i0_y, i1_y = np.searchsorted(self.xrange, xylim[1][0]), \
                     np.searchsorted(self.xrange, xylim[1][1])
        i0_e, i1_e = np.searchsorted(self.erange, elim[0]), \
                     np.searchsorted(self.erange, elim[1])
        data_img = cal_div(self.data[i0_x:i1_x, i0_y:i1_y, i0_e:i1_e], norm)
        data_xrange = self.xrange[i0_x:i1_x]
        data_yrange = self.yrange[i0_y:i1_y]
        data_erange = self.erange[i0_e:i1_e+1]
        if vmax_list is None:
            return data_img, data_xrange, data_yrange, data_erange
        else:
            return data_img, data_xrange, data_yrange, data_erange, vmax_list[i0_e:i1_e]
    def get_mo(self, xylim=[[-np.inf,np.inf],[-np.inf,np.inf]], elim=[-np.inf,np.inf], composite=False):
        img, x_grid, y_grid, e_grid = self.get_data(xylim=xylim, elim=elim)
        if composite: 
            print(y_grid, cal_sum(img[:,:,:], axis=(0,2)))
            mo_y1 = cal_average(y_grid, cal_sum(img[:,:,:], axis=(0,2)))
            e_grid = np.array([e_grid[0], e_grid[-1]])
        else: mo_y1 = np.array([cal_average(\
            y_grid, cal_sum(img[:,:,i], axis=0)) for i in range(len(e_grid)-1)])
        return np.array(mo_y1), e_grid
    
def plot_prad(fn_in_base_list, fn_in_range, mag, xylim_in, erange_in, \
              xylim_out, elim_out, count_norm, vmax_list, fn_out, \
              title_doc, runs_doc, nrows, ncols, show=False, 
              Limg=3.0, Lpadx=0.8, Lpady=0.2, Lfont=12, Lmargin=2.0, LfontTitle=20):
    n_panel = len(fn_in_base_list)
    for n_time in fn_in_range:
        t = n_time/10.0
        plt.clf()
        fig = plt.figure(figsize=((Limg+Lpadx)*ncols+2.0*Lmargin,\
                                  (Limg+Lpady)*nrows*n_panel+2.0*Lmargin))
        plt.subplots_adjust(left=Lmargin/((Limg+Lpadx)*ncols+2.0*Lmargin), \
                            right=1.0-Lmargin/((Limg+Lpadx)*ncols+2.0*Lmargin), \
                            bottom=Lmargin/((Limg+Lpady)*nrows*n_panel+2.0*Lmargin), \
                            top=1.0-Lmargin/((Limg+Lpady)*nrows*n_panel+2.0*Lmargin))
        #plt.rc('font', size=Lfont, family="Times New Roman")
        fig.text(0.1, 1.0 - 0.5*Lmargin/((Limg+Lpady)*nrows*2+2.0*Lmargin), \
                 title_doc + '\n' + 'From top to Bottom:     ' + ',    '.join(runs_doc) + \
                 "\n At t={0}ns".format(t), size=LfontTitle)
        for i_panel, fn_in_base in enumerate(fn_in_base_list):
            grid = AxesGrid(fig, int(n_panel*100+10+i_panel+1),
                nrows_ncols=(nrows, ncols), axes_pad=(Lpadx, Lpady), \
                label_mode="L", share_all=False, \
                cbar_location="right", cbar_mode="each", \
                cbar_size="10%", cbar_pad="0%",
                cbar_set_cax=True,)
            ds = data_prad('{0}_{1:04d}.npz'.format(fn_in_base, n_time), xylim_in, erange_in, mag)
            data_img, x_grid, y_grid, erange, vmax_list1 = ds.get_data(xylim_out, elim_out, \
                                                                       count_norm, vmax_list)
            for i in range(len(erange)-1):
                im = grid[i].imshow(data_img[:,:,i], vmax=vmax_list1[i], vmin=0.0, origin='low', \
                                    cmap='bds_highcontrast', extent=(\
                                        y_grid[0]*1.5-y_grid[1]*0.5, \
                                        y_grid[-1]*1.5-y_grid[-2]*0.5, \
                                        x_grid[0]*1.5-x_grid[1]*0.5, \
                                        x_grid[-1]*1.5-x_grid[-2]*0.5),)
                cax = grid.cbar_axes[i]
                cax.colorbar(im)
                cax.toggle_label(True)
                cax.axis[cax.orientation].set_label("{0:.1f}MeV to {1:.1f}MeV".format(erange[i], erange[i+1]))
        if show: plt.show()
        else:
            fn_out_fig = "{0}_{1:04d}.png".format(fn_out, n_time)
            plt.savefig(fn_out_fig)
            print(fn_out_fig)
            
def plot_shift(fn_in_base_list, fn_in_range, mag, xylim_in, erange_in, xylim_out, elim_out, \
               fn_out, title_doc, runs_doc, nrows, ncols, plot_mark, y1_min, y1_max, tmin, tmax, \
               show=False, Limg=3.0, Lpadx=0.3, Lpady=0.5, Lfont=12, Lmargin=2.0, LfontTitle=20):
    data_t = []
    data_y1 = []
    for n_time in fn_in_range:
        data_t.append(n_time/10.0)
        data_y1.append([])
        for fn_in_base in fn_in_base_list:
            ds = data_prad('{0}_{1:04d}.npz'.format(fn_in_base, n_time), xylim_in, erange_in, mag)
            mo_y1, erange = ds.get_mo(xylim=xylim_out, elim=elim_out)
            data_y1[-1].append(mo_y1)
    data_t = np.array(data_t)
    data_y1 = np.array(data_y1)
    
    plt.clf()    
    fig = plt.figure(figsize=((Limg+Lpadx)*ncols+2.0*Lmargin,\
                              (Limg+Lpady)*nrows+2.0*Lmargin))
    plt.subplots_adjust(left=Lmargin/((Limg+Lpadx)*ncols+2.0*Lmargin), \
                        right=1.0-Lmargin/((Limg+Lpadx)*ncols+2.0*Lmargin), \
                        bottom=Lmargin/((Limg+Lpady)*nrows+2.0*Lmargin), \
                        top=1.0-Lmargin/((Limg+Lpady)*nrows+2.0*Lmargin))
    #plt.rc('font', size=Lfont, family="Times New Roman")
    #fig.text(0.1, 1.0 - 0.5*Lmargin/((Limg+Lpady)*nrows*2+2.0*Lmargin), \
    #         title_doc, size=LfontTitle)
    grid = AxesGrid(fig, 111,  # similar to subplot(144)
                    nrows_ncols=(nrows, ncols),
                    axes_pad=(Lpadx, Lpady),
                    label_mode="L",
                    share_all=False)
    for i_run in range(len(fn_in_base_list)):
        grid[0].plot(data_t, np.full_like(data_t, i_run*10.0), plot_mark[i_run], label=runs_doc[i_run])
        grid[0].legend()
        grid[0].set_title(r'${\langle}x{\rangle}$')
    
    for i in range(len(erange)-1):
        for i_run in range(len(fn_in_base_list)):
            grid[i+1].plot(data_t, data_y1[:,i_run,i], plot_mark[i_run])
        grid[i+1].set_title("{0:.1f}MeV to {1:.1f}MeV".format(erange[i], erange[i+1]))
        
    for img in grid:
        img.set_xlim(tmin, tmax)
        img.set_ylim(y1_min, y1_max)
        img.set_aspect((tmax-tmin)/(y1_max-y1_min))
        img.set_xlabel('t(ns)')
        img.set_ylabel(r'shift(${\mu}$m)')
        
    if show: plt.show()
    else:
        fn_out_fig = "{0}.png".format(fn_out)
        plt.savefig(fn_out_fig)
        print(fn_out_fig)