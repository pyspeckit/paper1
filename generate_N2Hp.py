import pyspeckit
from pyspeckit.spectrum.models import n2hp
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const
from astropy.io import fits


from pyspeckit.spectrum.units import SpectroscopicAxis

rest_freq=93173.772e6*u.Hz
T0=(const.h*rest_freq/const.k_B).decompose().value

def J_nu(T, T_nu):
    ''' Brightness temperature calculated in the R-J regime
    '''
    return T_nu/(np.exp(T_nu/T) -1)
#
#
rms_thin=0.001
rms_thick=0.3
#
xarr = SpectroscopicAxis(
            np.linspace(93164661398.45563,93182630148.45563,num=921)*u.Hz,
            refX=rest_freq)
# thin
signal_thin = pyspeckit.models.n2hp.n2hp_vtau.hyperfine(xarr, Tex=9.0, tau=0.01, 
    xoff_v=0.0, width=0.3)
signal_thin += rms_thin*np.random.randn(len(signal_thin))
signal_thin = np.float32(signal_thin)
# thick
signal_thick = pyspeckit.models.n2hp.n2hp_vtau.hyperfine(xarr, Tex=9.0, tau=9.0, 
    xoff_v=0.0, width=0.3)
signal_thick += rms_thick*np.random.randn(len(signal_thick))
signal_thick = np.float32(signal_thick)

# creating it from scratch
sp_thin = pyspeckit.Spectrum(data=signal_thin, xarr=xarr, 
    header={'OBJECT':'Test1', 'TELESCOP':'30m', 'LINE':'NNH+(1-0)',
            'VELO-LSR':0.0, 'RA':0.0, 'dec':0.0,
            'BUNIT':'K', 'SPECSYS':'LSRK', 'VELO-LSR':0.0})
#   header=cube.header)
sp_thin.xarr.refX = rest_freq
sp_thin.xarr.velocity_convention='radio'
sp_thin.xarr.convert_to_unit('m/s')
sp_thin.write('N2Hp_thin.fits')
sp_thin.error[:]=rms_thin

# creating it from scratch
sp_thick = pyspeckit.Spectrum(data=signal_thick, xarr=xarr, 
    header={'OBJECT':'Test2', 'TELESCOP':'30m', 'LINE':'NNH+(1-0)',
            'VELO-LSR':0.0, 'RA':0.0, 'dec':0.0,
            'BUNIT':'K', 'SPECSYS':'LSRK', 'VELO-LSR':0.0})
#   header=cube.header)
sp_thick.xarr.refX = rest_freq
sp_thick.xarr.velocity_convention='radio'
sp_thick.xarr.convert_to_unit('m/s')
sp_thick.write('N2Hp_thick.fits')
sp_thick.error[:]=rms_thick


#########################################
######    Opthically thick model   ######
#########################################
sp_thick.specfit()
sp_thick.Registry.add_fitter('n2hp_vtau', pyspeckit.models.n2hp.n2hp_vtau_fitter,4)
sp_thick.specfit(fittype='n2hp_vtau', guesses=[3.94, 1.0, 0, 0.309], 
    verbose_level=4, signal_cut=3, 
    limitedmax=[False,True,True,True], 
    limitedmin=[True,True,True,True], 
    minpars=[0, 0, -1, 0.05], maxpars=[30.,50.,1,1.0], 
    fixed=[False,False,False,False])
sp_thick.plotter(errstyle='fill')
sp_thick.specfit.plot_fit()
plt.show()
plt.savefig('pyspeckit_N2Hp_thick_4par.png')


#########################################
######    Opthically thin model    ######
#########################################
sp_thin.specfit()
sp_thin.Registry.add_fitter('n2hp_vtau', pyspeckit.models.n2hp.n2hp_vtau_fitter,4)
sp_thin.specfit(fittype='n2hp_vtau', guesses=[3.94, 1.0, 0, 0.309], 
    verbose_level=4, signal_cut=3, 
    limitedmax=[False,True,True,True], 
    limitedmin=[True,True,True,True], 
    minpars=[0, 0, -1, 0.05], maxpars=[30.,50.,1,1.0], 
    fixed=[False,False,False,False])
sp_thin.plotter(errstyle='fill')
sp_thin.specfit.plot_fit()
plt.show()
plt.savefig('pyspeckit_N2Hp_thin_4par.png')

# optically thin case

sp_thin_3par = pyspeckit.Spectrum(data=signal_thin, xarr=xarr, 
    header={'OBJECT':'Test1', 'TELESCOP':'30m', 'LINE':'NNH+(1-0)',
            'VELO-LSR':0.0, 'RA':0.0, 'dec':0.0,
            'BUNIT':'K', 'SPECSYS':'LSRK', 'VELO-LSR':0.0})
sp_thin_3par.xarr.refX = rest_freq
sp_thin_3par.xarr.velocity_convention='radio'
sp_thin_3par.xarr.convert_to_unit('m/s')
sp_thin_3par.error[:]=rms
sp_thin_3par.specfit()
sp_thin_3par.Registry.add_fitter('n2hp_vtau', pyspeckit.models.n2hp.n2hp_vtau_fitter,4)
# model (excitation temperature, optical depth, line center, and line width)
sp_thin_3par.specfit(fittype='n2hp_vtau', guesses=[3.94, 0.1, 0, 0.309], 
    verbose_level=4, signal_cut=3, 
    limitedmax=[False,True,True,True], 
    limitedmin=[True,True,True,True], 
    minpars=[0, 0, -1, 0.05], maxpars=[30.,50.,1,1.0], 
    fixed=[False,True,False,False])

sp_thin_3par.plotter(errstyle='fill')
sp_thin_3par.specfit.plot_fit()
plt.savefig('pyspeckit_N2Hp_thin_3par.png')
plt.show()

# plot comparisons
# optically thin model, without optically thin approximation (4-parameter model)
plt.close()
sp_thin.xarr.convert_to_unit('km/s')
v_arr=sp_thin.xarr
signal_velo=sp_thin.data
Area=6.108E-2
tau=0.487
Tex_CLASS=T0/np.log(1+T0/(Area/tau +J_nu(2.73,T0)))
signal_4par_class = pyspeckit.models.n2hp.n2hp_vtau.hyperfine(v_arr, Tbackground=2.73, Tex=Tex_CLASS, tau=tau, 
    xoff_v=-0.004, width=0.714/2.35482)
signal_4par_fit = pyspeckit.models.n2hp.n2hp_vtau.hyperfine(v_arr, Tex=2.89591, tau=0.450383, 
    xoff_v=-0.000168891, width=0.3033)
plt.plot(v_arr, signal_velo, color='k', drawstyle='steps-mid')
plt.plot(v_arr, signal_4par_fit, color='red', drawstyle='steps-mid')
plt.plot(v_arr, signal_4par_class, color='blue', drawstyle='steps-mid')
plt.plot(v_arr, signal_4par_fit-signal_4par_class, color='green', drawstyle='steps-mid')
plt.savefig('pyspeckit_N2Hp_compare_thin_4par.png')

# plot comparisons
# optically thin model, with optically thin approximation (3-parameter model)

sp_thin_3par.xarr.convert_to_unit('km/s')
v_arr=sp_thin_3par.xarr
signal_velo=sp_thin_3par.data

Area=0.058
tau=0.1
Tex_CLASS=T0/np.log(1+T0/(Area/tau +J_nu(2.73,T0)))

signal_3par_class = pyspeckit.models.n2hp.n2hp_vtau.hyperfine(v_arr, Tex=Tex_CLASS, tau=0.1, 
    xoff_v=-0.004, width=0.722/2.35482)
signal_3par_fit = pyspeckit.models.n2hp.n2hp_vtau.hyperfine(v_arr, Tex=3.42535, tau=0.1, 
    xoff_v=-0.00199191, width=0.306108)
plt.close()
plt.plot(v_arr, signal_velo, color='k', drawstyle='steps-mid')
plt.plot(v_arr, signal_3par_fit, color='red', drawstyle='steps-mid')
plt.plot(v_arr, signal_3par_class, color='blue', drawstyle='steps-mid')
plt.plot(v_arr, signal_3par_fit-signal_3par_class, color='green', drawstyle='steps-mid')
plt.savefig('pyspeckit_N2Hp_compare_thin_3par.png')






# plot comparisons
# optically thin model, without optically thin approximation (4-parameter model)
plt.close()
sp_thick.xarr.convert_to_unit('km/s')
v_arr=sp_thick.xarr
thick_velo=sp_thick.data
Area=51.1
tau=8.26
Tex_CLASS=T0/np.log(1+T0/(Area/tau +J_nu(2.73,T0)))
thick_4par_class = pyspeckit.models.n2hp.n2hp_vtau.hyperfine(v_arr, Tbackground=2.73, Tex=Tex_CLASS, tau=tau, 
    xoff_v=-0.001, width=0.691/2.35482)
thick_4par_fit = pyspeckit.models.n2hp.n2hp_vtau.hyperfine(v_arr, Tex=9.33, tau=8.29, 
    xoff_v=-0.000168891, width=0.29385)
plt.plot(v_arr, thick_velo, color='k', drawstyle='steps-mid')
plt.plot(v_arr, thick_4par_fit, color='red', drawstyle='steps-mid')
plt.plot(v_arr, thick_4par_class, color='blue', drawstyle='steps-mid')
plt.plot(v_arr, thick_4par_fit-thick_4par_class, color='green', drawstyle='steps-mid')

plt.savefig('pyspeckit_N2Hp_compare_thick.png')