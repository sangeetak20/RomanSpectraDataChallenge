import copy
import numpy as np
import scipy as sp
from scipy.stats import gmean
from scipy.optimize import curve_fit, leastsq, minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import re, os, glob
from astropy.io import fits
from astropy.table import Table
# import pandas as pd
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(70,1-0.7,name='GenericCDM')
from astropy.cosmology import z_at_value
# import seaborn as sns
from scipy import interpolate
from scipy.integrate import quad
import sys
import shelve
from scipy.interpolate import UnivariateSpline ,splev, splrep
from matplotlib.colors import LogNorm
import h5py

def redshift(vel):
    return vel/300000.0
    #This function also represent the line dispersion in A through a velocity dispersion in km/s also taking into account
    # that the spectrograph itself already broadens the emission lines. This way you automatically fit for the intrinsic line dispersion
def line_width(vel_sigma,rest_line,inst_res_fwhm):
    sigma = vel_sigma/(300000.0-vel_sigma)*rest_line
    return np.sqrt(sigma**2+(inst_res_fwhm/2.354)**2)

def line_width_recons(vel_sigma,rest_line,inst_res_fwhm=0):
    sigma = vel_sigma/(300000.0-vel_sigma)*rest_line
    return np.sqrt(sigma**2+(inst_res_fwhm/2.354)**2)

def gauss(wave,amplitude,vel,vel_sigma, rest_wave,inst_res_fwhm):
    line = (amplitude)*np.exp(-(wave-(rest_wave*(1+redshift(vel))))**2/(2*(line_width(vel_sigma, rest_wave,inst_res_fwhm))**2))
    return line

def gauss_recons(wave,amplitude,vel,vel_sigma,rest_wave):
    line = (amplitude)*np.exp(-(wave-(rest_wave*(1+redshift(vel))))**2/(2*(line_width_recons(vel_sigma, rest_wave))**2))
    return line

def Ha_gauss_model(wave,amp_Ha_core,amp_NII_core,vel_core,vel_sigma_core,amp_Ha_wing,vel_wing,vel_sigma_wing,m,c):
    Ha_core = gauss(wave,amp_Ha_core,vel_core,vel_sigma_core,6562,2.4)
    NII6548_core = (1/2.61)*gauss(wave,amp_NII_core,vel_core,vel_sigma_core,6548,2.4)
    NII6584_core = gauss(wave,amp_NII_core,vel_core,vel_sigma_core,6584,2.4)
    Ha_wing = gauss(wave,amp_Ha_wing,vel_wing,vel_sigma_wing,6562,2.4)
    cont = (wave/1000)*m+c
    return Ha_core + NII6548_core + NII6584_core + Ha_wing + cont

def Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel,vel_sigma):
    Hb = gauss(wave,amp_Hb,vel,vel_sigma,4861.33,2.4)
    OIII_4959 = (0.33)*gauss(wave,amp_OIII5007,vel,vel_sigma,4958.9,2.4)
    OIII_5007 = gauss(wave,amp_OIII5007,vel,vel_sigma,5006.8,2.4)
    return Hb + OIII_4959 + OIII_5007

def Hb_Fe_doublet_gauss(wave,amp_Hb,amp_Fe5018,vel,vel_sigma):
    Hb = gauss(wave,amp_Hb,vel,vel_sigma,4861.33,2.4)
    Fe_4923 = 0.81*gauss(wave,amp_Fe5018,vel,vel_sigma,4923,2.4)
    Fe_5018 = gauss(wave,amp_Fe5018,vel,vel_sigma,5018,2.4)
    return Hb+Fe_4923+Fe_5018

def MgII_gauss(wave,amp_MgII,vel,vel_sigma):
    MgII = gauss(wave,amp_MgII,vel,vel_sigma,2800,2.4)
    return MgII

def CIII_gauss(wave,amp_CIII,vel,vel_sigma):
    CIII = gauss(wave,amp_CIII,vel,vel_sigma,1909,2.4)
    return CIII

def CIV_gauss(wave,amp_CIV,vel,vel_sigma):
    CIV = gauss(wave,amp_CIV,vel,vel_sigma,1549,2.4)
    return CIV

def SiIV_gauss(wave,amp_SiIV,vel,vel_sigma):
    SiIV = gauss(wave,amp_SiIV,vel,vel_sigma,1402,2.4)
    return SiIV

def SiIII_gauss(wave,amp_SiIII,vel,vel_sigma):
    SiIII = gauss(wave,amp_SiIII,vel,vel_sigma,1892,2.4)
    return SiIII

def AlIII_gauss(wave,amp_AlIII,vel,vel_sigma):
    AlIII = gauss(wave,amp_AlIII,vel,vel_sigma,1860,2.4)
    return AlIII

def NV_gauss(wave,amp_NV,vel,vel_sigma):
    NV = gauss(wave,amp_NV,vel,vel_sigma,1240,2.4)
    return NV

def Hb_gauss_model(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,m,c):
    #(amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    #broad_OIII = 0
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1)
    Hb_broad2 = 0#Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2)
    cont = (wave/1000.0)*m+c
    return narrow_OIII+Hb_broad1+Hb_broad2+cont


def MgII_model(wave,amp_MgII,vel_MgII,vel_sigma_MgII,amp_MgII_blr,vel_MgII_blr,vel_sigma_MgII_blr,m,c):
    #(amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_MgII = MgII_gauss(wave,amp_MgII,vel_MgII,vel_sigma_MgII)
    #broad_OIII = 0
    broad_MgII = MgII_gauss(wave,amp_MgII_blr,vel_MgII_blr,vel_sigma_MgII_blr)
    cont = (wave/1000.0)*m+c
    return narrow_MgII+broad_MgII+cont

def CIII_model(wave,amp_AlIII,amp_SiIII,amp_CIII,vel,vel_sigma,amp_AlIII_blr, amp_SiIII_blr, amp_CIII_blr,vel_blr,vel_sigma_blr,m,c):
    #(amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_CIII = CIII_gauss(wave,amp_CIII,vel,vel_sigma)
    narrow_SiIII = SiIII_gauss(wave,amp_SiIII,vel,vel_sigma)
    narrow_AlIII = AlIII_gauss(wave,amp_AlIII,vel,vel_sigma)
    blr_CIII = CIII_gauss(wave,amp_CIII_blr,vel_blr,vel_sigma_blr)
    blr_SiIII = SiIII_gauss(wave,amp_SiIII_blr,vel_blr,vel_sigma_blr)
    blr_AlIII = AlIII_gauss(wave,amp_AlIII_blr,vel_blr,vel_sigma_blr)
    #broad_OIII = 0
    cont = (wave/1000.0)*m+c
    return narrow_CIII+narrow_SiIII+narrow_AlIII+blr_CIII+blr_SiIII+blr_AlIII+cont

def CIV_model(wave,amp_CIV,vel,vel_sigma,amp_CIV_blr,vel_blr,vel_sigma_blr,m,c):
    #(amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_CIV = CIV_gauss(wave,amp_CIV,vel,vel_sigma)
    blr_CIV = CIV_gauss(wave,amp_CIV_blr,vel_blr,vel_sigma_blr)

    #broad_OIII = 0
    cont = (wave/1000.0)*m+c
    return narrow_CIV+blr_CIV+cont

def HeII_gauss(wave,amp_HeII,vel,vel_sigma):
    HeII = gauss(wave,amp_HeII,vel,vel_sigma,1640,2.4)
    return HeII

def cont(wave,m,c):    
    continuum = (wave/1000.0)*m+c
    return continuum

def log_lum(z,flux):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    D_L = cosmo.luminosity_distance(z)
    dist = D_L.to(u.cm)
    log_emission_line_lum = np.log10(4*np.pi)+2*np.log10(dist.value)+np.log10(flux)
    return log_emission_line_lum