# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:00:10 2025

@author: fbondu
"""

import numpy                        as     np
from   scipy.special                import roots_legendre
from   perf.utils.LibWindowSpectrum import FourierTransform

def ENB_bin(window):
    # Equivalent noise bandwidth in units of bins
    N = np.size(window)
    res = N*np.sum(window**2)/np.sum(window)**2
    return res

def concentration(functions, W, Ndev=None):
    # functions 2D array of windows
    # W dimensionless half-width (relative to 1/2)
    # Ndev number of points for GL quadrature
    #  measure of concentration in the frequency domain with Gauss-Legendre quadrature
    if np.ndim(functions)==2:
        Nfuncs, _ = np.shape(functions)
    else:
        Nfuncs = 1
    if Ndev == None:
        IntegOrder = (2*Nfuncs+32)*2 # see Wang estimate for Legendre expansion
    else:
        IntegOrder = Ndev
    QuadGL_roots, QuadGL_weights = roots_legendre(IntegOrder)
    xi = (QuadGL_roots+1)*W/2 # [0, W] interval
    if Nfuncs == 1:
        _, TF = FourierTransform(functions, Mode='FT',TF_VecFreq=xi)
        lambdas = sum(np.abs(TF)**2*QuadGL_weights)*W
    else:
        lambdas = np.zeros(Nfuncs)
        for k in np.arange(Nfuncs):
            _, TF  = FourierTransform(functions[k], Mode='FT',TF_VecFreq=xi)
            lambdas[k] = sum(np.abs(TF)**2*QuadGL_weights)*W
    return lambdas

def concentrationDefect(functions, W, Ndev=None):
    # functions 2D array of windows
    # W dimensionless half-width (relative to 1/2)
    # Ndev number of points for GL quadrature
    #  measure of concentration in the frequency domain with Gauss-Legendre quadrature
    # returns 1-lambda
    if np.ndim(functions)==2:
        Nfuncs, _ = np.shape(functions)
    else:
        Nfuncs = 1
    if Ndev == None:
        IntegOrder = (2*Nfuncs+32)*2 # see Wang estimate for Legendre expansion
    else:
        IntegOrder = Ndev
    QuadGL_roots, QuadGL_weights = roots_legendre(IntegOrder)
    xi = QuadGL_roots*(0.5-W)/2 + (0.5+W)/2 # [W, 0.5] interval
    if Nfuncs == 1:
        _, TF = FourierTransform(functions, Mode='FT',TF_VecFreq=xi)
        lambdasC = sum(np.abs(TF)**2*QuadGL_weights)*(0.5-W)
        # (1-W)/2 vector size * 2 for [-0.5, W] interval as well
    else:
        lambdasC = np.zeros(Nfuncs)
        for k in np.arange(Nfuncs):
            _, TF  = FourierTransform(functions[k], Mode='FT',TF_VecFreq=xi)
            lambdasC[k] = sum(np.abs(TF)**2*QuadGL_weights)*(0.5-W)
    return lambdasC 