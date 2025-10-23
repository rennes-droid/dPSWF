# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:59:04 2025

@author: fbondu
"""

# computation of energy concentration with integration of Fourier Spectrum
# !! not accurate with default point number in GL integration if lambda << 1

import numpy              as     np
import matplotlib.pyplot  as     plt
from   dPSWF              import PSWF

OrderN = 21
NW     = 10
Npts   = 2_000

c = np.pi*NW

# test of DPSS through Legendre polynomials, as is Wang doi:10.4208/jms.v50n2.17.01

PsiObj = PSWF(c, OrderN, Npts, normalizationType='window')


#######################
# eigenvalues         #
#######################

# compare with DPSS in Persival and Walden "spectral analysis for physical applications"
# reprinted 1998
# table pg 382

for k,lamb in enumerate(PsiObj.lambdas):
    print(format(k,'2d'),format(lamb,'.17f'))
    

##########################################################################
# concentration values calculated with GL quadrature in frequency domain #
##########################################################################

from perf.utils.LibWindowPerf import concentration, concentrationDefect

W = NW/Npts

Vconc  = concentration(PsiObj.psi_functions, W)
VconcD = concentrationDefect(PsiObj.psi_functions, W)

plt.subplot(121)
plt.semilogy(PsiObj.lambdas,'+')
plt.semilogy(Vconc,'--')
plt.grid(visible='on',which='both')
plt.ylabel(r'$\lambda_k$')
plt.legend([r'$\lambda$ Wang','concentration'])
plt.subplot(122)
plt.semilogy(1-PsiObj.lambdas,'+')
plt.semilogy(VconcD,'--')
plt.grid(visible='on',which='both')
plt.ylabel(r'$1-\lambda_k$')
# much better estimate of 1-\lambda
# with Gauss Legendre integration of Fourier Transform in [W,1/2]