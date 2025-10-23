# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:59:04 2025

@author: fbondu
"""

import numpy              as     np
import matplotlib.pyplot  as     plt
from   dPSWF              import PSWF

OrderN = 25
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
    
plt.subplot(121)
plt.plot(PsiObj.lambdas)
plt.grid(visible='on',which='both')
plt.ylabel(r'$\lambda_k$')
plt.subplot(122)
plt.semilogy(1-PsiObj.lambdas)
plt.grid(visible='on',which='both')
plt.ylabel(r'$1-\lambda_k$')