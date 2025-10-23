# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:36:12 2025

@author: fbondu
"""

import numpy                as np
from   scipy.signal.windows import dpss

OrderN = 30 # maximum index of psi function
NW     = 10  # c/pi parameter, or relative resolution in frequency bins
Npts   = 2_000 # ok up to 20 000 000 at least
Keigen = 19 # < OrderN+1, maximum index for computation of orthogonality

c = np.pi*NW
vec_x = np.linspace(-1, 1, Npts)

scidpss, scivals = dpss(Npts, NW, Kmax=OrderN+1, sym=True, norm=2, return_ratios=True)

# normalization defects (sum of squares equal to one)
TabNorms = np.sum(scidpss**2,1)
print('\n   norm of DPSS maximum error: ', '{:.3g}'.format(max(np.abs(TabNorms-1))))

TabPdtScal = np.zeros((Keigen+1,Keigen+1))
for k in range(Keigen+1):
    for j in np.arange(k+1,Keigen+1):
        TabPdtScal[k,j] = sum(scidpss[k]*scidpss[j])
print('   orthogonality of DPSS max error: ','{:.3g}'.format(max(np.max(np.abs(TabPdtScal),0))))