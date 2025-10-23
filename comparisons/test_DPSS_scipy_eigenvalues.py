# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:36:12 2025

@author: fbondu
"""

from   scipy.signal.windows     import dpss
from   perf.utils.LibWindowPerf import concentration

OrderN = 14 # maximum index of psi function
NW     = 8  # c/pi parameter, or relative resolution in frequency bins
Npts   = 31 # ok up to 20 000 000 at least
Keigen = 14 # < OrderN+1, maximum index for display of eiqenvalues

# test of scipy DPSS

scidpss, scivals = dpss(Npts, NW, Kmax=OrderN+1, sym=True, norm=2, return_ratios=True)

W = NW/Npts
conc = concentration(scidpss, W)

print('\nDPSS internal computation of eigenvalues // concentration computation')
# print eigenvalues, compare with Percival et Walden pg 382
for k in range(Keigen+1):
    print('k={:2d} {:.15f}'.format(k,scivals[k]),'  k={:2d} {:.15f}'.format(k,conc[k]))