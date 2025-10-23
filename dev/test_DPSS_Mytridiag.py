# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:05:59 2025

@author: fbondu
"""

from   DPSS_tridiag       import DPSS
from   time               import time
import matplotlib.pyplot  as plt

OrderN = 21      # maximum index of psi function
NW     = 10      # c/pi parameter, or relative resolution in frequency bins
Npts   = 200_000 # wrong if >= 92682
Kplot  = 3       # < OrderN

tic = time()
DPSSObj = DPSS(NW,OrderN,Npts,return_lambdas=False)
toc = time()
print('evaluation of ',OrderN+1,' DPSS on ',Npts,' points in ',toc-tic,' seconds')

tic = time() # separate computation of eigenvalues
DPSSObj._Lambdas()
toc = time()
print('evaluation of ',OrderN+1,' eigenvalues in ',toc-tic,' seconds')

print(1-DPSSObj.lambdas)

plt.figure(1)
for k in range(Kplot):
    plt.plot(DPSSObj.psi_functions[k])
    
plt.figure(2)
plt.semilogy(1-DPSSObj.lambdas)