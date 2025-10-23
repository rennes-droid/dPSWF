# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:59:04 2025

@author: fbondu
"""

import numpy             as np
from   dPSWF             import PSWF

OrderN = 19
NW     = 10
Npts   = 2_000_000

c = np.pi*NW

Psi = PSWF(c, OrderN, Npts, normalizationType='L2')

# norms

print('\n')
print('Evaluation of digitized PSWF with Gaussian-Legendre quadratures')
TabNorms = np.zeros(OrderN+1)
for k in range(OrderN+1):
    TabNorms[k] = Psi.Norm_LG_L2(k)
print('   norm of PSWF error: ', '{:.3g}'.format(max(np.abs(TabNorms-1))))

TabPdtScal = np.zeros((OrderN+1,OrderN+1))
for k in range(OrderN+1):
    for j in np.arange(k+1,OrderN+1):
        TabPdtScal[k,j] = Psi.PdtScal_LG_L2(k, j)
print('   orthogonality of PSWF max error: ','{:.3g}'.format(max(np.max(np.abs(TabPdtScal),0))))

# from  psutil import virtual_memory
# print(virtual_memory())

print('\n')
print('Evaluation of digitized PSWF as windows')
# test of digitized PSWF as DPSS normalized for windows
Psi.set_normalization('window')
TabNorms = np.sum(Psi.psi_functions**2,1)
print('   norm of maximum PSWF error: ', '{:.3g}'.format(max(np.abs(TabNorms-1))))

TabPdtScal = np.zeros((OrderN+1,OrderN+1))
for k in range(OrderN+1):
    for j in np.arange(k+1,OrderN+1):
        TabPdtScal[k,j] = sum(Psi.get_psi_k(k)*Psi.get_psi_k(j))/Npts
print('   orthogonality of PSWF max error: ', '{:.3g}'.format(max(np.max(np.abs(TabPdtScal),0))))