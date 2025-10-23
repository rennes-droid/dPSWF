# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:59:04 2025

@author: fbondu
"""

import numpy             as     np
from   dPSWF             import PSWF
from   time              import time

OrderN = 19
NW     = 10
Npts   = 2_000_000

c = np.pi*NW

# test of DPSS through Legendre polynomials, as is Wang doi:10.4208/jms.v50n2.17.01

tic = time()
PsiObj = PSWF(c, 2, 2) # precompilation of functions
toc = time()
print('precompilation time ',format(toc-tic,'.3f'),' seconds')

tic = time()
PsiObj = PSWF(c, OrderN, Npts, normalizationType='L2')
toc = time()
print('evaluation of ',OrderN+1,' PSWF decomposition',' on ', Npts,' points on Legendre Polynomials in ',format(toc-tic,'.3f'),' seconds')