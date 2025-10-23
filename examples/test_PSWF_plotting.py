# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:59:04 2025

@author: fbondu
"""

import numpy             as np
import matplotlib.pyplot as plt
from   dPSWF             import PSWF

OrderN = 21 # max index of psi function
NW     = 10
Npts   = 200_000
Kplot  = 3 #  max order to plot < OrderN

c = np.pi*NW

Psi = PSWF(c, OrderN, Npts, normalizationType='L2')

for k in range(0,Kplot+1):
    plt.plot(Psi.UnitVec,Psi.get_psi_k(k))

# print eigenvalues
for k in range(len(Psi.lambdas)):
    print('k={:2d} {:.15f}'.format(k,Psi.lambdas[k]))