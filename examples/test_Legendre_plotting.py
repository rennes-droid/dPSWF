# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:25:20 2025

@author: fbondu
"""

import numpy              as     np
import matplotlib.pyplot  as     plt
from   dPSWF              import PolyLegendreAllFromRoots

# compare with values and curves with 
# https://en.wikipedia.org/wiki/Legendre_polynomials

OrdreMax   = 90
Npoints    = 100_000
normalized = False
# ordreMax 300 works

Pn = PolyLegendreAllFromRoots(OrdreMax, Npoints, normalized)

#########################
# plot some polynomials #
#########################

OrdreMaxDisplay = 6 # < OrdreMax
for k in range(0,OrdreMaxDisplay):
    plt.plot(Pn.VecUnit, Pn.TabPolys[k])

n = 80 # n < OrdreMax
plt.plot(Pn.VecUnit,Pn.TabPolys[n])

#############################
# display first polynomials #
#############################
# compare with wikipedia 10 first polynomials
print('first Legendre polynomials coefficients')
for k in range(0,OrdreMaxDisplay):
    polyn = np.polynomial.polynomial.polyfromroots(Pn.RootsTable[k])*Pn.CalibsTable[k]
    print(k,'  ',polyn)