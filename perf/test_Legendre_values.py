# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:25:20 2025

@author: fbondu
"""

import numpy  as     np
from   dPSWF  import PolyLegendreAllFromRoots

OrdreMax   = 70
Npoints    = 3 # [-1, 0, 1]
normalized = False # classical Legendre polynomials, not normalized
# ordreMax 300 works

Pn = PolyLegendreAllFromRoots(OrdreMax, Npoints, normalized)


#############################
# check values at -1 and +1 #
#############################

if normalized==False:
    LegValsAtEnds  = np.zeros((2,OrdreMax+1), dtype=float)
    for k in np.arange(0,OrdreMax+1):
        LegValsAtEnds[:,k]  = Pn.TabPolys[k][[0,-1]]
    print('value at +1 relative error: ', max(np.abs(LegValsAtEnds[1,:]))-1)
    print('value at -1 relavite error: ', max(np.abs(LegValsAtEnds[-1,:])-1))
    