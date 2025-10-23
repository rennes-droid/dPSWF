# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:25:20 2025

@author: fbondu
"""

from dPSWF import PolyLegendreAllFromRoots
from time  import time

OrdreMax   = 100
Npoints    = 100_000
normalized = True
# ordreMax 300 works

###############################################
# evaluate several polynomials simultaneously #
###############################################

tic = time() # 1 function, 1 point: mainly measures compilation time
Pn = PolyLegendreAllFromRoots(1, 1, normalized)
toc = time()
print('compilation in ',format(toc-tic,'.3f'),' seconds')

# now functions with numba are compiled
tic = time()
Pn = PolyLegendreAllFromRoots(OrdreMax, Npoints, normalized)
toc = time()
print('evaluation of first ',OrdreMax,' polynomials on ',Pn.VecUnitLength,' points in ',format(toc-tic,'.3f'),' seconds')