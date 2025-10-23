# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:25:20 2025

@author: fbondu
"""

import numpy              as     np
from   dPSWF              import PolyLegendreAllFromRoots
from   time               import time

OrdreMax   = 70
Npoints    = 1_000_000
normalized = True
# ordreMax 300 works

Pn = PolyLegendreAllFromRoots(OrdreMax, Npoints, normalized)


if normalized == True:
#############################
# relative norm error       #
#############################

    tic = time()
    norms = np.zeros(OrdreMax+1)
    for k in range(0,OrdreMax+1):
        norms[k] = Pn.Norm(k)
    toc = time()
    print('norm of normalized polynomials maximum error: ', format(max(np.abs(norms-1)),'.3g'))
    print('norm of ',OrdreMax,' polynomials with Gauss Legendre quadrature of order ',2*OrdreMax+2,' in ',format(toc-tic,'.3f'),' seconds')

#############################
# measure orthogonality     #
#############################

    tic = time()
    pdt = np.zeros((OrdreMax+1,OrdreMax+1))
    for k in range(0,OrdreMax+1):
        for j in range(k+1, OrdreMax+1):
            pdt[k] = Pn.ScalarProduct(k,j)
    toc = time()
    print('orthogonality of orthogonal normalized polynomials maximum error: ',format(max(np.max(np.abs(pdt),0)),'.3g'))
    print('orthogonality of ',(OrdreMax*(OrdreMax+1))//2,' polynomials with Gauss Legendre quadrature of order ',2*OrdreMax+2,' in ',format(toc-tic,'.3f'),' seconds')