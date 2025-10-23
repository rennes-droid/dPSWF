# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:25:20 2025

@author: fbondu
"""


from dPSWF import PolyLegendreAllFromRoots


OrdreMax   = 2
Npoints    = 10
normalized = True

# ordreMax 300 works

###############################################
# evaluate several polynomials simultaneously #
###############################################

Pn = PolyLegendreAllFromRoots(OrdreMax, Npoints, normalized)