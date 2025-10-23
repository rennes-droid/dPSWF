# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:59:04 2025

@author: fbondu
"""

import numpy as np
from dPSWF import PSWF

OrderN = 2
NW     = 10
Npts   = 3

c = np.pi*NW

# test of DPSS through Legendre polynomials, as is Wang doi:10.4208/jms.v50n2.17.01

PSWFObj = PSWF(c, 2, 2) # precompilation of functions