# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:36:12 2025

@author: fbondu
"""

import numpy                as np
import matplotlib.pyplot    as plt
from   scipy.signal.windows import dpss

OrderN = 21 # maximum index of psi function
NW     = 10  # c/pi parameter, or relative resolution in frequency bins
Npts   = 2_000 # ok up to 20 000 000 at least
Kplot  = 3  # < OrderN+1, number of functions to plot

vec_x = np.linspace(-1, 1, Npts)

# test of scipy DPSS

scidpss, scivals = dpss(Npts, NW, Kmax=OrderN+1, sym=True, norm=2, return_ratios=True)

plt.figure(1)
for k in range(0,Kplot):
    plt.plot(vec_x, scidpss[k,:])
# scipy/dpss does not necessarily have the convention of 
# - positive value at x=0 for even functions
# - postivie slope at x=0 for odd functions