# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:40:27 2025

@author: fbondu
"""


import numpy                    as     np
import matplotlib.pyplot        as     plt
from   dPSWF                    import PSWF
from   perf.utils.LibWindowSpectrum import FourierTransform

OrderN = 21 # maximum index of psi function
Npts   = 500 # ok up to 10 000 000 at least
NW     = 10

c = np.pi*NW

def ENB_bin_arbitrary(window,factor=10):
    # for functions that are not necessarily even and positive
    # approximation of ENB for an arbitrary window
    # maximum of spectra not necessary at f=0
    N = len(window)
    vecx = np.linspace(0,0.5,N*factor)
    tf_win = FourierTransform(window, Mode='FT', TF_VecFreq=vecx)
    wf2_max = np.max(np.abs(tf_win))**2
    return np.sum(window**2)*N/wf2_max

Psis  = PSWF(c, OrderN, Npts, normalizationType='window')
ENB = np.zeros(OrderN+1)
for k in np.arange(OrderN):
    print(k)
    ENB[k] = ENB_bin_arbitrary(Psis.psi_functions[k])

plt.figure(1)
plt.plot(ENB)
plt.grid(visible=True,which='both',axis='both')
plt.title(r'window ENB vs window index, $NW$=10')
plt.xlabel('index')
plt.ylabel('ENB')