# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:23:13 2025

@author: fbondu
"""

import numpy                    as     np
import matplotlib.pyplot        as     plt
from   dPSWF                    import PSWF
from   perf.utils.LibWindowPerf import ENB_bin
from   scipy.optimize           import fsolve

OrderN = 21 # maximum index of psi function
Npts   = 10_000 # ok up to 10 000 000 at least

def Navg_vs_NW(NW):
    return 2*NW - np.log(NW)
    #return np.floor(2*NW - np.log(NW)).astype(int) won’t do well
    # arXiv:2103.11586v1 Karnik et al. 2021

def NW_vs_Navg(Navg):
    def eqzero(NW, Navg):
        return Navg - Navg_vs_NW(NW)
    res = fsolve(eqzero, Navg/2, args=(Navg,))
    return res

VecNW = np.linspace(3,100,200)
ENB   = np.copy(VecNW)

for k,NW in enumerate(VecNW):
    c     = np.pi*NW
    Psis  = PSWF(c, OrderN, Npts, normalizationType='window')
    wpsi0 = Psis.psi_functions[0]
    ENB[k] = ENB_bin(wpsi0)

plt.figure(1)
plt.plot(VecNW,ENB)
plt.grid(visible=True,which='both',axis='both')
plt.title(r'ENB vs NW for $\psi_0$ function')
plt.xlabel('NW')
plt.ylabel('ENB')

VecNavg = np.linspace(2,100,num=99)
gain    = np.copy(VecNavg)
for k,Navg in enumerate(VecNavg):
    NW    = NW_vs_Navg(Navg)
    c     = np.pi*NW
    Psis  = PSWF(c, OrderN, Npts, normalizationType='window')
    wpsi0 = Psis.psi_functions[0]
    ENB_pswf  = ENB_bin(wpsi0)
    ENB_welch = 1.5*Navg
    gain[k] = ENB_welch/ENB_pswf

plt.figure(2)
plt.plot(VecNavg, gain)
plt.title('gain in frequency resolution Welch/PSWF vs average number')
plt.xlabel('number of averages')
plt.ylabel('frequency resolution gain')
plt.grid(visible=True,which='both',axis='both')