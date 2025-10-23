# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:23:13 2025

@author: fbondu
"""

import numpy                as np
import matplotlib.pyplot    as plt
import PSWF_Legendre        as MyPSWF
from   WindowEval           import ENB_bin

OrderN = 21 # maximum index of psi function
NW     = 10  # c/pi parameter, or relative resolution in frequency bins
Npts   = 10_000 # ok up to 10 000 000 at least
Kplot  = 3  # < OrderN+1
Keigen = 10

VecNW = np.linspace(1.5,100,200)
ENB   = np.copy(VecNW)

for k,NW in enumerate(VecNW):
    c     = np.pi*NW
    PSWF  = MyPSWF.PSWF(c, OrderN, Npts, normalizationType='window')
    wpsi0 = PSWF.psi_functions[0]
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
    NW    = (1+Navg)/2
    c     = np.pi*NW
    PSWF  = MyPSWF.PSWF(c, OrderN, Npts, normalizationType='window')
    wpsi0 = PSWF.psi_functions[0]
    ENB_pswf  = ENB_bin(wpsi0)
    ENB_welch = 1.5*Navg
    gain[k] = ENB_welch/ENB_pswf

plt.figure(2)
plt.plot(VecNavg, gain)
plt.title('gain in frequency resolution Welch/PSWF vs average number')
plt.xlabel('number of averages')
plt.ylabel('frequency resolution gain')
plt.grid(visible=True,which='both',axis='both')