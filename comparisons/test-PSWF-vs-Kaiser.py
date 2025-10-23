# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 09:51:45 2025

@author: fbondu
"""

import numpy             as np
import matplotlib.pyplot as plt
from   dPSWF             import PSWF

from scipy.signal.windows import kaiser
from scipy.special        import iv


OrderN = 21 # maximum index of psi function
NW     = 10  # c/pi parameter, or relative resolution in frequency bins
Npts   = 2_000 # ok up to 10 000 000 at least
Kplot  = 3  # < OrderN+1
Keigen = 10

c = np.pi*NW

vecx = np.linspace(-1,1,Npts)

def MyKaiser(vecx, n, beta):
    vecsq = beta*np.sqrt(1-(vecx)**2)
    return iv(n, vecsq)/iv(n,beta)

PSWF = PSWF(c, OrderN, Npts, normalizationType='L2')
psi0 = PSWF.psi_functions[0]
psi0 = psi0/np.sqrt(sum(psi0**2))

kai  = kaiser(Npts, c)
kai  = kai/np.sqrt(sum(kai**2))

kaip = MyKaiser(vecx, 0, c)
kaip = kaip/np.sqrt(sum(kaip**2))

# time domain

plt.plot(vecx, psi0)
plt.plot(vecx, kai)
plt.plot(vecx, kaip)


# frequency domain
f_psi0 = np.fft.fft(psi0)[:Npts//2]
f_kai  = np.fft.fft(kai)[:Npts//2]

# well, only on points where f =1/N, misleading

plt.figure()
plt.loglog(np.abs(f_psi0))
plt.loglog(np.abs(f_kai))

# very similar performances !!