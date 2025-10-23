# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 10:51:32 2025
@author: fbondu

See how PSWF can be extrapolated outside [-1,+1] using Fourier Transform
possibly use that to compute eigenvalues (compute confiment in [-1, 1] wrt total energy)

"""

import numpy                        as np
import matplotlib.pyplot            as plt
from   dPSWF                        import PSWF
from   perf.utils.LibWindowSpectrum import FourierTransform

OrderN = 31 # maximum index of psi function
NW     = 10  # c/pi parameter, or relative resolution in frequency bins
Npts   = 10_000 # 

c = np.pi*NW
Psi = PSWF(c, OrderN, Npts, normalizationType='L2')

n = 19
Psi0 = Psi.psi_functions[n]
lambda0 = Psi.lambdas[n]
vecx = Psi.UnitVec

# vecx  = np.linspace(0, 1.4,10_000)
# Psi_x = PSWF.Eval(vecx, 0)

# plt.semilogy(PSWF.UnitVec,PSWF.psi_functions[0])
# plt.plot(vecx, Psi_x)
#plt.ylim([0,1e3])

plt.figure(1)
plt.plot(vecx, np.abs(Psi0))

#freq = np.linspace(-10, 10, 100)
#freq, yF = FourierTransform(Psi0,parity='even', Ts=2/Npts, UseExtVecFreq=True, VecFreq=freq)
Tsampling = 1 # whatever number anyway
Tmes      = Npts*Tsampling # total measurement time
freq, yF  = FourierTransform(Psi0, Ts=Tsampling, center = Npts/2)
yF        = yF *2./Tmes

# rescaling according to Wang equation 2.15
mu = np.sqrt(2*np.pi*lambda0/c)
B = c/(np.pi*Tmes)
plt.semilogy(freq/B,np.abs(yF/mu))
# ok PSWF are the same in [-1,+1] as expected