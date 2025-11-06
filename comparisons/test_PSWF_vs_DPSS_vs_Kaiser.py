# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:52:39 2025

@author: fbondu

comparison of DPSS vs PSWF computed with Legendre polynomials

"""
import numpy                as np
import matplotlib.pyplot    as plt
import scipy.signal.windows as win
from   dPSWF                        import PSWF
from   perf.utils.LibWindowSpectrum import FourierTransform
from   perf.utils.LibWindowPerf     import concentrationDefect

OrderN = 19 # maximum index of psi function
NW     = 10  # c/pi parameter, or relative resolution in frequency bins
Npts   = 10_000 # 
Kplot  = 3  # < OrderN+1
Keigen = 10

# PSWF computed with Legendre polynomials
c     = np.pi*NW
PSWF  = PSWF(c, OrderN, Npts, normalizationType='window')
wpsi0 = PSWF.psi_functions[0]

# PSWF (DPSS) computed with long tridiagonal matrix
scidpss, scivals = win.dpss(Npts, NW, Kmax=OrderN+1, sym=True, norm=2, return_ratios=True)
wdpss0 = scidpss[0]

# Kaiser window
wkaiser = win.kaiser(Npts, c, sym=True)
wkaiser = wkaiser/np.sqrt(sum(wkaiser**2))

# plot windows
plt.figure(0)
ax1 = plt.subplot(211)
ax1.plot(wpsi0,'b')
ax1.plot(wdpss0,color='orange')
ax1.plot(wkaiser,color='green')
ax1.title.set_text('windows')
ax1.legend(['dPSWF_0','DPSS_0','Kaiser'])
ax2 = plt.subplot(212)
ax2.title.set_text('window differences')
ax2.plot(wpsi0-wdpss0, color='orange')
ax2.plot(wpsi0-wkaiser, color='green')
ax2.legend(['dPSWF_0 - DPSS_0','dPSWF_0 - Kaiser'])
ax2.set_ylim([-7e-5,7e-5])
     # all windows on top of each other

# now spectra of each window
vecFreq  = np.linspace(0, 0.5, num=Npts*10)
_, tf_psi0  = FourierTransform(wpsi0,  center=Npts/2, Mode='FT', TF_VecFreq=vecFreq)
_, tf_dpss  = FourierTransform(wdpss0, center=Npts/2, Mode='FT', TF_VecFreq=vecFreq)
_, tf_kai   = FourierTransform(wkaiser, center=Npts/2, Mode='FT', TF_VecFreq=vecFreq)

plt.figure(2)
ax1 = plt.subplot(211)
ax1.loglog(vecFreq,np.abs(tf_psi0),color='b')
ax1.loglog(vecFreq,np.abs(tf_dpss),color='orange')
ax1.loglog(vecFreq,np.abs(tf_kai),'green')
ax1.legend(['dPSWF_0','DPSS_0','Kaiser'])
ax1.grid(visible=True,which='both',axis='both')
ax1.title.set_text(r'absolute value of Fourier transforms of windows')
ax2 = plt.subplot(212)
ax2.loglog(vecFreq,np.abs(tf_psi0)/np.abs(tf_dpss),color='orange')
ax2.loglog(vecFreq,np.abs(tf_psi0)/np.abs(tf_kai), color='green')
ax2.loglog(vecFreq,np.abs(tf_dpss)/np.abs(tf_kai), color='red')
ax2.legend(['tfdPSWF_0 / tfDPSS_0','tfdPSWF_0 / tfKaiser','tfDPSS_0 / tfKaiser'])

plt.figure(3)
plt.semilogy(1-PSWF.lambdas,'bo')
plt.semilogy(np.abs(1-scivals),'b--')

# concentrations in the -W,+W band
W = NW/Npts
concD_PSWF = concentrationDefect(wpsi0, W)
concD_dpss = concentrationDefect(wdpss0,W)
concD_kai  = concentrationDefect(wkaiser,W)
print('concentration defect dPSWF  ',concD_PSWF)
print('concentration defect DPSS   ',concD_dpss)
print('concentration defect Kaiser ',concD_kai)