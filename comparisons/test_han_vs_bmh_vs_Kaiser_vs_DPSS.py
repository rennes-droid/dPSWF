# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:33:26 2025

@author: fbondu
"""

# test of windows for Welch method

import scipy.signal.windows         as win
import numpy                        as np
import matplotlib.pyplot            as plt
from   perf.utils.LibWindowSpectrum import FourierTransform
from   perf.utils.LibWindowPerf     import concentration, ENB_bin

N = 200 # length of all windows

def WinNorm(window):
    # to garanty that all windows have same definition of norm
    # np.sum(w(n)**2) = 1
    return window/np.sqrt(np.sum(window**2))

facMul = 100
freqFresp = np.linspace(1/(N*facMul),0.5,N*facMul)

han = win.hann(N, sym=False)
han = WinNorm(han)
_, f_han = FourierTransform(han,Mode='FT',TF_VecFreq=freqFresp)

bmh = win.blackmanharris(N, sym=False)
bmh = WinNorm(bmh)
_, f_bmh = FourierTransform(bmh,Mode='FT',TF_VecFreq=freqFresp)

NW = 4.0
kai = win.kaiser(N, NW*np.pi, sym=False)
kai = WinNorm(kai)
_, f_kai = FourierTransform(kai,Mode='FT',TF_VecFreq=freqFresp)

dpss,lam = win.dpss(N, NW, sym=False, return_ratios=True)
dpss = WinNorm(dpss)
_, f_dpss = FourierTransform(dpss,Mode='FT',TF_VecFreq=freqFresp)

plt.figure(1)
plt.plot(han)
plt.plot(bmh)
plt.plot(kai)
plt.plot(dpss)
plt.legend(['Hann','BMH',r'Kaiser $\beta=4.0\pi$','DPSS NW=4.0'])
plt.title('normalized non-symmetric windows N=200')


plt.figure(2)
plt.loglog(freqFresp,np.abs(f_han)**2)
plt.loglog(freqFresp,np.abs(f_bmh)**2)
plt.loglog(freqFresp,np.abs(f_kai)**2)
plt.loglog(freqFresp,np.abs(f_dpss)**2)
plt.legend(['Hann','BMH',r'Kaiser $\beta=4.0\pi$','DPSS NW=4.0'])
plt.ylim([1e-12,2e2])
plt.grid(visible='on',which='both')
plt.title('Filtering performance of non-symmetric windows N=200')
plt.ylabel(r'|W(f)|$^2$')

# Spectral concentration

def print_window_perf(window,W):
    enb_bin = ENB_bin(window)
    print('ENB - bins %5.3f'%(enb_bin))
    enb_freq_red = enb_bin/len(window)
    c = concentration(window, enb_freq_red/2)
    print('energy fraction not in [-ENB/2,+ENB/2] %5.3f'%(1-c))
    c = concentration(window, W)
    print('NW= ',NW,' energy fraction not in [-NW,+NW] %3.1e'%(1-c))
    return

W = NW/N
print('Hanning')
print_window_perf(han,W)
print('\nBMH')
print_window_perf(bmh,W)
print('\nKaiser')
print_window_perf(kai,W)
print('\nDPSS')
print_window_perf(dpss,W)