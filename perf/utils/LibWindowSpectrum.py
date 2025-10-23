# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 09:34:11 2025

@author: fbondu
"""

import numpy  as     np
from   typing import Literal
from   numba  import njit

@njit
def fast_FT(VecWindow, VecFreq):
    N = len(VecWindow)
    vecExp = np.exp(-1j*2*np.pi*VecFreq[:,None]*np.arange(N))
    VecTF = VecWindow*vecExp
    VecTF = np.sum(VecTF,1)
    return VecTF

def FourierTransform(VecWindow, Ts=1, center=0, parity:Literal['none','even','odd']='none', Mode:Literal['fft','FT']='fft', TF_VecFreq=np.array([0])):
    """
    Parameters
    ----------
    VecWindow : numpy array of floats
        windows coefficients
    Ts : float, optional
        sampling time. The default is 1.
    center : float, optional
        center of window, in bins or in time if Ts != 1. The default is 0.
    parity of window : Literal['none','even','odd'], optional
        returns real value for even windows and imaginary value for odd windows. The default is 'none'.
    Mode : Literal['fft','FT'], optional
        'FFT' for computation with FFT; 'FT' for computation with user-defined frequency vector. The default is 'fft'.
    TF_VecFreq : array of floats, optional
        user defined frequency vector for 'FT' mode. The default is np.array([0]).

    Returns
    -------
    frequency vector (array), Fourier transform vector (array).

    """
    N    = len(VecWindow)
    if Mode == 'fft': # compute Fourier Transform with FFT
        freq  = np.fft.fftfreq(N)
        freq  = np.fft.fftshift(freq)
        freq  = freq/Ts
        VecTF = np.fft.fft(VecWindow)
        VecTF = np.fft.fftshift(VecTF)
    else: # own vector of frequencies
        VecTF = fast_FT(VecWindow, TF_VecFreq*Ts)
        freq  = TF_VecFreq
    if parity == 'even': # imaginary part negligible and meaningless
        VecTF = np.real(VecTF)
    elif parity == 'odd': # real part negligible and meaninless
        VecTF = np.imag(VecTF)
    VecTF = VecTF * Ts # / Hz unit
    VecTF = VecTF * np.exp(1j*2*np.pi*freq*center) # compensate for center not at k=0
    return freq, VecTF