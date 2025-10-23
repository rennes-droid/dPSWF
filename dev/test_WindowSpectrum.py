# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:35:45 2025

@author: fbondu
"""

import numpy                        as     np
import matplotlib.pyplot            as     plt
from   perf.utils.LibWindowSpectrum import FourierTransform

N = 101

# create gaussian curve
w_center = N/3
w_sigma  = 6
x        = np.arange(N)

# test with user defined vector of frequencies
freq2 = np.linspace(-0.5, 0.5,1000)

# create first 2 HG functions, compute their Fourier Transform
w0       = (1/(w_sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*(x-w_center)**2/w_sigma**2)
freq, w0_fft = FourierTransform(w0, center=w_center) # specify center to remove pure delay
_, w0_tf     = FourierTransform(w0, center=w_center, Mode='FT', TF_VecFreq=freq2)

w1       = w0*(x-w_center)/2
freq, w1_fft = FourierTransform(w1,center=w_center)
_, w1_tf = FourierTransform(w1, center=w_center, Mode='FT', TF_VecFreq=freq2)

# time domain plot
plt.figure(1)
plt.plot(w0)
plt.plot(w1)

# frequency domain plot
plt.figure(2)
plt.plot(freq, w0_fft.real) # even function
plt.plot(freq, w1_fft.imag) # odd  function
plt.plot(freq2, w0_tf.real)
plt.plot(freq2, w1_tf.imag)

    # ==>  parity, sign of transform ok