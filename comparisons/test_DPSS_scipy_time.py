# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:36:12 2025

@author: fbondu
"""

from   scipy.signal.windows import dpss
from   time                 import time

# from   threadpoolctl     import threadpool_limits
# was necessary with numpy version 1.26 around win.dpss instruction for Npts > 92682

OrderN = 19 # maximum index of psi function
NW     = 10  # c/pi parameter, or relative resolution in frequency bins
Npts   = 2_000_000 # ok up to 20 000 000 at least

# test of scipy DPSS

tic = time()
scidpss, scivals = dpss(Npts, NW, Kmax=OrderN+1, sym=True, 
                        norm=2, return_ratios=True)
toc = time()
print('evaluation of ',OrderN+1,' DPSS functions on ', Npts,
      ' points in ',format(toc-tic,'.3f'),' seconds')