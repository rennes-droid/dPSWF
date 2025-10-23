# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:06:15 2025

@author: fbondu
"""

import numpy as np
import scipy.linalg   as linpack


# test eigenvalues and eigenvectors

diag = np.array([2.5, 2.5, 1])
e    = np.array([0.5, 0])

evals, evecs = linpack.eigh_tridiagonal(diag, e)

# eigenvalues are classed along increasing values 1.0, 2.0, 3.0

# corresponding eigenvectors are along columns
# V1 = (0,0,1)
# V2 = 1/sqrt(2) * (-1, 1, 0)
# V3 = 1/sqrt(2) * (1, 1, 0)