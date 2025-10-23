# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 09:23:22 2025

@author: fbondu

own computation of DPSS
eigenvalues computed with Gauss-Legendre quadrature of the Fourier Transform of the window
-- might be long for long windows !

much slower than scipy.signal.

"""

import numpy          as     np
from   scipy.linalg   import eigh_tridiagonal
from   WindowEval     import concentration


class DPSS:
    # public variables
    NW            = 0 # bandwith in terms of bins (1/N) - usually a few units
    LambdasVec    = None # vector of eigenvalues
    psi_functions = None # array of psi functions, first index function number, second index x vector index
    OrderN        = 0 # computes all functions from 0 to OrderN, OrderN included
    Length        = 0 # number of points in each psi_function

    # private variables

    # public functions
    def __init__(self, NW:float, OrderN:int, Length:int, return_lambdas=False):
        self.NW = NW
        self.OrderN = OrderN
        self.Length = Length
        # tridiagonal matrix, cf. Percival Walden pg 386
        t = np.arange(Length)
        D_Vec = ((Length-1-2.*t)/2)**2 * np.cos(np.asarray(2*np.pi*NW/Length))
        E_Vec = t[1:]*(Length-t[1:])/2.
        self.evals, self.psi_functions = eigh_tridiagonal(D_Vec,E_Vec,select='i',select_range=(Length-OrderN-1,Length-1))
        self.psi_functions = self.psi_functions[:,::-1].T
            # transpose : make the matrix "row first", so that psi_functions[k] returns the window
        # Gauss-Legendre integration numbers
        if return_lambdas == True:
            self.lambdas = self._Lambdas()
        return

    # private functions
    def _TFvec(self, k_psi, freq): 
        # psi row vector
        # if freq array, should be column vector
        if type(freq)==np.ndarray:
            freq = freq.reshape((len(freq),1))
        vecExp = np.exp(-1j*2*np.pi*freq*np.arange(self.Length))
        Psi = self.psi_functions[k_psi]
        TF = Psi*vecExp
        if type(freq)==np.ndarray:
            res = np.sum(TF,1)
        else: # scalar frequency
            res = np.sum(res)
        return res
    
    def _Lambdas(self):
        self.lambdas = concentration(self.psi_functions, self.NW/self.Length)