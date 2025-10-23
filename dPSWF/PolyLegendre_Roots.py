# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 2025

@author: fbondu

representation of Legendre Polynomials based on their roots
For all Legendre polynomials, Pn(1)=1, 
    easily demonstrated by recurrence
    from the Bonnet recurrence law ;
The polynomials could also normalized 
    such that integral on [-1,1] of square modulus is one
Works for high orders on n, even when the leading factor for monic polynomial could not be evaluable (gamma(2n) crashes)

"""

import numpy       as     np
from scipy.special import roots_legendre
from numba         import njit

@njit
def fast_polyvalfromroots(VecUnit, roots):
    # VecUnit is a vector on [-1,1] ; value at +1 at end is mandatory
    n = len(VecUnit)
    result = np.ones(n)
    for i in range(n):
        for r in roots:
            result[i] *= (VecUnit[i] - r)
    return result

class PolyLegendreAllFromRoots:
    """ OrderN: maximum order of Legendre polynomial (will compute OrderN + 1 functions);
    Npoints: numbers of equally spaced numbers on interval [-1.,1.];
    normalized: if false, Pn(1)=1; else integral of |Pn|^2 on [-1.,1.] is one; 
    uses scipy to recover all roots; 
    Attribute TabPolys is an array containing the values of polynomials on [-1,1]"""
    
    OrderN        = 0 # all polynomials from 0 to orderN
    RootsTable    = np.zeros(OrderN+1, dtype=np.ndarray) # RootsTable[k] is an array with root values
    CalibsTable   = np.zeros(OrderN+1, dtype=np.float64) # array of calibrations
    VecUnitLength = 0
    VecUnit       = None
    Normalized    = False
    TabPolys      = None

    # polynomial of order k is defined by CabibsTable[k]*Product(x-xroot[i])
    
    def __init__(self, OrderN, Npoints, normalized=False):
        self.Normalized = normalized
        self.OrderN = OrderN
        # prepare for vectorial product
        self.GLquad_N = 2*OrderN+2 
        self.GLquad_roots, self.GLquad_weights = roots_legendre(self.GLquad_N)
        # construct table of roots
        self.RootsTable     = np.zeros(OrderN+1,dtype=np.ndarray)
        self.RootsTable[0]  = np.zeros(0)
        for k in range(1,OrderN+1):
            roots, _            = roots_legendre(k)
            self.RootsTable[k]  = roots
        # construct UnitVector with equidistant points on [-1,1]
        self.VecUnitLength = Npoints
        self.VecUnit = np.linspace(-1,1, Npoints)
        Nhalf = Npoints//2
        HalfVecUnit = self.VecUnit[Nhalf:]
        # get multiplication factor, ie "calibration"
        self.CalibsTable  = np.zeros(OrderN+1,dtype=np.float64)
        TabRes = np.zeros((self.OrderN+1,len(HalfVecUnit)))
        for k in range(self.OrderN+1):
            TabRes[k,:] = fast_polyvalfromroots(HalfVecUnit, self.RootsTable[k])
            self.CalibsTable[k] = 1/TabRes[k,-1]
            TabRes[k,:] = TabRes[k,:]*self.CalibsTable[k] # Pn(1) = 1
            if self.Normalized==True:
                self.CalibsTable[k] = self.CalibsTable[k]*np.sqrt(0.5+k)
                TabRes[k,:] = TabRes[k,:]*np.sqrt(0.5+k)
        TabRes_neg = np.copy(np.flip(TabRes,1)) # image for negative x values
        TabRes_neg[1::2] = -TabRes_neg[1::2] # flip sign for odd functions
        if Npoints%2 == 0 :
            TabRes = np.concatenate((TabRes_neg,TabRes),1)
        else:
            TabRes = np.concatenate((TabRes_neg,TabRes[:,1:]),1) # don’t repeat zero value
        self.TabPolys = TabRes
        return

    def Eval(self, vector:np.ndarray, indexk:int):
        """computes the polynomial on any vector, not necessarily equally spaced terms on a [-1,1] interval;
        requires prior initialization or recomputing setUnitVector to get the calibration values"""
        return fast_polyvalfromroots(vector, self.RootsTable[indexk])*self.CalibsTable[indexk]

    def EvalAll(self, vector:np.ndarray):
        """computes the polynomials on any vector, not necessarily equally spaced terms on a [-1,1] interval;
        requires prior initialization or recomputing setUnitVector to get the calibration values"""
        TabRes = np.zeros((self.OrderN+1,len(vector)))
        for k in range(self.OrderN+1):
            TabRes[k,:] = self.Eval(vector, k)
        return TabRes

    def ScalarProduct(self, ind1:int, ind2:int):
        """ product of two Legendre polynomials defined by their indices with Gauss-Legendre quadrature"""
        feval1 = self.Eval(self.GLquad_roots, ind1)
        feval2 = self.Eval(self.GLquad_roots, ind2)
        prod   = self.GLquad_weights * feval1 * feval2
        return np.sum(prod)        
    
    def Norm(self, indk):
        """ norm of a Legendre Polynomial defined by its index"""
        return self.ScalarProduct(indk, indk)