# -*- coding: utf-8 -*-
"""
Generation of discretized Prolate Spheroidal Wave Functions (PSWF)

The traditional calculation of the Discrete Prolate Spheroidal Sequence (DPSS) of length N is 
by the eigenvectors and eigenvalues of a tridiagonal NxN matrix, cf Percival section 8.3
(ex of scipy.signal.windows.dpss).

Instead, use Bouwkamp (1947, J. Math. Phys.) decomposition on Legendre polynomials.
With polynomials up to order K, the eigenvectors and eigenvalues of a (2K+30)*(2K+30) tridiagonal matrix
must be computed. 
See details in 
    Wang           2017 J Math study https://doi.org/10.4208/jms.v50n2.17.01
    Osipov Rokhlin 2013 ArXiv        http://arxiv.org/abs/1301.1707
The evaluation of Legendre polynomials of order > 80 requires defining the polynomials by the roots
The evaluation on sequences where data point number N can be very large (hundred of thousands, or millions) is not a problem

F. Bondu Apr 2025

"""

import numpy       as     np
from dPSWF         import PolyLegendreAllFromRoots
from scipy.linalg  import eigh_tridiagonal
from scipy.special import roots_legendre
from numba         import njit
from typing        import Literal

@njit # Wang equation 3.6 compiled version for speed
def fast_multiPsiEval(TabPsi,MatBeta,TabPolyLeg,N_lines,N_Polys):
    for k in range(N_lines):
        for j in range(N_Polys):
            TabPsi[k] = TabPsi[k] + MatBeta[j,k]*TabPolyLeg[j,:]
    return TabPsi

class PSWF:
    """
Input parameters:
    c time-bandwidth parameter = pi * NW, where NW is the frequency resolution in bins
    OrderN : order for maximum PSWF to be calculated (first OrderN+1 functions calculated)
    Number of equidistants points on [-1,1] interval 
    Normalization : 'L2' or 'window'

return object with accessible variables:
    c             # gain bandwidth factor
    OrderN        # max order of PSWF
    UnitVec       # vector or value onto evaluate PSWF’s
    psi_functions # array of psi functions, first index function number, second index x vector index
    lambdas       # vector for eigenvalues, as defined in Cada ; corresponds to mu_n in Wang paper
and accessible functions:
    Eval(x, k): evaluation on a single point or an array of function k =< OrderN
    EvalPsiAll(x): evalation of all functions <= OrderN at a single value or on an array
    get_psi_k(k): return psi of index k on UnitVec
    set_normalization: to switch between normalization types
    Norm_LG(k): norm of the function on [-1,1] interval, calculated with LG integration
"""

    # internal variables
    Pn      = None #  All Legendre Polynomials object
    MatBeta = None # beta matrix / decomposition on Legendre polynomials
    QuadGL_n        = 0  # index for Gauss-Legendre integration
    QuadGL_roots    = 0
    QuadGL_weights  = 0
    QuadGL_PSWF_tab = 0
    normalizationType = 'L2' # 'L2…(default): PSWF such that its square module integrated on [-1,1] is 1
                      # 'window': sum of square module for defined x values is equal to vector x length

    def __init__(self, c:float, OrderN:int, UnitVecNpoints:int,
                 normalizationType:Literal['L2','window']='L2'):
        self.c      = c # time-bandwidth parameter
        self.OrderN = OrderN
        # number of Legendre polynomials to compute
        OrderM      = 2*OrderN + 30# cf Wang between 3.6 and 3.7
        # internal normalized Legendre polynomials
        self.Pn     = PolyLegendreAllFromRoots(OrderM, UnitVecNpoints, normalized=True)
        # unit vector onto which compute the Slepian functions
        self.UnitVec = self.Pn.VecUnit
        Nhalf = UnitVecNpoints//2
        HalfVec = self.UnitVec[Nhalf:]
        # compute Beta_n_k matrix
        self._PSWF_Legendre_expansion()
        # have posivive value at zero for even orders 
        # and positive derivative at zero for odd orders        
        self._flip_sign_PSWF()
        # prepare for normalization with Gauss-Legendre integration
        self.QuadGL_n = 2*OrderM + 2
        self.QuadGL_roots, self.QuadGL_weights = roots_legendre(self.QuadGL_n)
        # memorize value of psi functions with L2 norm on Gauss-Legendre points Sonce for all
        self.QuadGL_PSWF_tab = self.EvalPsiAll(self.QuadGL_roots)
        self.lambdas = self._Lambdas()
        psi_functions = np.zeros((self.OrderN+1,len(HalfVec)))
        fast_multiPsiEval(psi_functions,self.MatBeta,self.Pn.TabPolys[:,Nhalf:],self.OrderN+1,self.Pn.OrderN+1)
        # for k in range(self.OrderN+1):
        #     for j in range(self.Pn.OrderN+1):
        #         psi_functions[k] = psi_functions[k] + self.MatBeta[j,k]*self.Pn.TabPolys[j,Nhalf:]
        psi_functions_neg = np.copy(np.flip(psi_functions,1)) # image for negative x values
        psi_functions_neg[1::2] = -psi_functions_neg[1::2] # flip sign for odd functions
        if UnitVecNpoints%2 == 0 :
            psi_functions = np.concatenate((psi_functions_neg,psi_functions),1)
        else:
            psi_functions = np.concatenate((psi_functions_neg,psi_functions[:,1:]),1) # don’t repeat zero value
        self.psi_functions = psi_functions
        self.normalizationType = 'L2'
        self.set_normalization(normalizationType)
        return 

    def EvalPsiAll(self, vector_x):
        # returns a table with all Psi functions evaluated on vector_x
        TypeIn = type(vector_x) # if not np.ndarray, is an int or float
        if  TypeIn != np.ndarray: # single value
            vector_x = np.array([vector_x])
        TabPolyLeg = self.Pn.EvalAll(vector_x)
        psi_functions = np.zeros((self.OrderN+1,len(vector_x)))
        fast_multiPsiEval(psi_functions,self.MatBeta,TabPolyLeg,self.OrderN+1,self.Pn.OrderN+1)
        if TypeIn != np.ndarray:
            psi_functions = psi_functions[:,0]
        return psi_functions

    def Eval(self, vector_x, indexk):
        TypeIn = type(vector_x) # if not np.ndarray, is an int or float
        if  TypeIn != np.ndarray: # single value
            vector_x = np.array([vector_x])
        TabPolyLeg = self.Pn.EvalAll(vector_x)
        psi_function = np.zeros((1,len(vector_x)))
        for j in range(self.Pn.OrderN+1):
            psi_function = psi_function + self.MatBeta[j,indexk]*TabPolyLeg[j,:]
        if  TypeIn == np.ndarray: # array
            psi_function = psi_function[0]
        else:
            psi_function = psi_function[0][0]    
        return psi_function

    def get_psi_k(self, indexk):
        #  returns only one psi function already calculated
        return self.psi_functions[indexk]

    def PdtScal_LG_L2(self, ind1_psi,ind2_psi):
        g1 = self.QuadGL_PSWF_tab[ind1_psi,:]
        g2 = self.QuadGL_PSWF_tab[ind2_psi,:]
        return np.sum(g1*g2*self.QuadGL_weights)

    def set_normalization(self, norm:Literal['L2','window']='L2'):
        # normalization change
        if (self.normalizationType == 'L2') and (norm=='window'):
            # remember the normalization for L2 norm
            self.psi_L2Norms = np.sum(self.psi_functions**2,1) # norm along columns
            calibs = np.reshape(self.psi_L2Norms,(-1,1))
            M = len(self.UnitVec)
            calibs = np.repeat(calibs,M,1)
            self.psi_functions = self.psi_functions / np.sqrt(calibs)
        elif (self.normalizationType == 'window') and (norm == 'L2'):
            calibs = np.reshape(self.psi_L2Norms,(-1,1))
            M = len(self.UnitVec)
            calibs = np.repeat(calibs,M,1)/M
            self.psi_functions = self.psi_functions * np.sqrt(calibs) 
        self.normalizationType = norm
        return

    def Norm_LG_L2(self, indn):
        return self.PdtScal_LG_L2(indn,indn)

    def _PSWF_Legendre_expansion(self): # wang eq 3.6 and 3.4
    # computation of beta matrix Wang eq 3.3
    # for the expansion of all psi_n (from n=0 to orderM) on normalized Legendre polynomials
    # see Wang J. Math. Study 50(2017) pp 101-143
    # expansion parameters are eigenvectors of a tri-diagonal matrix
    # orderM is assumed to be even, for example M = 2N+30 
        OrderPe = self.Pn.OrderN//2
        a_kk   = np.zeros(OrderPe+1, dtype=float)
        for k in range(0,OrderPe+1):
            a_kk[k] = self._akk(2*k)
        a_kk2 = np.zeros(OrderPe, dtype=float)
        for k in range(0,OrderPe):
            a_kk2[k] = self._akk2(2*k)
        _, eigvecs_e = eigh_tridiagonal(a_kk, a_kk2) # eigenvectors are on columns !    

        OrderPo = self.Pn.OrderN//2 - 1
        a_kk   = np.zeros(OrderPo+1, dtype=float)
        for k in range(0,OrderPo+1):
            a_kk[k] = self._akk(2*k+1)
        a_kk2 = np.zeros(OrderPo, dtype=float)
        for k in range(0,OrderPo):
            a_kk2[k] = self._akk2(2*k+1)
        _, eigvecs_o = eigh_tridiagonal(a_kk, a_kk2) # eigenvectors are on columns !    
        # interleave columns of zeroes in even tab
        eigvecs_e2 = np.zeros((OrderPe+1,self.Pn.OrderN+1))
        eigvecs_e2[:,::2] = eigvecs_e
        # interveave columns of zeroes in odd tab
        eigvecs_o2 = np.zeros((OrderPo+1,self.Pn.OrderN+1))
        eigvecs_o2[:,1::2] = eigvecs_o
        # construct beta_nk matrix
        self.MatBeta = np.zeros((self.Pn.OrderN+1,self.Pn.OrderN+1))
        self.MatBeta[::2,:]  = eigvecs_e2
        self.MatBeta[1::2,:] = eigvecs_o2
        return 

    def _akk(self,k:int):
        return k*(k+1) + (2*k*(k+1)-1)/((2*k+3)*(2*k-1)) * self.c**2

    def _akk2(self,k:int):
        return (k+1)*(k+2)/( (2*k+3) * np.sqrt((2*k+1)*(2*k+5))) * self.c**2

    def _flip_sign_PSWF(self):
        # even PSWF  have positive value at x = zero
        Psi_0 = self.EvalPsiAll(0.)
        for k in range(0,self.OrderN+1,2):
            if Psi_0[k] < 0:
                self.MatBeta[:,k] = -self.MatBeta[:,k]
        # odd functions have positive derivative at zero
        for k in range(1,self.OrderN+1,2):
            if self._deriv_atZero(k) < 0:
                self.MatBeta[:,k] = -self.MatBeta[:,k]
        return

    def _deriv_atZero(self, index_n):
        vec_kplus1  = np.arange(1,self.Pn.OrderN+1)
        beta_vec = self.MatBeta[0:-1,index_n]
        Legkplus1     = self.Pn.EvalAll(np.array([0.]))[1:,0]
        # roll vector of Legendre polynomials at zero
        # P’(n,x=0) = - P(n+1,x=0) * (n+1) according to recurrence rule with Legendre polynomials
        return -sum( beta_vec * vec_kplus1 * Legkplus1 * np.sqrt( (vec_kplus1-0.5)/(vec_kplus1+0.5) ) )

    def _Lambdas(self):
        LambdaVec = np.zeros(self.OrderN+1)
        for k in np.arange(self.OrderN+1):
            if k%2 == 0: # Wang eq 3.9
                LambdaVec[k] = (-1)**(k//2) * np.sqrt(2)*self.MatBeta[0,k] \
                    / self.Eval(0,k)
            else: # Wang eq 3.10
                LambdaVec[k] = (-1)**(k//2) * np.sqrt(2/3)*self.c*self.MatBeta[1,k] \
                    / self._deriv_atZero(k)
        MuVec = LambdaVec**2 * self.c/(2*np.pi)
        if MuVec[0]>1:
            MuVec[0] = 1 - np.finfo(float).eps # epsilon machine below 1
        # Lambda Values (mu values in Wang paper) are always below 1,
        # and always in decreasing order.
        # The following for loop corrects for (small) numerical errors
        for k in range(1,len(MuVec)):
            if MuVec[k]>=MuVec[k-1]:
                MuVec[k]=MuVec[k-1] -  np.finfo(float).eps
        return MuVec