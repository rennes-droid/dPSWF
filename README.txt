########
# GOAL #
########

computation of prolate spheroidal wave functions 
by expansion on Legendre polynomials
in Python language

- significant improvement in speed compared to DPSS with only slightly compiled code.
Code optimization is likely possible.
- DPSS computed with tridiagonal matrices are not the best confinment windows
- calculation of 1-\lamda with concentration integration more precise
when n < Integer(NW/2) compared to DPSS (Percival Walden) and Wang methods

##################
# MAIN FUNCTIONS  #
##################

Calculation of Legendre polynomials on max N+1 polynomials at once
in dPSWF/PolyLegendre_Roots.py
returns an object where roots, calibration factors, evaluation functions

Calculation of all PSWF up to order OrderN at once on a given vector on [-1,1]
in dPSWF/PSWF_Legendre
returns an object with psi functions, lambda values, 
and functions for example to compute the values on other intervals

##################
# CODE DIRECTORY #
##################

/dPSWF : the two main functions of the package

/test : test of the software

/perf : performance evaluation
/perf/utils : functions to compute Fourier transforms, confinement values,
noise equivalent bandwidth of a window

/comparisons : comparison with dpss, with other windows in Welch spectrum evaluation

/dev : used for the software development

/examples : use cases

#############
INSTALLATION
############

with pip 

F. Bondu, Institut FOTON, CNRS/Université de Rennes, 2025