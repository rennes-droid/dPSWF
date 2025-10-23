# -*- coding: utf-8 -*-
"""
Created on October 2025

Installation file
"""

from setuptools import setup, find_packages

setup(
    name='dPSWF',
    version='0.1.0',
    description='Expansion of PSWF on Legendre polynomials',
    author='F. Bondu',
    packages=find_packages(),  # trouve automatiquement le dossier my_pswf
    install_requires=[
        'numpy',
        'scipy',
        'numba',
        'typing'
    ],
    python_requires='>=3.8',
)