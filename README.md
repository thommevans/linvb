linvb
=====

A Python package for variational Bayes linear regression.

This code is currently a bit outdated and various things probably need to be fixed up, eg. importing some external packages may be broken.

The three core modules are:

vbr_class.py - Contains the definition for the VB class of object, including its attributes and methods.
vbr_routines.py - Contains the actual implementation of the VB algorithm.
vbr_utilties.py - Contains miscellaneous routines, most of which relate to constructing the basis matrix.
The other two modules are:

vbr_basis_functions.py - Contains definitions for some commonly used basis functions in the format that they are required by the VB object methods when the basis matrix is constructed.
vbr_planet.py - Specialised routines for including transit and eclipse basis functions. Requires the additional planetc package.
