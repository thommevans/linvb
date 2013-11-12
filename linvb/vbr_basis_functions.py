import numpy as np
import math
import pdb
import matplotlib.pyplot as plt


###########################################################################
#
# This module holds the definitions for various basis functions that
# are routinely used in variational Bayesian regression applications.
# The functions must be defined according to the following format:
#
#   1. The first argument must receive **all** non-parametric inputs.
#      Hence, if the function accepts multivariate input, then it must
#      be passed in as the first argument in the form of an NxD array,
#      where N is the number of data points and D is the number of
#      different input variables.
#
#   2. All remaining arguments must be keyword arguments (**kwargs).
#
###########################################################################


def poly1d( x, order=None, func=None ):
    """
    1D nth-order polynomial.

    It is possible to pass a function in as a keyword argument to operate
    on the inputs before returning the polynomial. To be explicit, suppose
    func=np.log and order=2, then the output would be:

      output = [ np.log( x ) , np.log10( x**2. ) ]

    The default behaviour is to not apply any such transformation to the
    input, so func=None and order=2 returns:

      output = [ x , x**2 ]

    Note that a constant offset is not added, as this is done separately
    via the model_add_offset attribute and we don't want duplicate columns
    of 1s in the basis matrix.
    """
    outarray = np.zeros([len(x),order])
    if func!=None:
        x = func(x)
    for i in range(1,order+1):
        outarray[:,i-1] = x**i
    return outarray


def local_poly1d( x, order=None, ixs=None, func=None ):
    """
    1D nth-order polynomial, the same as poly1d() except that it ignores all
    data outside [xmin,xmax].
    """
    n = len(x)
    outarray = np.zeros( [n, 2] )
    outarray[:,0] = outarray[:,0]+1.0
    outarray_ixs = polynomial1d(x[ixs], order=order, func=func)
    outarray[ixs,1] = outarray_ixs[:,0]
    return outarray


def poly2d_crossterms( xy, order=None, func=None ):
    """
    Returns the polynomial crossterms between two variables, going from 
    (x**0)*(y**order) through to (x**order)*(y**0).

    For the remaining terms that are not crossterms, poly1d should be used.
    i.e. poly1d( x, order=ored, func=func ) and poly1d( y, order=order, func=func )
    """
    outarray = np.empty([np.shape(xy)[0], order+1])
    x = xy[:,0]
    y = xy[:,1]
    if func!=None:
        x = func(x)
        y = func(y)
    for i in range(order+1):
        outarray[:,i] = (x**i)*(y**(order-i))
    return outarray


def gaussian1d( x, means=None, widths=None ):
    """
    Returns an Nxn array where N is the number of scalar data points where n
    1D Gaussians with individually specified means and widths are evaluated.

    Note that points beyond 5 sigma of the mean are assumed to make zero
    contribution to the model and so are not evaluated to save time, which
    can be significant for large datasets.
    """
    ndata = len( x )
    nfuncs = len( means )
    print 'Evaluating %i 1D Gaussian functions (only at locations within 5 sigma of mean)' % ( nfuncs )
    outarray = np.zeros( [ ndata, nfuncs ] )
    for i in range( len( means ) ):
        ixs = ( abs( x-means[i] ) < 5*widths[i] ) # only bother evaluating within 5-sig of mean
        outarray[ixs,i] = np.exp( -( ( x[ixs]-means[i] )**2. ) / 2. / ( widths[i]**2. ) )
    return outarray


def gaussian2d( xy, means=None, widths=None ):
    """
    Returns an Nxn array where N is the number of 2D data points and where n
    is the number of 2D Gaussians with individually specified means and scalar
    widths.

    Note that the widths are the same along both axes (i.e. the same for both
    input variables). For this reason, the input variables should probably be
    standardised before being passed to this function, especially if their
    units are significantly different.
    .
    """
    ndata = np.shape(xy)[0]
    nfuncs = np.shape(means)[0]
    outarray = np.zeros([ndata, nfuncs])
    x = xy[:,0]
    y = xy[:,1]
    print 'Evaluating %i 2D Gaussian functions (only at locations within 5 sigma of mean)' % (nfuncs)
    xmeans = means[:,0]
    ymeans = means[:,1]
    for i in range(len(xmeans)):
        ixs = ((abs(x-xmeans[i])<5*widths[i])*\
               (abs(y-ymeans[i])<5*widths[i])) # only bother evaluating within 5-sig of mean
        outarray[ixs,i] = np.exp( -((x[ixs]-xmeans[i])**2.+(y[ixs]-ymeans[i])**2.)/2./(widths[i]**2.) )
    return outarray


def harmonics( x, n=None, period0=None ):
    """
    Returns an N x 2n array where N is the number of data points and n is the number
    of harmonics. To be explicit, the output will be:

      output = [ sin( 1*(2*np.pi/period0)*x) , 1*cos( (2*np.pi/period0)*x),
                 sin( 2*(2*np.pi/period0)*x) , 2*cos( (2*np.pi/period0)*x),
                 ...
                 sin( n*(2*np.pi/period0)*x) , n*cos( (2*np.pi/period0)*x) ]

    where each entry in the above is a column vector.
    """
    outarray = np.empty([len(x),2*n], dtype=float)
    w0 = 2*np.pi/period0
    for i in range(n):
        w = float((i+1))*w0
        outarray[:,i] = np.sin(w*x)
        outarray[:,i+n] = np.cos(w*x)
    return outarray


def sinusoids(x, pmin=None, pmax=None, n=None):
    """
    Returns an N x 2n array containing sinusoids (cos and sin) with periods evenly 
    spaced between pmin and pmax, where N is the number of data points and n is the
    number of sinusoids. To be explicit, the output will be:

      output = [ sin( (2*np.pi/period1)*x) , cos( (2*np.pi/period1)*x),
                 sin( (2*np.pi/period2)*x) , cos( (2*np.pi/period2)*x),
                 ...
                 sin( (2*np.pi/periodn)*x) , cos( (2*np.pi/periodn)*x) ]

    where each entry in the above is a column vector, and period1=pmin and periodn=pmax.

    Note that pmin and pmax are assumed to have the same units as the input x.
    """
    outarray = np.empty([len(x),2*n], dtype=float)
    ps = np.r_[pmin:pmax:1j*n]
    for i in range(n):
        outarray[:,i] = np.sin(2*np.pi*x/ps[i])
        outarray[:,i+n] = np.cos(2*np.pi*x/ps[i])
    return outarray


def boxcar( x, a=None, b=None, d=None ):
    """
    Simple boxcar function, set to zero everywhere except a<x<b where it is set
    to d. Returns an Nx1 array.
    """
    outarray = np.zeros(len(x))
    ixs = ((x>a)*(x<b))
    outarray[ixs] = d
    outarray = np.reshape(outarray, [len(outarray),1])
    return outarray
