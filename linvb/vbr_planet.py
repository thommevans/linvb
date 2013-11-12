try:
    from MyFunctions import EccLightCurveModels
except:
    print '\nCould not import EccLightCurveModels.py from the MyFunctions module!\n'
try:
    from planet import orbit
except:
    print '\nCould not import orbit.py from the planet module!\n'
import matplotlib.pyplot as plt
import numpy as np
import pdb, sys
EPS_MIN = sys.float_info.min


#################################################################################
#
#  This is a specialised module to allow planet-related basis functions, such as
#  transits/eclipses, phase curves etc, to be appended to pre-normalised basis
#  matrices.
#
#################################################################################


def append_transit_function( vbr_object, time, system_params, transit_type=None, transit_function='ma02' ):
    """
    Appends a transit or eclipse basis function to the existing normalised training
    basis matrix, and records the appended array, so that it can be automatically
    appended to the normalised predictive basis matrix when that's constructed later.
    The basis function is set to zero out-of-transit, and is negative within transits
    and eclipses.

    Because we want to vary their depths independently, we must construct separate 
    basis functions for the primary transits and secondary eclipses, which is what 
    the transit_type keyword argument allows. Specifically, it accepts the following
    two options:

    **'primary' - a basis function containing only primary transits, with zeros where
      the secondary eclipses should occur.
    **'secondary' - a basis function containing only secondary eclipses, with zeros
      where the primary transits should occur.

    The content of the system_params input array depends on the type of limb darkening:
      [ T0, P, aRs, RpRs, b, c1, c2, ecc, omega ]          ... quadratic 
      [ T0, P, aRs, RpRs, b, c1, c2, c3, c4, ecc, omega ]  ... 4-parameter nonlinear
    The length of the array is used to determine which type of limb darkening is to be
    used.

    Note that b=a*cosi/Rs and omega is the argument of periastron, which equals 90degrees
    for a circular orbit by convention, if T0 is to be the time of mid-transit. Do not
    confuse omega here with the longitude of the ascending node Omega, which is set to
    180degrees by convention.

    Also note that no value is provided for the secondary flux ratio, because this only
    affects the depth of the secondary eclipse. Because this depth is free to vary in
    the fitting, we set it to an arbitrary value here if transit_type=='secondary', and
    then standardise it so that the basis function has a comparable amplitude to all
    the other basis functions.

    On a practical matter, if you only have the time of central transit or eclipse,
    you must convert it to T0 before passing into this function. If the orbit is
    circular, this can be done trivially by setting T0=Tmid and omega=np.pi/2. For a
    non-circular orbit, it is less straightforward, but can be done using the get_t0()
    function in the orbit.py module.
    """
    print '\nAppending %s transit basis function...' % transit_type
    if transit_function=='ma02':
        transit_shape = ma02( time, system_params, transit_type=transit_type )
    elif transit_function=='piecewise_tr':
        transit_shape = piecewise_tr( time, system_params, transit_type )
    # Ensure that the basis function is zero out of transit:
    if vbr_object.target_log_units==False:
        transit_basis = transit_shape-1
    else:
        transit_basis = np.log( transit_shape )
    # Normalise the transit function to the same level as the target data;
    # by doing this, we are implying that the parameters for the transit
    # function represent our best guess; note that if we're attempting to
    # detect a faint signal, it's best to overestimate its depth otherwise
    # the VB algorithm is reluctant to move the transit depth away from
    # zero even if there's a faint signal present:
    transit_basis = transit_basis / vbr_object.target_train_norm_scaling
    if transit_type=='primary':
        appendage_name = 'Primary Transit'
    elif transit_type=='secondary':
        appendage_name = 'Secondary Eclipse'
    vbr_object.append_training_basis_matrix( transit_basis, appendage_name=appendage_name )
    # Record the column index containing the transit basis function:
    if transit_type=='primary':
        vbr_object.transit_basis_ix = vbr_object.phi_ixs_appendages_postnorm[-1]
    elif transit_type=='secondary':
        vbr_object.eclipse_basis_ix = vbr_object.phi_ixs_appendages_postnorm[-1]
    return None


def ma02( time, system_params, transit_type ):
    """
    This evaluates transit and eclipse functions using the analytic formulas
    of Mandel & Agol 2002. To do this, it uses the EccLightCurve_aRs() function
    in the EccLightCurveModels.py module.

    If transit_type=='primary', then a light curve with only the primary transits
    will be returned. Alternatively, if transit_type=='secondary', a light curve
    with only the secondary eclipses will be returned.
    """
    # Work out if we're using quadratic or 4-parameter nonlinear limb darkening:
    if len(system_params)==12:
        T0, P, aRs, RpRs, b, c1_q, c2_q, ecc, omega, foot, Tgrad, sec_depth = system_params
        # Just in case, force the limb darkening to be zero if it's an eclipse:
        if transit_type=='secondary':
            c1_q = c2_q = 0.0
        system_params_arr = np.array([ T0, P, aRs, RpRs, b, c1_q, c2_q, ecc, omega, \
                                       foot, Tgrad, sec_depth ])
        ld_type = 'quad'
    elif len(system_params)==14:
        T0, P, aRs, RpRs, b, c1_nl, c2_nl, c3_nl, c4_nl, ecc, omega, foot, Tgrad, sec_depth = system_params
        # Just in case, force the limb darkening to be zero if it's an eclipse:
        if transit_type=='secondary':
            c1_nl = c2_nl = c3_nl = c4_nl = 0.0
        system_params_arr = np.array([ T0, P, aRs, RpRs, b, c1_nl, c2_nl, c3_nl, c4_nl, \
                                       ecc, omega, foot, Tgrad, sec_depth ])
        ld_type = 'nonlin'
    else:
        pdb.set_trace() #this shouldn't happen
    # Evaluate the transit function:
    transit_shape, ycoord = EccLightCurveModels.EccLightCurve_aRs( system_params_arr, time, \
                                                                   return_y=True, ld=ld_type )
    if transit_type=='primary':
        transit_shape[ycoord>0] = 1 # i.e. when planet further away than star
    elif transit_type=='secondary':
        transit_shape[ycoord<=0] = 1 # i.e. when planet closer than star
    return transit_shape


def piecewise_tr( time, system_params, transit_type ):
    """
    This evaluates transit and eclipse functions using the piece-wise linear
    approximation of Carter et al 2009. The ingress-egress times and transit
    duration are approximated. Note of course that the flat-bottomed function
    prevents a straightforward mapping of the fitted signal back to Rp/Rs etc
    for a primary transit due to the degeneracy of limb darkening which is not
    accounted for here at all.

    If transit_type=='primary', then a light curve with only the primary transits
    will be returned. Alternatively, if transit_type=='secondary', a light curve
    with only the secondary eclipses will be returned.
    """
    if (transit_type!='primary')*(transit_type!='secondary'):
        pdb.set_trace()
    # Unpack the system parameters, but be careful to account for the possibility that
    # we could be using either a quadratic or 4-parameter nonlinear limb darkening law,
    # as that will affect the number of system parameters that have been passed in:
    if len(system_params)==12:
        T0, P, aRs, RpRs, b, c1_q, c2_q, ecc, omega, foot, Tgrad, sec_depth = system_params
    elif len(system_params)==14:
        T0, P, aRs, RpRs, b, c1_nl, c2_nl, c3_nl, c4_nl, ecc, omega, foot, Tgrad, sec_depth = system_params
    else:
        pdb.set_trace() #this shouldn't happen
    # Calculate the true anomalies of the times of central transit and the times of
    # central transit:
    f_tr = - np.pi / 2.0-omega
    f_ec = + np.pi / 2.0-omega
    # Now we proceed to calculate the corresponding eccentric anomalies, using the
    # standard relation between the true anomaly and the eccentric anomaly:
    cosf_tr = np.cos( f_tr )
    cosf_ec = np.cos( f_ec )
    cosean_tr = ( ecc + cosf_tr ) / ( 1. + ecc*cosf_tr )
    cosean_ec = ( ecc + cosf_ec ) / ( 1. + ecc*cosf_ec )
    ean_tr = np.arccos( cosean_tr )
    ean_ec = np.arccos( cosean_ec )
    # Convert the eccentric anomaly to the mean anomaly using the simple formula:
    man_tr = ean_tr - ecc*np.sin( ean_tr )
    man_ec = ean_ec - ecc*np.sin( ean_ec )
    # Given the definition of the mean anomaly as the fraction of the orbital period
    # that has passed since periastron, calculate the time since periastron:
    delt_tr = ( P*man_tr ) / ( 2*np.pi )
    delt_ec = ( P*man_ec ) / ( 2*np.pi )
    # Hence, calculate the times of transit and eclipse:
    t_tr = T0+delt_tr
    t_ec = T0+delt_ec
    # Use Equations 6-10 of Carter et al 2008 to calculate the key parameters that
    # define the transit shape:
    n = 2.*np.pi/P
    b0 = b * ( ( 1. - ecc**2. )/( 1. + ecc*np.sin( omega ) ) )
    tau0 = ( ( np.sqrt( 1. - ecc*2. ) )/( 1 + ecc*np.sin( omega ) ) ) / n / aRs
    bigT = 2 * tau0 * np.sqrt( 1-b0**2. )
    tau = 2. * tau0 * RpRs / np.sqrt( 1.-b0**2. )
    # We have now calculated the parameters that define our approximated shape for
    # the transits/eclipses. Next, we need to find the times of transit and eclipse
    # that occured immediately before our time series (in case there is some overlap
    # of a partially-complete transit right at the start of our time series, with the
    # central time actually falling before the start):
    if t_tr<time.min():
        while t_tr<time.min():
            t_tr += P
        t_tr -= P
    else:
        while t_tr>time.min():
            t_tr -= P
    if t_ec<time.min():
        while t_ec<time.min():
            t_ec += P
        t_ec -= P
    else:
        while t_ec>time.min():
            t_ec -= P
    # Build up a list of the transit and eclipse times until the one immediately
    # after the end of our time series:
    t_trs = [t_tr]
    while t_tr<time.max():
        t_tr += P
        t_trs += [ t_tr ]
    t_ecs = [ t_ec ]
    while t_ec<time.max():
        t_ec += P
        t_ecs += [ t_ec ]
    # Construct the approximated light curve one transit and one eclipse at a time:
    f = np.ones( len( time ) )
    tr_depth = RpRs**2.
    if transit_type!='secondary':
        for t_tr in t_trs:
            # Full transit times:
            ixs_23 = abs( time-t_tr ) < 0.5*bigT - 0.5*tau
            f[ixs_23] = f[ixs_23] - tr_depth
            # Ingress times:
            t_1 = t_tr - 0.5*bigT - 0.5*tau
            t_2 = t_tr - 0.5*bigT + 0.5*tau
            ixs_12 = ( ( time>t_1 ) * ( time<t_2 ) )
            f[ixs_12] = f[ixs_12] - tr_depth + ( tr_depth/tau )*( -time[ixs_12] + t_tr - 0.5*bigT + 0.5*tau ) 
            # Egress times:
            t_3 = t_tr + 0.5*bigT - 0.5*tau
            t_4 = t_tr + 0.5*bigT + 0.5*tau
            ixs_34 = ( ( time>t_3 ) * ( time<t_4 ) )
            f[ixs_34] = f[ixs_34] - tr_depth + ( tr_depth/tau )*( +time[ixs_34] - t_tr - 0.5*bigT + 0.5*tau ) 
    if transit_type!='primary':
        for t_ec in t_ecs:
            # Full transit times:
            ixs_23 = abs( time-t_ec ) < 0.5*bigT - 0.5*tau
            f[ixs_23] = f[ixs_23] - sec_depth
            # Ingress times:
            t_1 = t_ec - 0.5*bigT - 0.5*tau
            t_2 = t_ec - 0.5*bigT + 0.5*tau
            ixs_12 = ( ( time>t_1 ) * ( time<t_2 ) )
            f[ixs_12] = f[ixs_12] - sec_depth + ( sec_depth/tau )*( -time[ixs_12] + t_ec - 0.5*bigT + 0.5*tau ) 
            # Egress times:
            t_3 = t_ec + 0.5*bigT - 0.5*tau
            t_4 = t_ec + 0.5*bigT + 0.5*tau
            ixs_34 = ( ( time>t_3 ) * ( time<t_4 ) )
            f[ixs_34] = f[ixs_34] - sec_depth + ( sec_depth/tau )*( +time[ixs_34] - t_ec - 0.5*bigT + 0.5*tau )
    return f


def calc_transit_depth( vbr_object, transit_type='primary', print_precision=1e-6 ):
    """
    Uses the posterior distribution over the transit/eclipse function to calculate the
    inferred depth and associated uncertainty. The transit_function_ix specifies which
    column in the basis matrix corresponds to the transit/eclipse basis function.
    """
    # Find the in-transit data points:
    if transit_type=='primary':
        ix = vbr_object.transit_basis_ix
    elif transit_type=='secondary':
        ix = vbr_object.eclipse_basis_ix
    # Extract the transit basis column from the normalised basis matrix, and unnormalise
    # it, making sure to use the same factor as was used in append_transit_function():
    transit_basis = vbr_object.phi_pred_norm[:,ix].flatten() * vbr_object.target_train_norm_scaling
    intransit_ixs = np.arange( vbr_object.n_data_pred )[transit_basis!=0]
    n_in = len(intransit_ixs)
    if n_in>0:
        # Treat the inferred distributions over the weights as scaled normal random variables,
        # where the scaling factor is the value of the transit basis function at each of the
        # in-transit points:
        mu_x = ( transit_basis * vbr_object.model_weights_means[ix] )[intransit_ixs]
        sig_x = np.sqrt( ( ( transit_basis * vbr_object.model_weights_stdvs[ix] )[intransit_ixs] )**2. )
        # Now we need to make sure these distributions can be translated back to the native units
        # of the target data:
        # If we were fitting to the data in its native units, then we're done:
        if vbr_object.target_log_units==False:
            if mu_x.min()<0:
                inferred_depth_mean = 0 - mu_x[ np.argmin( mu_x ) ]
            else:
                inferred_depth_mean = 0 - mu_x[ np.argmax( mu_x ) ]
            inferred_depth_median = inferred_depth_mean
            inferred_depth_mode = inferred_depth_mean
            uncertainty = sig_x.max()
        # Otherwise, if we're working with a multiplicative model (i.e. in log flux), then we
        # have a little more work, because the posterior is a log-normal distribution:
        elif vbr_object.target_log_units==True:
            mean_tr = np.zeros( n_in )
            median_tr = np.zeros( n_in )
            mode_tr = np.zeros( n_in )    
            sig_tr = np.zeros( n_in )
            for i in range(n_in):
                var_x = sig_x[i]**2.
                var_tr = ( np.exp( var_x ) - 1.0 ) * np.exp( 2*mu_x[i] + var_x )
                sig_tr[i] = np.sqrt( var_tr )
                mean_tr[i] = np.exp( mu_x[i] + var_x/2. )
                median_tr[i] = np.exp( mu_x[i] )
                mode_tr[i] = np.exp( mu_x[i] - var_x )
            if mean_tr.min()<0:
                inferred_depth_mean = 1. - mean_tr[ np.argmin( mean_tr ) ]
                inferred_depth_median = 1. - median_tr[ np.argmin( mean_tr ) ]
                inferred_depth_mode = 1. - mode_tr[ np.argmin( mean_tr ) ]
            else:
                inferred_depth_mean = 1. - mean_tr[ np.argmax( mean_tr ) ]
                inferred_depth_median = 1. - median_tr[ np.argmax( mean_tr ) ]
                inferred_depth_mode = 1. - mode_tr[ np.argmax( mean_tr ) ]
            uncertainty = sig_tr.max()
            # NOTE: In general, the mean, median and mode of a log-normal distribution do
            # not coincide. We're counting on the fact that they roughly do here for the
            # mean and standard deviation of the transit depth to be meaningful.
        else:
            pdb.set_trace() #this shouldn't happen
        # The stuff below here is to format the printed output nicely; hopefully it
        # does what it's supposed to, but if the printed results look strange, may
        # need to check here that it's not simply a bug in the output formatting:
        if ( ( inferred_depth_mean>(1e-4) ) + ( print_precision>1e-4) ):
            units = 'percent'
            factor = 1e2
        else:
            units = 'ppm'
            factor = 1e6
        dec_places = str( int( -np.log10( factor*print_precision ) ) ) ### here is where you need to think about it
        format_string = '%.'+dec_places+'f'
        mean = format_string % ( inferred_depth_mean * factor )
        median = format_string % ( inferred_depth_median * factor )
        mode = format_string % ( inferred_depth_mode * factor )
        uncert = format_string % ( uncertainty * factor )
        print '\n  Inferred %s transit depth:' % ( transit_type )
        print '        %s +/- %s %s' % ( mean, uncert, units )
        print '      (median = %s, mode = %s %s)' % ( median, mode, units )
        if transit_type=='primary':
            vbr_object.transit_depth_inferred_mean = inferred_depth_mean
            vbr_object.transit_depth_inferred_median = inferred_depth_median        
            vbr_object.transit_depth_inferred_mode = inferred_depth_mode
            vbr_object.transit_depth_inferred_stdv = uncertainty
        elif transit_type=='secondary':
            vbr_object.eclipse_depth_inferred_mean = inferred_depth_mean
            vbr_object.eclipse_depth_inferred_median = inferred_depth_median        
            vbr_object.eclipse_depth_inferred_mode = inferred_depth_mode
            vbr_object.eclipse_depth_inferred_stdv = uncertainty
    else:
        if transit_type=='primary':
            vbr_object.transit_depth_inferred_mean = None
            vbr_object.transit_depth_inferred_median = None
            vbr_object.transit_depth_inferred_mode = None
            vbr_object.transit_depth_inferred_stdv = None
        elif transit_type=='secondary':
            vbr_object.eclipse_depth_inferred_mean = None
            vbr_object.eclipse_depth_inferred_median = None
            vbr_object.eclipse_depth_inferred_mode = None
            vbr_object.eclipse_depth_inferred_stdv = None
    return None
    

def append_phasecurve( vbr_object, time, params, model_type='additive' ):
    """
    !!!Needs testing!!!
    
    Appends a planetary phase curve to the existing normalised training basis
    matrix, and records the appended array, so that it can be automatically
    appended to the normalised predictive basis matrix when that's constructed
    later.

    Requires the EccLightCurveModels.py and orbit.py modules.

    !!!Needs testing!!!
    """
    phasecurve_shape = simple_phasecurve( time, params, model_type=model_type )
    phasecurve_basis = phasecurve_shape/np.std(phasecurve_shape)
    vbr_object.append_training_basis_matrix(phasecurve_basis)
    return None


def simple_phasecurve( time, system_params, model_type='additive' ):
    """
    !!!Needs testing!!!
    
    A simple phase function that uses a simple sinusoidal variation only. It's
    of the same form as that used by Winn et al 2011 when modelling the observed
    phase curve of 55Cnc-e (see their Equation 1).

    !!!Needs testing!!!
    """
    # Unpack the system parameters, but be careful to account for the possibility that
    # we could be using either a quadratic or 4-parameter nonlinear limb darkening law,
    # as that will affect the number of system parameters that have been passed in:
    if len(system_params)==9:
        T0, P, aRs, RpRs, b, c1_q, c2_q, ecc, omega = system_params
        ld_type = 'quad'
    elif len(system_params)==11:
        T0, P, aRs, RpRs, b, c1_nl, c2_nl, c3_nl, c4_nl, ecc, omega = system_params
        ld_type = 'nonlin'
    else:
        pdb.set_trace() #this shouldn't happen
    # Append a nonzero secondary eclipse depth to the system_params array:
    system_params = np.array( [ system_params, [0.01] ] ) 
    # Calculate the orbital phase since mid-transit:
    orbphase = orbit.time_to_phase( time, P, T0, omega=omega, ecc=ecc )
    # Calculate the phase curve function according to Eqs 1 of Winn et al 2011
    # or Eqs 4&7 of Mislis et al 2011:
    cosz = -np.sin( incl ) * np.cos( orbphase )
    phase_curve = ( phase_amplitude/2.0 ) * ( 1+cosz )
    # Work out the in-transit and in-eclipse
    f, ycoords = EccLightCurveModels.EccLightCurve_aRs( system_pars, time, ld=ld_type, return_y=True )
    ixs_inec = ( ( f<1. )*( ycoords<0.0 ) )
    ixs_outec = ( f==1. ) + ( ( f<1. )*( ycoords>=0.0 ) )
    # Impose a flat 'bridge' across the times of eclipse, which makes it possible to
    # simultaneously fit a flat-bottomed eclipse function. Note that we do not need
    # to similarly enforce a flat-bottomed phase curve during times of primary transit!
    if model_type=='additive': 
        phase_curve[ixs_inec] = phase_curve[ixs_outec].max()        
    else:
        pdb.set_trace()
    return phase_curve

def lambert_phasecurve( time, params ):
    """
    UNDER CONSTRUCTION!!! UNTESTED!!!
    
    Supposed to return the phase curve for a Lambertian sphere.
    Uses the form given in Equation 6 of Mislis et al (2011).
    But I want to make sure this is all consistent with the equations
    for the same thing given by Seager in her Exoplanet Atmospheres book???

    UNDER CONSTRUCTION!!! UNTESTED!!!    
    """
    # Unpack the parameters:
    P, T0, omega, ecc, incl = params
    # Calculate the angle theta, as it is defined in Figure 2
    # of Mislis et al 2011:
    theta = orbit.time_to_phase( time, P, T0, omega=omega, ecc=ecc ) # Fig 2
    z = np.arccos(-np.sin(incl)*np.cos(theta)) # Eq 4
    phase_curve = phase_amplitude*(np.sin(z)+(np.pi-z)*np.cos(z)) # Eq 6
    # Work out the in-transit and in-eclipse
    f, ycoords = EccLightCurveModels.EccLightCurve_aRs( system_pars, time, ld=ld_type, return_y=True )
    ixs_inec = ( ( f<1. )*( ycoords<0.0 ) )
    ixs_outec = ( f==1. ) + ( ( f<1. )*( ycoords>=0.0 ) )
    # Impose a flat 'bridge' across the times of eclipse, which makes it possible to
    # simultaneously fit a flat-bottomed eclipse function. Note that we do not need
    # to similarly enforce a flat-bottomed phase curve during times of primary transit!
    if model_type=='additive': 
        phase_curve[ixs_inec] = phase_curve[ixs_outec].max()        
    else:
        pdb.set_trace()
    return phase_curve


def append_ellipsoid_distortion( vbr_object, time, params ):
    """
    !!!Needs testing!!!
    
    Appends a sinusoidal function approximating stellar ellipsoidal distortion to
    the existing normalised training basis matrix, and records the appended array,
    so that it can be automatically appended to the normalised predictive basis
    matrix when that's constructed later.

    ** params = [ P, T0, ecc, omega ]

    Requires the orbit.py module.

    !!!Needs testing!!!    
    """
    ellipsoid_distortion_shape = stellar_ellipsoid_distortion( time, system_params, template_amplitude )
    ellipsoid_distortion_basis = ellipsoid_distortion_shape / np.std( ellipsoid_distortion_shape )
    vbr_object.append_training_basis_matrix( ellipsoid_distortion_basis )
    return None

def stellar_ellipsoid_distortion( time, params, ellipsoid_amplitude ):
    """
    !!!Needs testing!!!
    
    Takes the system parameters and calculates a sinusoidal variation in the
    flux in phase with the planetary orbital period, such that the maximum
    brightness occurs when the planet is passing through the ascending and
    descending nodes, and the minimum brightness occurs when the planet at
    inferior and superior conjunction.

    Requires the orbit.py module.

    !!!Needs testing!!!        
    """
    P, T0, ecc, omega = params
    orbphase = orbit.time_to_phase( time, P, T0, omega=omega, ecc=ecc )
    cos2z = -np.sin( incl ) * np.cos( 2*orbphase )
    ellipsoid_distortion = ( ellipsoid_amplitude / 2.0 ) * ( 1+cos2z )
    return ellipsoid_distortion

