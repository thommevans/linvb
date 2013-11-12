import numpy as np
import scipy.linalg
import pdb
import math
import matplotlib.pyplot as plt


def describe( vbr_object ):
    """
    Prints some information about the VBR object to screen.
    """
    print '--------------------------------------------------------------------------------'
    print 'VBR object description:'
    print '\n  %i basis groups:' % ( vbr_object.n_basis_groups )
    for group in vbr_object.model_basis_types:
        print '   %s' % group
    if vbr_object.n_appendages >= 1:
        print '  %i appendages (post-normalisation):' % ( vbr_object.n_appendages )
        for appendage in vbr_object.model_appendage_names:
            print '   %s' % appendage
    print '\n  Has the unnormalised training basis matrix been constructed?\n    =%s' % vbr_object.training_basis_matrix_unnorm_status
    if vbr_object.training_basis_matrix_unnorm_status=='Yes':
        print '    size=%ix%i' % (np.shape(vbr_object.phi_train_unnorm)[0], np.shape(vbr_object.phi_train_unnorm)[1])
    print '\n  Have the training basis matrix and target data been normalised, ready for fitting?\n    =%s' \
          % vbr_object.training_basis_matrix_norm_status
    print '\n  Has the regression been done?\n    =%s' % vbr_object.regression_run_status
    print '\n  Has the predictive distribution been computed?\n    =%s' % vbr_object.predictive_distribution_status
    if vbr_object.fraction_1sigma!=None:
        print '    Data within predictive 1-sigma limits = %.2f percent' % (100*vbr_object.fraction_1sigma)
    if vbr_object.fraction_2sigma!=None:
        print '    Data within predictive 2-sigma limits = %.2f percent' % (100*vbr_object.fraction_2sigma)
    if ( vbr_object.disentangle_status=='No' ) + ( vbr_object.disentangle_status==False ):
        print '\n  Basis function contributions have **not** been separated/isolated.'
    else:
        print '\n  Basis function contributions **have** been separated/isolated.'
    print '--------------------------------------------------------------------------------'
    return None

def construct_basis_matrix( vbr_object, whichtype='train' ):
    """
    This routine constructs the 'core' basis matrix, and standardises/normalises
    it. More specifically, only the columns corresponding to the 'standard' model
    basis functions are generated by this routine; additional appended blocks for
    specialised basis functions (i.e. those not defined in the vbr_basis_functions.py
    module) must be added after this step using the append_training_basis_matrix()
    routine.

    Both the normalised and unnormalised forms of the basis matrix columns are
    generated.
    
    The 'whichtype' keyword argument controls whether the training or predictive
    inputs will be used to evaluate the basis functions. If whichtype is 'train',
    then the target training data will also be copied to a new variable and
    normalised.

    The following object attributes must be set for this task to work:
    **[model_basis_types and model_basis_kwargs] - these control the form of the 
        basis model, and are the same for both the training and predictive basis
        matrices
    **[model_basis_inputs_train or model_basis_inputs_pred] - the locations in input
        space that each of the basis functions are to be evaluated at; this does
        not need to be the same for the training and predictive basis matrices
    **target_train_unnorm - unnormalised target training data, which is needed if
        whichtype is set to 'train', as it will also be normalised; otherwise, if
        whichtype is set to 'pred', the shifts and scalings from the normalised
        training basis matrix will be used to perform the normalisation
    **model_add_offset - flag set to True or False, specifying whether or not a
        column of 1's will be added to the basis matrix

    Output is generated in the form of the following object attributes:
    **[phi_train_unnorm or phi_pred_unnorm] - the unnormalised core basis matrix
    **[phi_train_norm or phi_pred_norm] - the normalised core basis matrix that will
        actually be used in the regression algorithm
    **target_train_norm - the normalised target training data
    **[phi_train_norm_shifts and phi_train_norm_scalings]
      or [phi_pred_norm_shifts and phi_pred_norm_scalings] - the shift and scaling
      factors used to normalise each column of the core basis matrix
    **[target_train_norm_shift and target_train_norm_scaling] - the shift and scaling
        that was used to normalise the target training data
        
    """
    # Decide if we're constructing the training basis matrix or a predictive basis matrix, 
    # and set the variables to be used below accordingly:
    if whichtype=='train':
        if ( ( vbr_object.model_basis_types==None ) + ( vbr_object.model_basis_inputs_train==None ) )\
            * ( vbr_object.model_add_offset!=True ):
                print 'Do not have all the information needed to construct the training basis matrix!'
                print '  Missing either **model_basis_types** or **model_basis_inputs_train**'
                pdb.set_trace()
        else:
            n_data = len( vbr_object.target_train_unnorm )
            vbr_object.n_data_train = n_data
            if np.rank(vbr_object.model_basis_inputs_train[0])==0:
                vbr_object.model_basis_inputs_train = [ vbr_object.model_basis_inputs_train ]
            basis_inputs = vbr_object.model_basis_inputs_train
            # Work out if there are basis functions, or if we're only adding a constant offset:
            if (vbr_object.model_basis_types==None)+(vbr_object.model_basis_types==[]):
                offset_only = True
            else:
                offset_only = False
            # Because we're constructing a new basis matrix, we also want to make sure that we
            # erase any previous record of appendages to the basis matrix that might come later
            # after normalisation:
            vbr_object.phi_appendages_postnorm = None
            vbr_object.phi_ixs_appendages_postnorm = None
    if whichtype=='pred':
        if ((vbr_object.model_basis_types==None)+(vbr_object.model_basis_inputs_pred==None)\
            *(vbr_object.model_add_offset!=True)):
                print 'Do not have all the information needed to construct the predictive basis matrix!'
                print '  Missing either **model_basis_types** or **model_basis_inputs_train**; '
                print '  otherwise, **model_add_offset** must be True'
                pdb.set_trace()
        else:
            # Get the number of data points from the number of input locations that have been
            # provided. First check to see if we have a list of input arrays, or a single input
            # array:
            if np.rank(vbr_object.model_basis_inputs_pred[0])==0:
                vbr_object.model_basis_inputs_pred = [vbr_object.model_basis_inputs_pred]
            # Now we know that our inputs are in list format, we can work out the number of data
            # points. First, check to see if the set of inputs is a 1D array:
            elif np.rank(vbr_object.model_basis_inputs_pred[0])==0:
                n_data = len(vbr_object.model_basis_inputs_pred[0])
            # Otherwise, the first set of inputs must be a 2D array:
            else:
                n_data = np.shape(vbr_object.model_basis_inputs_pred[0])[0]
            vbr_object.n_data_pred = n_data
            basis_inputs = vbr_object.model_basis_inputs_pred
            # Work out if there are basis functions, or if we're only adding a constant offset:
            if (vbr_object.model_basis_types==None):
                offset_only = True
            else:
                offset_only = False
    # If we make it down to here, then we have enough information to construct a basis matrix.
    # Unless we are only generating a basis matrix consisting of a constant offset, the following
    # section builds up the basis matrix one block at a time, cycling through the various basis
    # function groups that have been provided:
    if offset_only==False:
        if np.rank( vbr_object.model_basis_types )==0:
            vbr_object.model_basis_types = [ vbr_object.model_basis_types ]
        basis_types = vbr_object.model_basis_types
        # Handle the unlikely case that a single number has been
        # passed in for the basis kwargs:
        try:
            if np.rank( vbr_object.model_basis_kwargs )==0:
                vbr_object.model_basis_kwargs = [ vbr_object.model_basis_kwargs ]
        except:
            pass
        basis_kwargs = vbr_object.model_basis_kwargs        
        vbr_object.n_basis_groups = len( basis_types )
        # Construct the basis matrix, one block at a time:
        basis_matrix = np.zeros( [ n_data, 1 ] )
        if whichtype=='train':
            vbr_object.phi_ixs_basis_groups = []
        for i in range( vbr_object.n_basis_groups ):
            if basis_kwargs[i]!=None:
                outarray = basis_types[i]( basis_inputs[i], **basis_kwargs[i] )
            else:
                outarray = basis_types[i]( basis_inputs[i] )
            basis_matrix = np.concatenate( [ basis_matrix, outarray ], axis=1 )
            # Record the column ixs taken up by the current basis group (note that
            # we must allow for the fact that we remove the first blank column of 
            # zeros further below, hence the -1):
            if whichtype=='train':
                first_column_ix = np.shape( basis_matrix )[1] - np.shape( outarray )[1] - 1
                last_column_ix = np.shape( basis_matrix )[1] - 1
                column_ixs = np.arange( first_column_ix, last_column_ix )
                vbr_object.phi_ixs_basis_groups = vbr_object.phi_ixs_basis_groups + [ column_ixs ]
        # Update the VBR object accordingly, depending on whether it is a training
        # or predictive basis matrix:
        if whichtype=='train':
            vbr_object.phi_train_unnorm = basis_matrix[:,1:]
            if ( ( vbr_object.model_basis_group_names==None ) + \
                 ( len(vbr_object.model_basis_group_names )!=vbr_object.n_basis_groups ) ):
                vbr_object.model_basis_group_names = ['']*vbr_object.n_basis_groups
        if whichtype=='pred':
            vbr_object.phi_pred_unnorm = basis_matrix[:,1:]
    # The following line normalises the basis matrix if one has been generated from basis functions
    # by this point, and, if it's a training data set, it will also normalise the target data:
    normalise_model_inputs( vbr_object, rescale=True, shift=True, whichtype=whichtype )
    # If requested, add a constant offset to the last column of the basis matrix:
    if vbr_object.model_add_offset==True:
        offset_column = np.ones( [ n_data, 1 ] )
        if whichtype=='train':
            if offset_only==True:
                vbr_object.phi_train_unnorm = offset_column
                vbr_object.phi_train_norm = offset_column
                vbr_object.phi_ixs_constant_offset = 0
                vbr_object.phi_train_norm_scalings = [0]
                vbr_object.phi_train_norm_shifts = [0]
            else:
                vbr_object.phi_train_unnorm = np.column_stack([vbr_object.phi_train_unnorm, offset_column])
                vbr_object.phi_train_norm = np.column_stack([vbr_object.phi_train_norm, offset_column])
                vbr_object.phi_ixs_constant_offset = np.shape(basis_matrix)[1]-1
                vbr_object.phi_train_norm_scalings = np.concatenate([vbr_object.phi_train_norm_scalings,[0]])
                vbr_object.phi_train_norm_shifts = np.concatenate([vbr_object.phi_train_norm_shifts,[0]])
        if whichtype=='pred':
            if offset_only==True:
                vbr_object.phi_pred_unnorm = offset_column
                vbr_object.phi_pred_norm = offset_column
            else:
                vbr_object.phi_pred_unnorm = np.column_stack([vbr_object.phi_pred_unnorm, offset_column])
                vbr_object.phi_pred_norm = np.column_stack([vbr_object.phi_pred_norm, offset_column])
    else:
        vbr_object.phi_ixs_constant_offset = None
    return None

def append_training_basis_matrix( vbr_object, appendage, appendage_name=None ):
    """
    Appends an NxK array to the existing NxM training basis matrix. Specifically,
    the input variable 'appendage' will be an NxK array that is appended. This is
    how specialised basis functions, such as planetary transit functions and phase
    curves, can be incorporated into the linear model.

    Any appended arrays are recorded in the object attribute phi_appendages_postnorm.
    That way, when a predictive training basis matrix is generated, the arrays stored
    in this variable can be easily added to it directly.

    Note that unlike the 'core' basis groups that are added to the basis matrix during
    the construct_basis_matrix() step, the 'appendage' basis functions are not recorded
    within the object. This is to allow for maximum flexibility, i.e. the appendage
    signal does not need to be restricted to signals that can be produced using a
    function that follows specific rules; it is irrelevant how the appendage signal is
    generated. But probably in practice, the reason I allowed for appendages to be
    added in this way is because sometimes the normalisation applied to the basis matrix
    columns in construct_basis_matrix() will not be appropriate,

    eg. you don't want to mean-subtract a transit basis function, because you very
    specifically require that it be equal to zero for all out-of-transit points.
    
    """
    if appendage_name==None:
        appendage_name = ''
    try:
        vbr_object.model_appendage_names = vbr_object.model_appendage_names+[appendage_name]
    except:
        vbr_object.model_appendage_names = [appendage_name]
    print '\nAppending to normalised training basis matrix...'
    vbr_object.phi_train_norm = np.column_stack([vbr_object.phi_train_norm, appendage])
    # Work out which columns the appendage takes up in the basis matrix so that they
    # can be accessed later on; this information will be needed when constructing the
    # predictive basis matrix for instance:
    first_column_ix = np.shape(vbr_object.phi_train_norm)[1]-1
    if np.rank(appendage)==1:
        ncolumns = 1
    elif np.rank(appendage)==2:
        ncolumns = np.shape(appendage)[1]
    vbr_object.phi_train_norm_scalings = np.concatenate([vbr_object.phi_train_norm_scalings,np.zeros(ncolumns)])
    vbr_object.phi_train_norm_shifts = np.concatenate([vbr_object.phi_train_norm_shifts,np.zeros(ncolumns)])
    last_column_ix = first_column_ix+ncolumns-1
    column_ixs = np.arange(first_column_ix,last_column_ix+1)
    if vbr_object.phi_appendages_postnorm==None:
        vbr_object.phi_appendages_postnorm = appendage
        vbr_object.phi_ixs_appendages_postnorm = [column_ixs]
    else:
        vbr_object.phi_appendages_postnorm = np.column_stack([appendage, vbr_object.phi_appendages_postnorm])
        vbr_object.phi_ixs_appendages_postnorm = vbr_object.phi_ixs_appendages_postnorm+[column_ixs]
    return None

def normalise_model_inputs( vbr_object, rescale=True, shift=True, whichtype='train' ):
    """
    Normalises the basis matrix. The whichtype keyword argument specifies whether
    it is the training or predictive basis matrix that is to be normalised. If
    whichtype is set to 'train', the target training data will be normalised at
    the same time.
    
    Output is generated in the form of the following object attributes:
    **[phi_train_norm or phi_pred_norm] - the normalised core basis matrix that will 
        actually be used in the regression algorithm
    **target_train_norm - the normalised target training data
    **[phi_train_norm_shifts and phi_train_norm_scalings] or
        [phi_pred_norm_shifts and phi_pred_norm_scalings] - the shift and scaling
        factors used to normalise each column of the core basis matrix
    **[target_train_norm_shift and target_train_norm_scaling] - the shift and scaling
        that was used to normalise the target training data
    """
    # Decide if we're normalising the training or predictive
    # basis matrix:
    if whichtype=='train':
        # If it's the training basis matrix, we need to work out what the scalings and shifts
        # will be for each column of the input and target data:
        try:
            print ' Generating normalised training basis matrix...'
            basis_matrix_unnorm = vbr_object.phi_train_unnorm
            # Calculate the scalings and shifts for each column:
            basis_shifts = np.mean(basis_matrix_unnorm, axis=0)
            basis_scalings = np.std(basis_matrix_unnorm, axis=0, ddof=1)
            # Work out which are the constant-valued columns:
            ixs = (basis_scalings==0)
            basis_scalings[ixs] = 1.0
            basis_shifts[ixs] = 0.0
            # And we will want to store these in the VBR object:
            vbr_object.phi_train_norm_shifts = basis_shifts
            vbr_object.phi_train_norm_scalings = basis_scalings
        except:
            # If we don't have a basis matrix to normalise (eg. if we are in the process of
            # constructing a basis matrix composed of a constant offset and/or unnormalised
            # appendages only), just skip this step:
            pass
        # Update the status flag:
        vbr_object.training_basis_matrix_unnorm_status = 'Yes'
        # We'll also want to normalise the target data:
        print ' Generating normalised target training data...'
        target_unnorm = vbr_object.target_train_unnorm
        target_shift = np.mean(target_unnorm, axis=0)
        target_scaling = np.std(target_unnorm, axis=0, ddof=1)
        # And also store this information in the VBR object:
        vbr_object.target_train_norm = ( target_unnorm - target_shift ) / target_scaling
        vbr_object.target_train_norm_shift = target_shift
        vbr_object.target_train_norm_scaling = target_scaling
        # If the white noise parameter is fixed, set it here:
        if vbr_object.model_beta_fixed==True:
            vbr_object.model_beta_mean_norm = (target_scaling**2.)/(vbr_object.model_whitenoise_mean_unnorm**2.)
            vbr_object.model_beta_stdv_norm = 0.0
        # Update the status flag:
        vbr_object.training_basis_matrix_norm_status = 'Yes'
    if whichtype=='pred':
        # Alternatively, if it's the predictive basis matrix we need to use the scalings
        # and shifts from the training data normalisation:
        try:
            print '\n Generating normalised predictive basis matrix...'
            basis_matrix_unnorm = vbr_object.phi_pred_unnorm
            basis_scalings = vbr_object.phi_train_norm_scalings
            basis_shifts = vbr_object.phi_train_norm_shifts
            ixs = (basis_scalings!=0)
            # If unnormalised columns have been added to the training basis matrix previously,
            # there will have been 0's appended to the basis_scalings and basis_shifts array
            # (see towards the end of the construct_basis_matrix() function), so we need to
            # ignore these here:
            basis_scalings = basis_scalings[ixs]
            basis_shifts = basis_shifts[ixs]
        except:
            # However, if none of the columns in the basis matrix need to be normalised
            # (eg. if we are in the process of constructing a basis matrix composed of a 
            # constant offset and/or unnormalised appendages only), just skip this step:
            pass
    # Now if we need to go ahead and normalise the basis matrix, do that:
    try:
        basis_matrix_norm = (basis_matrix_unnorm-basis_shifts)/basis_scalings
        # And update the VBR object accordingly:
        if whichtype=='train':
            vbr_object.phi_train_norm = basis_matrix_norm
            vbr_object.phi_train_norm_scalings = basis_scalings
            vbr_object.phi_train_norm_shifts = basis_shifts
        if whichtype=='pred':
            vbr_object.phi_pred_norm = basis_matrix_norm
    except:
        # Otherwise, just skip this step:
        pass
    print '    Done.'
    return None    
    
def unnormalise_model_outputs( vbr_object, means_pred_norm, stdvs_pred_norm ):
    """
    Takes the normalised output for the predictive distribution and un-normalises
    it, putting it back in the units of the original target training data. This 
    routine assumes that the do_linear_regression() task has already been run.

    The following object attributes need to be set:
    **[target_training_scaling and target_training_shift] - the shift and scaling
        that was used to normalise the target training data
    **[means_pred_norm and stdvs_pred_norm] - the normalised means and standard
        deviations of the Student's t, or very-nearly-Gaussian, predictive
        distributions, in the same units as the target training data

    Output is generated in the form of the following object attributes:
    **[model_pred_means_unnorm and model_pred_stdvs_unnorm] - means and standard
        deviations of the Student's t, or very-nearly-Gaussian, predictive
        distributions, in the same units as the target training data
    **model_whitenoise_mean_unnorm - if the white noise was treated as a free 
        variable in the model, the inferred value will be returned in same units
        as the target training data, taken from the inferred posterior distribution
        on the beta precision parameter
    """
    # Extract the shift and scaling that were used to normalise the data
    # for model fitting in the first place:
    scaling = vbr_object.target_train_norm_scaling
    shift = vbr_object.target_train_norm_shift
    # Adjust the predictive distribution accordingly and store in the
    # VBR object:
    vbr_object.model_pred_means_unnorm = means_pred_norm*scaling+shift
    vbr_object.model_pred_stdvs_unnorm = stdvs_pred_norm*scaling
    # If we're working in log units, provide the output in linear units as well:
    if vbr_object.target_log_units==True:
        vbr_object.model_pred_means_unnorm_unlogified, vbr_object.model_pred_stdvs_unnorm_unlogified = \
            unlogify_distribution(vbr_object.model_pred_means_unnorm, vbr_object.model_pred_stdvs_unnorm)
        vbr_object.target_train_unnorm_unlogified = np.exp(vbr_object.target_train_unnorm)
    # While we're at it, we can isolate the white noise component
    # of the estimated standard deviation in the model predictions,
    # and unnormalise it as well:
    beta = vbr_object.model_beta_mean_norm
    vbr_object.model_whitenoise_mean_unnorm = scaling/np.sqrt(beta)
    return None

def unlogify_distribution( mu_x, sig_x ):
    """
    If we have a normally distributed random variable X, such that:
        X ~ log(R)
    then R is a log-normal random variable.

    This routine takes the mean mu_x and standard deviation sig_x of the normal
    distribution X, and returns the mean mu_r and standard with mean mu_r and
    standard deviation sig_x, this function returns the mean and standard variation
    of the log-normal distribution R. If the median and/or mode differ signficantly
    from the mean of R, then a warning is printed to screen.
    """
    if type(mu_x)!=np.ndarray:
        mu_x = np.array(mu_x)
    if len(mu_x.flatten())!=np.size(mu_x):
        pdb.set_trace()
    if type(sig_x)!=np.ndarray:
        sig_x = np.array(sig_x)
    if len(sig_x.flatten())!=np.size(sig_x):
        pdb.set_trace()
    var_x = sig_x**2.
    var_r = (np.exp(var_x)-1.0)*np.exp(2*mu_x+var_x)
    sig_r = np.sqrt(var_r)
    mu_r = np.exp(mu_x + var_x/2.)
    med_r = np.exp(mu_x)
    mod_r = np.exp(mu_x-var_x)
    if ((mu_r-mod_r)/mu_r).max()>0.001:
        print '\n   WARNING: mode of log-normal distribution differs from mean by more than 0.1%\n'
    # It's probably better to return med_r instead of mu_r because np.exp(mu_x) is probably more
    # intuitive when thinking about converting a predictive distribution from log to non-log:
    return med_r, sig_r 

def calc_predictive_dist( wm, phi, smatrix, beta_fixed=None, cn=None, dn=None, calc_stdvs=True ):
    """
    This is the routine that actually calculates the means and standard deviations of
    the predictive distributions at each location in input space. Recall that the predictive
    distribution is strictly speaking a Student's t-distribution if the white noise beta
    parameter was a treated as a free variable in the model (see Eq 31 of Drugowitsch 2008),
    but the distribution is well-approximated by a Gaussian if the number of degrees of
    freedom is large, as reflected by the value of  model_cn. If on the other hand, the
    white noise beta parameter was held fixed, the predictive distribution is a Gaussian
    (see Eq 36 of Bishop & Tipping 2000).

    The means and standard deviations are returned as arrays rather than being stored as
    attributes within the object; the latter is done by get_predictive_distribution() in
    the vbr_routines.py module, which calls the current routine in the process.
    """
    # Calculate the means for the predictive distribution:
    n_data_pred = np.shape( phi )[0]
    print ' Calculating model predictive means...'
    means_pred_norm = np.array( phi*wm ).flatten()
    print '    Done.'
    if calc_stdvs==True:
        # Calculate the standard deviations for the predictive distribution:
        print ' Calculating model uncertainties... '
        if n_data_pred>5e4:
            print '   (could take a while - done one at a time in a loop)'
        stdvs_pred_norm = np.empty( n_data_pred )
        for i in range(n_data_pred):
            phi_row = phi[i,:]
            if beta_fixed:
                # Eq 38 of Bishop & Tipping 2000:
                stdvs_pred_norm[i] = np.sqrt( ( 1./beta_fixed ) + phi_row*smatrix*phi_row.T )
            else:
                # Eq 31 of Drugowitsch 2008:
                stdvs_pred_norm[i] = np.sqrt( np.array( dn*( 1 + phi_row*smatrix*phi_row.T ) / ( cn-1 ) ) )
        print '    Done.'
    else:
        stdvs_pred_norm = False
    return means_pred_norm, stdvs_pred_norm

def split_basis_model( vbr_object, ixs_subset ):
    """
    Once the linear regression has been performed, this routine extracts the contributions
    from the subset of basis functions with column indices specified by ixs_subset, as well
    as its complement. It returns the distributions for these two complementary subsets in
    unnormalised units of the target data. Hence:
       subset + complement = data
    See the documentation for the disentangle_basis_contributions() routine in the vbr_routines.py
    module for more information.

    Note that it returns the means and stdvs for the **predictive** distribution, which is
    not necessarily the same as the locations of the training data unless it has been defined
    in that way. Arrays containing these variables are returned, rather than storing them within
    the object as attributes; the latter is done by disentangle_basis_contributions() in the
    vbr_routines.py module, which calls the current routine in the process.
    """
    # Check that the regression has actually been run:
    if vbr_object.regression_run_status=='No':
        print 'Regression must be performed first!'
        pdb.set_trace()
    # Unpack the various objects that will be needed:
    phi_full = np.matrix( vbr_object.phi_pred_norm )
    w_full = np.matrix( vbr_object.model_weights_means ).T
    s_full = np.matrix( vbr_object.model_smatrix )
    # Determine the sorted indices of the subset and complement: 
    ixs_full = np.arange( vbr_object.n_basis_funcs )
    ixs_subset = np.sort( ixs_subset )
    ixs_complement = np.sort( np.setxor1d( ixs_subset, ixs_full ) )
    # Work out the number of basis functions there will be in the subset and its complement:
    nsubset = len( ixs_subset )
    ncomplement = len( ixs_complement )
    # Divide the inferred means for the weights between the subset and its complement:
    w_subset = np.matrix( w_full[ixs_subset] )
    w_complement = np.matrix( w_full[ixs_complement] )
    # Divide the normalised basis matrix between columns corresponding to the subset and its
    # complement:
    phi_subset = phi_full[:,ixs_subset]
    phi_complement = phi_full[:,ixs_complement]
    # To calculate the variances associated with the sub-models, we will use the diagonal blocks
    # of the s matrix that correspond to the subset and its complement:
    s_subset = np.matrix( np.zeros( [nsubset,nsubset] ) )
    for i in range(nsubset):
        ix = ixs_subset[i]
        s_subset[:,i] = s_full[:,ix][ixs_subset]
    s_complement = np.matrix(np.zeros( [ ncomplement, ncomplement ] ) )
    for i in range(ncomplement):
        ix = ixs_complement[i]
        s_complement[:,i] = s_full[:,ix][ixs_complement]
    # Now we have enough information to calculate the distributions for the subset and its complement:
    if vbr_object.model_beta_fixed==True:
        beta_fixed = vbr_object.model_beta_mean_norm
        cn = None
        dn = None
    else:
        beta_fixed = None
        cn = vbr_object.model_cn
        dn = vbr_object.model_dn
    pred_means_subset_norm, pred_stdvs_subset_norm = calc_predictive_dist( w_subset, phi_subset, s_subset, \
                                                                           beta_fixed=beta_fixed, cn=cn, dn=dn )
    pred_means_complement_norm, pred_stdvs_complement_norm = calc_predictive_dist( w_complement, phi_complement, \
                                                                                   s_complement, beta_fixed=beta_fixed, \
                                                                                   cn=cn, dn=dn )
    # Get the parameters that were used to scale and shift the target data for the regression:
    scaling = vbr_object.target_train_norm_scaling
    shift = vbr_object.target_train_norm_shift
    # Rescale the subset distribution so that it's in the units of the original target data:
    pred_means_subset_unnorm = scaling * pred_means_subset_norm
    pred_stdvs_subset_unnorm = scaling * pred_stdvs_subset_norm
    # Do the same for the complement distribution, and let it absorb the shift term also:
    pred_means_complement_unnorm = scaling * pred_means_complement_norm+shift
    pred_stdvs_complement_unnorm = scaling * pred_stdvs_complement_norm
    
    return [ pred_means_subset_unnorm, pred_stdvs_subset_unnorm ], [ pred_means_complement_unnorm, pred_stdvs_complement_unnorm ]

def check_sigmalimits( vbr_object ):
    """
    If the do_linear_regression() and get_predictive_distribution() tasks have been carried
    out, and if the predictive input locations are identical to the training input locations,
    then this routine will check the fraction of data points that fall within 1 and 2 sigma
    of the prediction means. The information will be printed to screen and also stored within
    the object in the form of the following object attributes:
    **[fraction_1sigma and fraction_2sigma] - the fraction of data points that fall within the
        1 and 2 sigma limits of the predictive distributions; these values can be compared with
        the theoretical values of 0.683 and 0.954 for a true Gaussian
    """
    if vbr_object.model_basis_inputs_pred!=vbr_object.model_basis_inputs_train:
        # Print a warning rather than calling get_predictive_distribution(), in case the
        # values already there are important:
        print 'WARNING: Predictive distribution not evaluated at the same locations as the '
        print 'training data; cannot calulate the fractions within 1 and 2 sigma!'
    else:
        ixs_1sigma = ( abs( vbr_object.model_pred_means_unnorm - vbr_object.target_train_unnorm ) < vbr_object.model_pred_stdvs_unnorm )
        vbr_object.fraction_1sigma = len( vbr_object.target_train_unnorm[ixs_1sigma] ) / float( vbr_object.n_data_train )
        ixs_2sigma = ( abs( vbr_object.model_pred_means_unnorm - vbr_object.target_train_unnorm ) < 2*vbr_object.model_pred_stdvs_unnorm )
        vbr_object.fraction_2sigma = len( vbr_object.target_train_unnorm[ixs_2sigma] ) / float( vbr_object.n_data_train )
        print '   Data within predictive 1-sigma limits = %.2f percent' % ( 100*vbr_object.fraction_1sigma )
        print '   Data within predictive 2-sigma limits = %.2f percent' % ( 100*vbr_object.fraction_2sigma )
    return None  

def find_1d_grid( data1d, requested_bin_spacing, npts_min ):
    """
    Takes a 1D data set and divides the domain into a uniform grid, with the specified spacings.
    Note that the requested_bin_spacing is given in the units of the input data1d.
    
    Returns the locations (i.e. bin centers) for each of the bins that contain more than the
    threshold number of data points.
    """
    # Get the range of the data:
    lower = data1d.min()
    upper = data1d.max()
    # Work out the number of bins we need if they're going be spaced as close
    # as possible to the requested spacing:
    nbins = np.ceil( ( upper - lower ) / float( requested_bin_spacing ) )
    # Compute the histogram:
    histogram, bin_edges = np.histogram( data1d, bins=nbins )
    # Strip the un-needed final entry in the bin_edges array:
    bin_edges = bin_edges[:-1]
    # Calculate the bin widths:
    bin_widths = bin_edges[1]-bin_edges[0]
    # Work out which bins have the minimum number of points:
    ixs = ( histogram >= npts_min )
    bin_edges = bin_edges[ixs]
    histogram = histogram[ixs]
    # Calculate the bin centers:
    nbins = len( histogram )
    bin_centers = np.empty( nbins )
    for i in range( nbins ):
        bin_centers[i] = bin_edges[i]+0.5*bin_widths
    return bin_centers

def find_2d_grid( data2d, requested_bin_spacing, npts_min ):
    """
    The same as the fit_1d_grid function, except for 2D data.
    
    Hence, the data is binned on a 2D grid, and the 2D coordinates
    of the bin centers are returned.

    Be aware that if you're dividing up a 2D plane spanned by
    two variables with different units, it's probably a good idea
    to standardise both sets of input variables before passing them
    in here. For example, if you're using this routine to find
    the centers of 2D Gaussian basis functions, I currently have
    the code set up to only work for orthogonal 2D Gaussians
    (i.e. diagonal covariance matrices), so the basis width will
    need to be suitable for both input variables. Standardising
    both input variables would hopefully help to ensure this.
    Note that the standardised input variables would also need to
    be used in constructing the 2D Gaussian basis functions in that
    case.
    """
    print 'NEED TO DOUBLE CHECK THIS IS DOING THE CORRECT THING: \n \
           previously the way i had it set up with vbr_script it was \n \
           not setting the basis widths properly!!!!!'
    # Get the range of data along the first axis:
    data_x = data2d[:,0]
    lower_x = data_x.min()
    upper_x = data_x.max()
    # Get the range of data along the second axis:
    data_y = data2d[:,1]
    lower_y = data_y.min()
    upper_y = data_y.max()
    # Work out the number of bins we need if they're going
    # be spaced as close as possible to the requested spacing:
    nbins_x = np.ceil( ( upper_x-lower_x ) / float( requested_bin_spacing ) )
    nbins_y = np.ceil( ( upper_y-lower_y ) / float( requested_bin_spacing ) )
    # Compute the histogram:
    histogram, bin_x_edges, bin_y_edges = np.histogram2d( data_x, data_y, bins=( nbins_x, nbins_y ) )
    # NOTE: The values in the histogram array will be such that the  x-axis is by default along
    # the 1st dimension (i.e. rows, increasing downwards), and the y-axis is by default along
    # the 2nd dimension (i.e. columns, increasing rightwards).
    # Strip the un-needed last entries of the bin_x_edges and bin_y_edges arrays:
    bin_x_edges = bin_x_edges[:-1]
    bin_y_edges = bin_y_edges[:-1]
    # Calculate the bin widths:
    bin_x_widths = bin_x_edges[1]-bin_x_edges[0]
    bin_y_widths = bin_y_edges[1]-bin_y_edges[0]    
    # Work out which bins contain the minimum number of points:
    ixs = ( histogram >= npts_min )
    histogram = histogram[ixs]
    # Convert the bin_edges arrays into bin_edges mesh grids:
    bin_x_mesh, bin_y_mesh = np.meshgrid( bin_x_edges, bin_y_edges )
    # Before extracting the relevant entries, we need to put the meshgrids into the same format
    # the histogram entries (see above):
    bin_x_edges = bin_x_mesh.T[ixs]
    bin_y_edges = bin_y_mesh.T[ixs]
    # Calculate the bin centers:
    nbins = len( histogram )
    bin_centers = np.empty( [ nbins, 2 ] )
    for i in range( nbins ):
        bin_centers[i,0] = bin_x_edges[i] + 0.5*bin_x_widths
        bin_centers[i,1] = bin_y_edges[i] + 0.5*bin_y_widths
    return bin_centers, bin_x_widths, bin_y_widths
    


def logdet( matrix, cautious=False ):
    """
    Exploit the fact that the inverse s matrix is positive definite to calculate the
    determinant of the s matrix via the Cholesky decomposition rather than the LU
    decomposition.

    If the cautious keyword is set to True, the routine will carefully check whether or
    not the matrix is positive definite, but this requires calculating the eigenvalues
    which adds some time to the computation; for this reason it is set to False by default.
    """
    if cautious==True:
        # First make sure that matrix is symmetric:
        if np.allclose( matrix.T, matrix ) == False:
            print 'MATRIX NOT SYMMETRIC'
            pdb.set_trace()
        # Second make sure that matrix is positive definite; the routine 
        # will stop with an error if the matrix is not positive definite:
        eigenvalues = scipy.linalg.eigvalsh( matrix )
    # Calculate the Cholesky decomposition, since this is about twice as 
    # fast as calculating the LU decomposition, which is the way scipy's
    # scipy.linalg.det() routine calculates determinants:
    cholesky_decomposition = scipy.linalg.cholesky( matrix )
    # The determinant is equal to the product of the diagonal terms in the
    # Cholesky decomposition matrix squared, so extract the diagonal:
    cholesky_diagonal_entries = np.diag( cholesky_decomposition ).flatten()
    # The log of the determinant is therefore equal to two times the sum
    # of the logarithms of the diagonal entries:
    logdet = 2.*np.sum( np.log( cholesky_diagonal_entries ) )
    return logdet
