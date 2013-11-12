import numpy as np
import scipy.special
import scipy.linalg
import vbr_utilities
import time
import pdb
import copy
import matplotlib.pyplot as plt
import sys

# TO DO: Check that the fixed Gamma hyper-hyperpriors that I use are appropriately uninformative...


######################################################################################################
#
# This code is mostly based on material contained in the following sources:
#
#   1. Bishop & Tipping, 2000, "Variational Relevance Vector Machines", in "Uncertainty in Artificial
#        Intelligence Proceedings 2000", p46-53
#   2. Section 3 of Chapter 10 in Bishop, 2006, "Pattern Recognition and Machine Learning", published
#        by Springer, USA.
#   3. Notes by Jan Drugowitsch, 2008 (updated 2010), accessible here:
#        http://www.lnc.ens.fr/~jdrugowi/code/bayes_linear_notes-0.1.3.pdf
#
######################################################################################################


def do_linear_regression( vbr_object, convergence=0.00001 ):
    """
    Uses the VB algorithm to perform linear regression. This routine assumes
    that the following object attributes have already been set:
      - phi_train_norm (normalised basis matrix)
      - target_train_norm (normalised target vector)

    The form of the Bayesian model is:

      p( t | Phi(x), w, beta ) = prod( N( t_n | Phi_n*w, beta^(-1) )

    where Phi_m is the mth row of the NxM basis matrix Phi(x), and where:

      p( w | alpha ) = prod( N( w_m | 0, alpha_m ) )
      p( alpha ) = prod( Gam( alpha_m | a, b_m ) )
      p( beta ) = Gam( beta | c, d )

    Note that this notation is more consistent with that used by Bishop &
    Tipping 2000 as opposed to Drugowitsch; the main difference is the notation
    of (a,b) and (c,d) for the hyperparameters on alpha and beta, which are
    swapped compared to Drugowitsch (which can be annoyingly confusing).
      
    Output is generated in the form of the following object attributes:
      **model_weights_means - means of the posterior distributions inferred
        for the linear basis weights
      **model_weights_stdvs - standard deviations of the posterior distributions
        for the linear basis weights
      **model_an, model_bn - parameters for the Gamma posteriors on the inferred
        alpha values, which themselves are the expected values of the Gaussian
        posteriors on each of the linear weights; model_an is constant and the same
        for all linear weights, but model_bn is a vector learned from the data with
        each entry corresponding to one of the linear weights
      **model_cn, model_dn - scalar parameters of the Gamma posterior on the white
        noise term beta; model_cn is constant, but model_dn is learned from the
        data
      **model_alpha - vector containing the inferred precisions for the Gaussian 
        posteriors on each of the linear weights, as learned from the data
      **model_beta - the expected value of the Gamma distribution for the white 
        noise precision; this will be inferred as part of the VB algorithm unless
        it has been explicitly fixed to some value which wil be indicated by the
        model_beta_fixed object attribute
      **model_smatrix - MxM matrix where M is the number of basis functions in the
        model, used to calculate various quantities; denoted by SN in the Bishop
        paper+textbook and denoted as V in the Drugowitsch notes
    """

    # Get the final number of basis functions that are to be used for the
    # regression (includes constant offset as a basis function):
    try:
        vbr_object.n_basis_funcs = np.shape( vbr_object.phi_train_norm )[1]
    except:
        vbr_object.n_basis_funcs = 0
    try:
        vbr_object.n_appendages = len( vbr_object.phi_ixs_appendages_postnorm )
    except:
        vbr_object.n_appendages = 0
    # Implement the VB algorithm over all unknown quantities until convergence
    # is reached:
    print '\nPerforming VB iterative algorithm:'
    t1=time.time()
    # The appropriate algorithm is selected depending on how the white noise
    # term is to be treated:
    if vbr_object.model_beta_fixed==True:
        iterate_betafixed( vbr_object, convergence=convergence )
    else:
        iterate_betafree( vbr_object, convergence=convergence )
    t2 = time.time()
    delt = (t2-t1)/60.
    print '   Time taken for VB convergence = %.3f minutes' % delt
    # Update the VBR object to store the results of the regression:
    vbr_object.regression_run_status = 'Yes'
    vbr_object.disentangle_status = 'No'
    print '\n Finished: VBR object has been updated with output from linear regression.'
    return None

def iterate_betafixed( vbr_object, convergence=None ):
    """
    The VB algorithm with the white noise fixed ahead of time. Shrinkage priors on each
    of the linear weights.
    
    This has been adapted from the Bishop & Tipping 2000 and Drugowitsch 2008 papers
    that have beta as a free parameter + shrinkage priors on each of the alpha terms,
    combined with the VB linear regression derivation given in the Bishop textbook
    where beta is held fixed and a single shrinkage prior is placed on all of the alpha
    terms.

    Note that we have maintained the Bishop (as opposed to Drugowitsch) notation for
    the hyperparameters on alpha and beta, i.e. (a0,b0) for beta and (c0,d0) for alpha.
    This can be a bit confusing but is done to maintain consistency of notation throughout
    the code.
    """

    # NOTE: It is most natural to work with matrices rather than numpy arrays when 
    # implementing the VB algorithm. I don't actually think it makes a significant
    # difference in speed, if any, but it definitely makes things syntactically
    # more simple, eg. a*b instead of np.multiply(a,b).
    
    # Rename a few quantities to make things more concise below:
    n_data = vbr_object.n_data_train
    n_basis_funcs = vbr_object.n_basis_funcs
    beta = vbr_object.model_beta_mean_norm # scalar
    # And put the data in matrix format for this routine only:
    phi = np.matrix(vbr_object.phi_train_norm)
    target = np.matrix(vbr_object.target_train_norm).T
    # Initialise the variational lower bound L(q) and set the maximum number
    # of iterations:
    lq_last = -sys.float_info.max
    max_iter = 500
    first_pass = True
    # Initialise an ignorant Gamma priors over the alpha parameters (TODO think about
    # whether or not the priors as I've defined them here are actually appropriate!!!):
    a0 = 1e-4#1e-2
    b0 = 1e-4
    # Compute a couple of matrix products that will be used later:
    phi_corr = phi.T * phi
    phitarget_corr = phi.T * target
    # Calculate the updated first-parameter values for the Gamma-distributions 
    # over the alpha parameters - these can be defined up here because they
    # remain unchanged during the iterative algorithm:
    an = a0 + 1./2.
    # Calculate the expected values of the alpha parameter priors:
    exp_alpha = np.matrix(np.ones(n_basis_funcs) * a0 / b0).T
    for iter in range(max_iter):
        # Calculate the s matrix and associated quantities:
        inv_s = np.matrix( np.diag( np.array( exp_alpha )[:,0] ) ) + beta * phi_corr
        s = np.matrix(scipy.linalg.inv(inv_s))
        # NOTE: To get the log determinant of s, we need to take the *negative* log
        # determinant of the *inverse* of s:
        logdet_s = -vbr_utilities.logdet(inv_s)
        # Calculate updated expectation values for the linear weights:
        exp_w = beta * np.dot( s, phitarget_corr )[:,0]
        # Evaluated the updated expected values for the Gamma distributions over each
        # of the alpha values:
        bn = b0 + 0.5 * ( ( np.array( exp_w )[:,0]**2 ) + np.diag(s) )
        exp_alpha = np.matrix( an / bn ).T 
        # Calculate the variational lower bound, but ignore the terms depending on the fixed
        # constants (a0,b0,c0,d0,n_data,D) because these are unchanging/irrelevant when comparing
        # successive lower bound values:
        term1 = + beta * exp_w.T * phitarget_corr
        term2 = - 0.5 * beta * np.trace( phi_corr * ( exp_w * exp_w.T + s ) )
        term3 = - ( an/2. ) * np.sum( np.array( np.diag( exp_w * exp_w.T + s ) ) / bn )
        term4 = - ( a0+0.5 ) * np.sum( np.log(bn) ) 
        term5 = - b0 * an * np.sum( 1./bn )
        term6 = + 0.5 * logdet_s
        lq = term1+term2+term3+term4+term5+term6
        # It can be proven that the variational bound *must* grow. So...
        if lq_last > lq:
            # ... otherwise, something has gone wrong or the problem is misbehaved/unstable!
            file = open('ERROR_LOG','w')
            file.write('Previous bound = %6.6f --> DECREASED to current bound = %6.6f' % (lq_last, lq))
            file.close()
            raise Exception('Variational bound should not reduce - see ERROR_LOG')
            return
        # Stop if change in variation bound is < 0.001%, i.e. arbitrary definition of convergence:
        if abs(lq_last - lq) < abs(convergence * lq):        
            break
        if first_pass==True:
            first_pass=False
        else:
            delq = abs(100*(lq-lq_last)/float(lq_last))
            print '  Step %d. Delta_L(q) = %g%%' % (iter, delq)
        lq_last = lq
    if iter<max_iter:
        print '\n Convergence reached! \n'
    if iter==max_iter:
        print '\n Maximum number of iterations reached; breaking loop \n'
    print 
    if iter == max_iter:    
        warnings.warn('Bayes:maxIter ... Bayesian linear regression reached maximum number of iterations.')    
    # Record the final value reached by the variational lower bound:
    vbr_object.lq = lq
    # Parameters controlling the widths of the normal priors over the linear weights:
    vbr_object.model_alpha = np.array(exp_alpha).flatten() # array
    # Hyperparameters controlling the Gamma hyperpriors over the alpha
    # parameters, which are the widths of the priors on the linear weights:
    vbr_object.model_an = an # scalar
    vbr_object.model_bn = np.array(bn) # array; one entry for each linear weight
    # The s matrix, which is used to calculate the covariance of the joint
    # posterior distribution over the linear weights:
    vbr_object.model_smatrix = np.array(s) # matrix
    # Marginalised posterior distributions over the linear weights:
    vbr_object.model_weights_means = np.array(exp_w).flatten() # array
    vbr_object.model_weights_stdvs = np.sqrt( np.diag(s) ) # array
    return None

def iterate_betafree( vbr_object, convergence=None ):
    """
    The VB algorithm with the white noise inferred by training the model on the data.
    Shrinkage priors on each of the linear weights.
    
    Taken from Bishop & Tipping 2000; a similar derivation is provided in Section 10.3
    of the Bishop textbook, but without shrinkage priors on each linear weight. Note
    that we have maintained the Bishop (as opposed to Drugowitsch) notation for the
    hyperparameters on alpha and beta, i.e. (a0,b0) for beta and (c0,d0) for alpha. This
    can be a bit confusing but is done to maintain consistency of notation throughout
    the code.
    """

    # NOTE: It is most natural to work with matrices rather than numpy arrays when 
    # implementing the VB algorithm. I don't actually think it makes a significant
    # difference in speed, if any, but it definitely makes things syntactically
    # more simple, eg. a*b instead of np.multiply(a,b).
    
    # Rename a few quantities to make things more concise below:
    n_data = vbr_object.n_data_train
    n_basis_funcs = vbr_object.n_basis_funcs
    # And put the data in matrix format for this routine only:
    phi = np.matrix( vbr_object.phi_train_norm )
    target = np.matrix( vbr_object.target_train_norm ).T
    # Initialise the lower bound L(q) and set the maximum number of iterations:
    lq_last = -sys.float_info.max
    max_iter = 500
    first_pass = True
    # Initialise an ignorant Gamma hyperprior over the alpha parameters:
    a0 = 1e-6#1e-2
    b0 = 1e-6
    # Do the same for the the beta parameter:
    c0 = 1e-6#1e-2
    d0 = 1e-6
    # Compute a couple of matrix products that will be used later:
    phi_corr = phi.T * phi
    phitarget_corr = phi.T * target
    # Calculate the updated first-parameter values for the Gamma-distributions
    # over the alpha and beta parameters - these can be defined up here because
    # they remain unchanged during the iterative algorithm:
    an = a0 + 1 / 2. #alpha
    cn = c0 + n_data / 2. #beta
    # Calculate the expected values of the alpha parameter priors:
    exp_alpha = np.matrix( np.ones( n_basis_funcs ) * a0 / b0).T 
    # Calculate values for a few more quantities that will be used later:
    n_basis_gammaln_an = n_basis_funcs * scipy.special.gammaln(an)
    gammaln_cn = scipy.special.gammaln(cn) 
    for iter in range(max_iter):
        # Calculate the s matrix and associated quantities - this is an important
        # part of the covariance matrix for the normal-part of the normal-gamma
        # posterior over the linear weights and beta:
        inv_s = np.matrix( np.diag( np.array(exp_alpha)[:,0] ) ) + phi_corr
        s = np.matrix( scipy.linalg.inv( inv_s ) )
        # NOTE: To get the log determinant of v, we need to take the *negative* log
        # determinant of the *inverse* of v:
        logdet_s = -vbr_utilities.logdet( inv_s )
        # Calculate updated expectation values for the linear weights:
        exp_w = np.dot( s, phitarget_corr )[:,0]
        # Calculate the updated second-parameter value in the Gamma distribution
        # over the beta term:
        sse = np.sum( np.power( phi * exp_w - target, 2 ), axis=0 )
        if np.imag( sse )==0:
            sse = np.real( sse )
        else:
            pdb.set_trace()
        dn = float( d0 + 0.5 * (sse + np.sum( ( np.array(exp_w)[:,0]**2 ) * np.array(exp_alpha)[:,0], axis=0 ) ) )
        # Evaluate the updated expected value for the Gamma distribution over the
        # beta parameter:
        exp_beta = cn / dn
        # Evaluated the updated expected values for the Gamma distributions over each
        # of the alpha values:
        bn = b0 + 0.5 * ( exp_beta * ( np.array(exp_w)[:,0]**2 ) + np.diag( s ) )
        exp_alpha = np.matrix(an / bn).T
        # Calculate the variational lower bound, but ignore the terms depending on the
        # constants (a0,b0,c0,d0,n_data,D) because these are irrelevant when comparing successive
        # lower bound values:
        term1 = - 0.5 * ( exp_beta*sse + np.sum( np.multiply( phi,phi*s ) ) )
        term2 = + 0.5 * logdet_s
        term3 = - d0 * exp_beta
        term4 = + gammaln_cn - cn * np.log(dn) + cn + n_basis_gammaln_an - an * np.sum(np.log(bn))
        lq = term1+term2+term3+term4
        # It can be proven that the variational bound *must* grow. So...
        if lq_last > lq:
            # ... otherwise, something has gone wrong or the problem is misbehaved/unstable!
            file = open('ERROR_LOG','w')
            file.write('Previous bound = %6.6f --> DECREASED to current bound = %6.6f' % (lq_last, lq))
            file.close()
            raise Exception('Variational bound should not reduce - see ERROR_LOG')
            return
        # Stop if change in variation bound is < 0.001%, i.e. arbitrary definition of convergence:
        if abs(lq_last - lq) < abs(convergence * lq):        
            break
        if first_pass==True:
            first_pass=False
        else:
            delq = abs(100*(lq-lq_last)/float(lq_last))
            print '  Step %d. Delta_L(q) = %g%%' % (iter, delq)
        lq_last = lq
    if iter<max_iter:
        print '\n Convergence reached! \n'
    if iter==max_iter:
        print '\n Maximum number of iterations reached; breaking loop \n'
    print 
    if iter == max_iter:    
        warnings.warn('Bayes:maxIter ... Bayesian linear regression reached maximum number of iterations.')    
    # Record the final value reached by the variational lower bound:
    vbr_object.lq = lq
    # Parameters controlling the widths of the normal priors over the linear weights:
    vbr_object.model_alpha = np.array(exp_alpha).flatten() # array
    # Hyperparameters controlling the Gamma hyperpriors over the alpha
    # parameters, which are the widths of the priors on the linear weights:
    vbr_object.model_an = an # scalar
    vbr_object.model_bn = np.array(bn) # array; one entry for each linear weight
    # Parameters controlling the Gamma posterior over the white noise:
    vbr_object.model_cn = cn # scalar
    vbr_object.model_dn = dn # scalar
    # The expected value of the Gamma posterior over the white noise:
    vbr_object.model_beta_mean_norm = exp_beta # scalar
    vbr_object.model_beta_stdv_norm = np.sqrt(cn)/dn # scalar
    # The s matrix, which is used to calculate the covariance of the joint
    # posterior distribution over the linear weights:
    vbr_object.model_smatrix = np.array(s) # matrix
    # Marginalised posterior distributions over the linear weights:
    vbr_object.model_weights_means = np.array(exp_w).flatten() # array
    vbr_object.model_weights_stdvs = np.sqrt((dn/cn)*np.diag(s)) # array
    # NOTE: The marginalised posterior distributions for the weights are
    # strictly speaking the product of a normal distribution and a gamma
    # distribution (see Eq 34 of Drugowitsch); however, we make a point
    # approximation for the white noise parameter (i.e. np.sqrt(an/bn)),
    # which is analogous to what Bishop & Tipping 2000 do in their Eq 36.
    return None
    
def get_predictive_distribution( vbr_object ):
    """
    Takes the output from the linear VB algorithm and calculates the predictive
    distribution at the specified locations in input space. Requires that the
    do_linear_regression() task has already been run, which is checked using the
    regression_run_status object attribute.

    In addition, the model_basis_inputs_pred object attribute must be specified,
    specifying where in input space the predictive distributions is to be evaluated.
    If this is not set explicitly, by default it will be set to same locations as
    the training data.

    Output is generated in the form of the following object attributes:
      **phi_pred_norm - the normalised basis matrix used to calculate the predictive
        distribution
      **model_pred_means_unnorm, model_pred_stdvs_unnorm - means and standard
        deviations of the Student's t, or very-nearly-Gaussian, predictive
        distributions, in the same units as the target training data
      **model_whitenoise_mean_unnorm - estimated or fixed white noise value in the
        same units as the target training data, taken from the inferred posterior
        distribution on the beta precision parameter

    If the target_log_units object attribute is set to True, the predictive distribution
    will be returned to its linear base by the vbr_utilities.unnormalise_model_outputs()
    routine. The output will be stored in object attributes with the same name as those
    above, but with an '_unlogified' suffix appended. For example, suppose that we're
    fitting to the log(flux) (i.e. magnitudes) but we want to express the output in flux,
    then the log(flux) predictive means will be contained in model_pred_means_unnorm
    and the equivalent flux values will be contained in model_pred_means_unnorm_unlogified.
    """

    print '\nComputing predictive distribution:'    
    # First, make sure that the predictive inputs have been provided
    # in a list format; otherwise, put them in a list format:
    if np.rank( vbr_object.model_basis_inputs_pred[0] )==0:
        vbr_object.model_basis_inputs_pred = [ vbr_object.model_basis_inputs_pred ]
    if vbr_object.model_basis_inputs_pred==vbr_object.model_basis_inputs_train:
        pred_matrix_norm = vbr_object.phi_train_norm
    # Often, we will want to evaluate the predictive distribution at the locations
    # of the training data, in which case we can save time by setting the predictive
    # basis matrices equal to the training basis matrices, rather than calculating
    # them all over again:
    if vbr_object.model_basis_inputs_pred==vbr_object.model_basis_inputs_train:
        vbr_object.phi_pred_unnorm = vbr_object.phi_train_unnorm        
        vbr_object.phi_pred_norm = vbr_object.phi_train_norm
        vbr_object.n_data_pred = vbr_object.n_data_train
    else:
        # Otherwise, we will need to construct a completely new predictive basis
        # matrix. To do this, we need to start off with the standard basis functions:
        pred_matrix_unnorm = vbr_utilities.construct_basis_matrix( vbr_object, whichtype='pred' )
        # The previous line will have generated a normalised matrix, but without any
        # extra appendages that might have also been added to the training basis matrix:
        pred_matrix_norm = vbr_object.phi_pred_norm
        # So we need to add any appendages separately:
        if vbr_object.phi_appendages_postnorm!=None:
            pred_matrix_norm = np.column_stack( [ vbr_object.phi_appendages_postnorm, pred_matrix_norm ] )
        # And at last, we update the VBR object, because at this point we have our
        # finished predictive basis matrix:
        vbr_object.phi_pred_norm = pred_matrix_norm
    # Now we calculate the means and stdvs of the predictive distribution:
    wm = np.matrix( vbr_object.model_weights_means.flatten() ).T
    phi = np.matrix( vbr_object.phi_pred_norm )
    smatrix = np.matrix( vbr_object.model_smatrix )
    # Determine whether or not the white noise term was fixed:
    if vbr_object.model_beta_fixed==True:
        cn = None
        dn = None
        beta_fixed = vbr_object.model_beta_mean_norm
    else:
        cn = vbr_object.model_cn
        dn = vbr_object.model_dn
        beta_fixed = None
    # This routine does the actual calculations: 
    means_pred_norm, stdvs_pred_norm = vbr_utilities.calc_predictive_dist( wm, phi, smatrix, beta_fixed=beta_fixed, cn=cn, dn=dn )
    # We need to unnormalise the values for the predictive distribution to get them back
    # in the same units for comparison with the original input data; if we were fitting in
    # log units, this step will also produce output in 'unlogified' format:
    vbr_utilities.unnormalise_model_outputs( vbr_object, means_pred_norm, stdvs_pred_norm )
    vbr_object.predictive_distribution_status = 'Yes'
    # Print the inferred white noise value to screen:
    if vbr_object.model_beta_fixed==True:
        print '   Fixed whitenoise = %.6f' % (vbr_object.model_whitenoise_mean_unnorm)
    else:
        print '   Inferred whitenoise = %.6f' % (vbr_object.model_whitenoise_mean_unnorm)
    # Finally, check to see if the predictive distribution is evaluated at the same
    # locations as the training data, and if so, print the 1 and 2 sigma information:
    if vbr_object.model_basis_inputs_pred==vbr_object.model_basis_inputs_train:
        vbr_utilities.check_sigmalimits( vbr_object )
        vbr_object.pred_at_train_locations_status = 'Yes'
    else:
        vbr_object.pred_at_train_locations_status = 'No'
    return None


def disentangle_basis_contributions(vbr_object, make_plots=True, unlogify_plots=True, abcissa_values=None, abcissa_label=None, ordinate_label=None, y_zoom=False):
    """
    This routine takes the inferred parameter distributions and uses them to separate contributions
    from subsets of basis functions within the overall basis model. Specifically, it will separate
    the contributions from each group of basis functions as they're organised within the variables:
     ** model_basis_group_names, model_appendage_names
     ** model_basis_types, model_basis_kwargs, 
     ** model_basis_inputs_train, model_basis_inputs_pred
     ** phi_ixs_basis_groups, phi_ixs_appendages_postnorm

    The following output variables are generated as object attributes:
    ** basis_groups_pred_means_unnorm, basis_groups_pred_stdvs_unnorm    
    ** basis_appendages_pred_means_unnorm, basis_appendages_pred_stdvs_unnorm
    Each of these variables contains a list, with the ith entry corresponding to the contribution
    from ith basis function group or the ith appendage.

    In addition to these outputs, a 'mirror' attribute is generated with a '_complements' suffix.
    Specifically, these complement variables are the complementary contribution from all other basis
    functions and appendages.

    If the target_log_units attribute is set to True, 'unlogified' versions of each variable will
    also be generated.

    Plots of each basis contribution can be made depending on how the optional arguments
    of this routine are specified; they are fairly self-explanatory.
    """
    # First do the standard basis groups:
    try:
        # Work out the number of groups:
        n_basis_groups = len(vbr_object.phi_ixs_basis_groups)
        if vbr_object.disentangle_status=='Yes':
            print '\nBasis group contributions have already been disentangled.'
        else:
            # Set up lists to hold the output arrays:
            group_names = []
            group_means = []
            group_stdvs = []
            # Also keep track of the complement distributions:
            group_complement_means = []
            group_complement_stdvs = []        
            # If the target data is specified as being in log units, set up additional lists to hold
            # the output arrays that have been converted from log to linear units:
            if vbr_object.target_log_units==True:
                group_means_unlogified = []
                group_stdvs_unlogified = []
                group_complement_means_unlogified = []
                group_complement_stdvs_unlogified = []            
            # Disentangle the basis contributions, one at a time:
            for i in range(n_basis_groups):
                print '\n Disentangling basis group %i of %i...' % (i+1,n_basis_groups)
                ixs_group = vbr_object.phi_ixs_basis_groups[i]
                subset_unnorm, complement_unnorm = vbr_utilities.split_basis_model(vbr_object, ixs_group)
                # Record the marginalised distribution for the group subset:
                group_names = group_names+[vbr_object.model_basis_group_names[i]]
                group_means = group_means+[subset_unnorm[0]]
                group_stdvs = group_stdvs+[subset_unnorm[1]]
                # Also record the means of the complement, for the purposes of plotting below:
                group_complement_means = group_complement_means+[complement_unnorm[0]]
                group_complement_stdvs = group_complement_stdvs+[complement_unnorm[1]]
                # If we have log units, also record the results in linear units:
                if vbr_object.target_log_units==True:
                    # Work out the arrays for the current step:
                    group_means_unlogified_i, group_stdvs_unlogified_i \
                        = vbr_utilities.unlogify_distribution(group_means[i], group_stdvs[i])
                    group_complement_means_unlogified_i, group_complement_stdvs_unlogified_i \
                        = vbr_utilities.unlogify_distribution(group_complement_means[i], group_complement_stdvs[i])
                    #group_complement_means_unlogified_i = np.exp(group_complement_means[i])
                    # Add the arrays for the current step to the list:
                    group_means_unlogified = group_means_unlogified+[group_means_unlogified_i]
                    group_stdvs_unlogified = group_stdvs_unlogified+[group_stdvs_unlogified_i]
                    group_complement_means_unlogified = group_complement_means_unlogified+ \
                                                        [group_complement_means_unlogified_i]
                    group_complement_stdvs_unlogified = group_complement_stdvs_unlogified+ \
                                                        [group_complement_stdvs_unlogified_i]
            # Install the output in the vbr object:
            vbr_object.basis_groups_pred_means_unnorm = group_means
            vbr_object.basis_groups_pred_stdvs_unnorm = group_stdvs
            vbr_object.basis_groups_pred_means_unnorm_complements = group_complement_means        
            vbr_object.basis_groups_pred_stdvs_unnorm_complements = group_complement_stdvs        
            if vbr_object.target_log_units==True:
                vbr_object.basis_groups_pred_means_unnorm_unlogified = group_means_unlogified
                vbr_object.basis_groups_pred_stdvs_unnorm_unlogified = group_stdvs_unlogified
                vbr_object.basis_groups_pred_means_unnorm_unlogified_complements = group_complement_means_unlogified
                vbr_object.basis_groups_pred_stdvs_unnorm_unlogified_complements = group_complement_stdvs_unlogified
    except:
        n_basis_groups = 0
        print ' \nWARNING: No basis groups found for model!\n'
        vbr_object.basis_groups_pred_means_unnorm = None
        vbr_object.basis_groups_pred_means_unnorm_complement = None        
        vbr_object.basis_groups_pred_stdvs_unnorm = None
    # Now do the same as above for any special appendages that have been included in the model:
    try:
        n_appendages = len(vbr_object.phi_ixs_appendages_postnorm)
        if vbr_object.disentangle_status=='No':
            appendage_names = []
            appendage_means = []
            appendage_stdvs = []
            appendage_complement_means = []
            if vbr_object.target_log_units==True:
                appendage_means_unlogified = []
                appendage_stdvs_unlogified = []
                appendage_complement_means_unlogified = []
            for i in range(n_appendages):
                print '\n Disentangling appendage %i of %i...' % (i+1,n_appendages)
                ixs_appendage = vbr_object.phi_ixs_appendages_postnorm[i]
                subset_unnorm, complement_unnorm = vbr_utilities.split_basis_model(vbr_object, ixs_appendage)
                appendage_names = appendage_names+[vbr_object.model_appendage_names]
                appendage_means = appendage_means+[subset_unnorm[0]]
                appendage_stdvs = appendage_stdvs+[subset_unnorm[1]]
                appendage_complement_means = appendage_complement_means+[complement_unnorm[0]]
                if vbr_object.target_log_units==True:
                    # Work out the arrays for the current step:
                    appendage_means_unlogified_i, appendage_stdvs_unlogified_i \
                        = vbr_utilities.unlogify_distribution(appendage_means[i], appendage_stdvs[i])
                    appendage_complement_means_unlogified_i = np.exp(appendage_complement_means[i])
                    # Add the arrays for the current step to the list:
                    appendage_means_unlogified = appendage_means_unlogified+[appendage_means_unlogified_i]
                    appendage_stdvs_unlogified = appendage_stdvs_unlogified+[appendage_stdvs_unlogified_i]
                    appendage_complement_means_unlogified = appendage_complement_means_unlogified+ \
                                                            [appendage_complement_means_unlogified_i]
            vbr_object.basis_appendages_pred_means_unnorm = appendage_means
            vbr_object.basis_appendages_pred_means_unnorm_complements = appendage_complement_means        
            vbr_object.basis_appendages_pred_stdvs_unnorm = appendage_stdvs
            if vbr_object.target_log_units==True:
                vbr_object.basis_appendages_pred_means_unnorm_unlogified = appendage_means_unlogified
                vbr_object.basis_appendages_pred_means_unnorm_unlogified_complements = appendage_complement_means_unlogified
                vbr_object.basis_appendages_pred_stdvs_unnorm_unlogified = appendage_stdvs_unlogified
    except:
        n_appendages = 0
        print ' \n         No appendages found for model\n'
        vbr_object.basis_appendages_pred_means_unnorm = None
        vbr_object.basis_appendages_pred_means_unnorm_complements = None        
        vbr_object.basis_appendages_pred_stdvs_unnorm = None
    vbr_object.disentangle_status = 'Yes'
    # Plot the results if requested:
    if make_plots==True:
        print 'Plotting separate basis group contributions...'
        # Prepare the data for plotting:
        pred_means = np.zeros([vbr_object.n_data_pred, n_basis_groups+n_appendages])
        pred_stdvs = np.zeros([vbr_object.n_data_pred, n_basis_groups+n_appendages])
        corrected_data = np.zeros([vbr_object.n_data_pred, n_basis_groups+n_appendages])
        axes_titles = []
        if n_basis_groups>0:
            for i in range(n_basis_groups):
                if (vbr_object.target_log_units==True)*(unlogify_plots==True):
                    pred_means[:,i] = vbr_object.basis_groups_pred_means_unnorm_unlogified[i]
                    pred_stdvs[:,i] = vbr_object.basis_groups_pred_stdvs_unnorm_unlogified[i]
                    corrected_data[:,i] = np.exp(vbr_object.target_train_unnorm - \
                                          vbr_object.basis_groups_pred_means_unnorm_complements[i])
                else:
                    pred_means[:,i] = vbr_object.basis_groups_pred_means_unnorm[i]
                    pred_stdvs[:,i] = vbr_object.basis_groups_pred_stdvs_unnorm[i]
                    corrected_data[:,i] = vbr_object.target_train_unnorm- \
                                          vbr_object.basis_groups_pred_means_unnorm_complements[i]
                axes_titles = axes_titles+[vbr_object.model_basis_group_names[i]]
        if n_appendages>0:
            for i in range(n_appendages):
                if (vbr_object.target_log_units==True)*(unlogify_plots==True):
                    pred_means[:,i+n_basis_groups] = vbr_object.basis_appendages_pred_means_unnorm_unlogified[i]
                    pred_stdvs[:,i+n_basis_groups] = vbr_object.basis_appendages_pred_stdvs_unnorm_unlogified[i]
                    corrected_data[:,i+n_basis_groups] = np.exp(vbr_object.target_train_unnorm_unlogified - \
                                                                vbr_object.basis_appendages_pred_means_unnorm_unlogified_complements[i])
                else:
                    pred_means[:,i+n_basis_groups] = vbr_object.basis_appendages_pred_means_unnorm[i]
                    pred_stdvs[:,i+n_basis_groups] = vbr_object.basis_appendages_pred_stdvs_unnorm[i]
                    corrected_data[:,i+n_basis_groups] = vbr_object.target_train_unnorm-vbr_object.basis_appendages_pred_means_unnorm_complements[i]
                axes_titles = axes_titles+[vbr_object.model_appendage_names[i]]
        # Now set up the plotting:
        shade = [0.5,0.5,0.5]
        if abcissa_values==None:
            x = np.arange(vbr_object.n_data_pred)
        else:
            x = abcissa_values
        # Work out how the subplots will be divided between figures:
        n_axes_perfig = 4 
        n_axes_total = n_basis_groups+n_appendages 
        n_figs = np.ceil(n_axes_total/float(n_axes_perfig)) 
        # Get the minimum and maximum extents of the data:
        x_low = x.min()
        x_upp = x.max()
        y_min = np.max([(pred_means-pred_stdvs).min(),corrected_data.min()])
        y_max = np.max([(pred_means+pred_stdvs).max(),corrected_data.max()])
        # Specify the axes properties (vertical position specified below):
        edge_buffer = 0.08 
        ax_xlow = 1.5*edge_buffer
        ax_width = 1-2*edge_buffer
        ax_height = (1-2*edge_buffer)/float(n_axes_perfig)
        axes_counter = 0
        fig_counter = 0
        for i in range(n_axes_total):
            # Specify the vertical position for the current axis:
            ax_ylow = 1-(axes_counter%n_axes_perfig+1)*ax_height-edge_buffer
            if axes_counter%n_axes_perfig==0:
                fig = plt.figure()
                fig_counter += 1
                fig.suptitle('Basis Contributions (%i of %i)' % (fig_counter,n_figs))
                ax0 = fig.add_axes([ax_xlow,ax_ylow,ax_width,ax_height])
            else: 
                ax = fig.add_axes([ax_xlow,ax_ylow,ax_width,ax_height], sharex=ax0)
            cax = plt.gca()
            cax.plot(x,corrected_data[:,i],'.g',ms=10,alpha=0.7)  
            cax.fill_between(x,pred_means[:,i]-pred_stdvs[:,i],pred_means[:,i]+pred_stdvs[:,i],color=shade)
            cax.plot(x,pred_means[:,i],'-r',lw=2)
            cax.set_xlim([x_low,x_upp])
            # Define the minimum and maximum extents of the current subplot axes:
            if y_zoom==True:
                y_low = corrected_data[:,i].min()
                y_upp = corrected_data[:,i].max()
            else:
                y_low = y_min-0.1*(y_max-y_min)
                y_upp = y_max+0.1*(y_max-y_min)
            cax.set_ylim([y_low,y_upp])
            # Specify where to locate the text within the subplot axis:
            x_text = x_low+0.05*(x_upp-x_low)
            y_text = y_upp-0.1*(y_upp-y_low)
            cax.text(x_text,y_text,axes_titles[i],fontsize=12)
            if axes_counter%n_axes_perfig!=n_axes_perfig-1:
                plt.setp(cax.xaxis.get_ticklabels(), visible = False)
            else:
                if abcissa_label!=None:
                    cax.set_xlabel(abcissa_label)
            axes_counter += 1
        plt.draw()
    print '\n'
    return None

def basis_subset_contribution( vbr_object, make_plots=True ):
    """
    TO DO - MAYBE?
    """
    # similar to 'disentangle_basis_groups', but for an arbitrary combination of basis functions.
    #vbr_utilities.divide_basis_model(...)
    print '\n\nNot implemented yet!\n\n'
    return None

def get_marginalised_loglikelihood( vbr_object ):
    """
    Evaluates the loglikelihood of the data given the model, marginalised over all parameters of the
    model. In other words, it is the joint pdf of the predictive distribution evaluated at the locations
    of the input data. The output is stored in the object attribute 'marginalised_loglikelihood'.

    If the white noise beta parameter was free, this joint pdf should strictly speaking be the product
    of Student's t distributions (see Equation 31 of the 2008 report by Drugowitsch), one for each data
    point. However, here we approximate the joint pdf as a multivariate Gaussian with diagonal covariance
    function.

    This is a good approximation, provided the number of degrees of freedom of the original Student's t
    distributions is large, i.e. the parameter cn is large.
    """
    # Make a copy of the object so that we can force the predictive locations to be the
    # same as the locations of the input training data:
    temp_object = copy.deepcopy( vbr_object )
    # Check to see if we need to reevaluate the predictive distribution at the locations
    # of the training data, or if this has been done already:
    if temp_object.pred_at_train_locations_status=='No':
        temp_object.model_basis_inputs_pred = temp_object.model_basis_inputs_train
        if temp_object.regression_run_status=='No':
            temp_object.do_linear_regression()
    # Get the number of data points:
    ndata = temp_object.n_data_train
    # Get the training data:
    data = temp_object.target_train_unnorm
    # We are approximating the marginalised likelihood as a multivariate normal distribution, although
    # strictly speaking it's a multivariate Student's t distribution. Proceeding with our approximation,
    # specify the mean vector and covariance matrix:
    mean_vector = temp_object.model_pred_means_unnorm
    stdv_vector = temp_object.model_pred_stdvs_unnorm
    # Evaluate the log pdf of the multivariate normal marginalised likelihood distribution, and install
    # this value in the original object:
    vbr_object.logp = np.sum( np.log( 1./stdv_vector ) ) + \
                      np.sum( -( ( data-mean_vector )**2. ) / 2. / ( stdv_vector**2. ) )
    return None


def plot_beta( vbr_object ):
    """
    Plots the posterior distribution over beta.
    """
    if vbr_object.model_beta_fixed==True:
        print '\n\nBeta parameter was fixed (i.e. delta distribution)\n\n'
    else:
        shape = vbr_object.model_cn
        rate = vbr_object.model_dn
        sig0 = 1. / np.sqrt( vbr_object.model_beta_mean_norm )
        sig_x = np.r_[ 1e-6:3*sig0:1j*1000 ]
        beta_x = 1./(sig_x**2.)
        term1 = np.log(1.) - scipy.special.gammaln(shape)
        term2 = shape * np.log( rate )
        term3 = (shape-1) * np.log( beta_x )
        term4 = -rate*beta_x
        beta_logpdf = term1 + term2 + term3 + term4
        plt.figure()
        plt.plot( sig_x/sig0, np.exp(beta_logpdf), '-g', lw=2 )
        plt.xlabel('Normalised white noise value')
        plt.ylabel('Probability')
        plt.title('Inferred White Noise Distribution')
    return None
