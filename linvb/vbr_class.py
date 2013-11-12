import numpy as np
import vbr_utilities, vbr_routines
import pdb

class vbr():
    """
    Class for a variational Bayesian regression (VBR). Objects store the input and
    target data as attributes and have methods to perform linear regression on those
    data at specified locations in the input space. 
    """
    def __init__( self, inputs_train=None, target_train=None, target_log_units=False ):
        """
        Initialise a new VBR object. Default values for most attributes is NoneType.
        The only attributes that can be set at the __init__ call are the training inputs
        and/or training target data, being passed in as optional arguments. Another
        notable exception is the model_add_offset attribute, which is set to True by
        default (i.e. a column of 1's will be added to the basis matrix unless this
        is set to False).
        """
        # Input and target data:
        self.model_basis_inputs_train = inputs_train
        self.target_train_unnorm = target_train
        self.target_log_units = target_log_units
        self.n_data_train = None
        # Basis function groups:
        self.model_basis_types = None
        self.model_basis_kwargs = None
        self.model_add_offset = True
        self.n_basis_groups = None
        self.n_basis_funcs = None
        # Unnormalised basis matrix, both training and predictive:
        self.phi_train_unnorm = None
        self.phi_pred_unnorm = None
        # Normalised basis matrix with normalisation
        # terms:
        self.phi_train_norm = None
        self.phi_train_norm_scalings = None
        self.phi_train_norm_shifts = None
        self.phi_pred_norm = None
        # Any special blocks, eg. transit functions, that were appended
        # to the basis matrix **after** normalisation:
        self.phi_appendages_postnorm = None
        # Normalised target data with normalisation
        # terms:
        self.target_train_norm = None
        self.target_train_norm_shift = None        
        self.target_train_norm_scaling = None
        # Variables related to output from the regression:
        self.model_basis_inputs_pred = None
        self.means_pred_unnorm = None
        self.stdvs_pred_unnorm = None
        self.model_whitenoise_mean_unnorm = None
        self.model_weights_means = None
        self.model_weights_stdvs = None
        self.model_alpha = None
        self.model_beta_fixed = False
        self.model_beta_mean_norm = None
        self.model_beta_stdv_norm = None
        self.model_smatrix = None
        self.model_an = None
        self.model_bn = None
        self.model_cn = None
        self.model_dn = None
        self.fraction_1sigma = None
        self.fraction_2sigma = None
        # Status flags:
        self.training_basis_matrix_unnorm_status = 'No'        
        self.training_basis_matrix_norm_status = 'No'
        self.regression_run_status = 'No'
        self.predictive_distribution_status = 'No'
        self.pred_at_train_locations_status = 'No'
        self.disentangle_status = 'No'

    def set_attributes( self, target_train_unnorm=None, target_log_units=None, model_whitenoise_mean_unnorm=None, model_basis_group_names=None, model_basis_types=None, model_basis_inputs_train=None, model_basis_inputs_pred=None, model_basis_kwargs=None, model_add_offset=None ):
        """
        Function that allows various VBR object attributes to be set manually in preparation for the regression.
        However, these properties should not be tampered with at any point after the regression; if they are, then
        the regression should be performed again. To-do: add a feature that prevents attributes being tampered
        with externally, i.e. making the set_attributes() routine the only possible way to change attribute values,
        because it sets all of the status variables back to 'No' to notify that the analysis needs to be redone
        with the new attribute values.
        """
        if target_train_unnorm!=None: self.target_train_unnorm = target_train_unnorm
        if target_log_units!=None: self.target_log_units = target_log_units
        # Specify if we will be inferring the white noise term or holding it fixed to some
        # specified value:
        if model_whitenoise_mean_unnorm!=None:
            self.model_whitenoise_mean_unnorm = model_whitenoise_mean_unnorm
            self.model_beta_fixed = True
        if model_basis_group_names!=None: self.model_basis_group_names = model_basis_group_names        
        if model_basis_types!=None: self.model_basis_types = model_basis_types
        if model_basis_inputs_train!=None: self.model_basis_inputs_train = model_basis_inputs_train
        if model_basis_inputs_pred!=None: self.model_basis_inputs_pred = model_basis_inputs_pred
        if model_basis_kwargs!=None: self.model_basis_kwargs = model_basis_kwargs
        if model_add_offset!=None: self.model_add_offset = model_add_offset
        # If anything has changed, we should set the flags to warn that the regression needs
        # to be redone and/or the predictive distribution needs to be reevaluated:
        if (target_train_unnorm!=None)+(target_log_units!=None)+(model_basis_types!=None)+(model_basis_inputs_train!=None) \
        +(model_basis_inputs_pred!=None)+(model_basis_kwargs!=None)+(model_add_offset!=None):
            self.regression_run_status = 'No'
            self.predictive_distribution_status = 'No'
            self.pred_at_train_locations_status = 'No'
            self.disentangle_status = 'No'

    def describe( self ):
        """
        Print information about the current state of the VBR object, including
        whether or not the basis matrix has been constructed, if it has been
        normalised, if a regression has been performed etc.
        """
        vbr_utilities.describe( self )

    def construct_basis_matrix( self, whichtype='train' ):
        """
        Construct the core basis matrix, both normalised and unnormalised. A few
        details on this task:
          - The whichtype keyword argument specifies whether or not the training
            or predictive basis matrix will be constructed. In practice, the only
            difference between these two options is in the former the
            model_basis_inputs_train will be used as input to the basis functions
            whereas in the latter case the model_basis_inputs_pred will be used
            instead.
          - The basis model information itself is contained in the model_basis_types,
            and model_basis_kwargs variables. These remain the same regardless of
            whether whichtype is set to 'train' or 'pred'.
          - If an offset column of 1's is added to the basis matrix, it will be
            done so only after the other columns have been normalised. Hence, only
            the phi_train_norm and phi_pred_norm basis matrices will contain a
            first column of 1's.
          - Only those basis functions specified in the model_basis_types variable
            will be used to construct the core basis matrix, as well as an offset
            term if model_add_offset is set to True (as is the default).
          - Any additional specialised basis functions (eg. transit functions) should
            be added after this step using the append_training_basis_matrix() task.
        """
        vbr_utilities.construct_basis_matrix( self, whichtype=whichtype )

    def append_training_basis_matrix( self, appendage, appendage_name=None ):
        """
        Appends an NxK array ('appendage') to the NxM training basis matrix.
        This is useful for adding in specialised basis functions (eg. transit
        light curve functions) after the rest of the basis has
        been set.

        It should be emphasised that it only appends to the **normalised** basis
        matrix. The thinking behind this is that you'll only be wanting to append
        specialised basis functions for the fitting, and it is only the normalised
        basis matrix that is used for the fitting.

        Also, appending extra blocks is best done separately, rather than within
        the construct_basis_matrix() task. This is because it makes it possible to
        append things later on without having to reconstruct the entire basis matrix
        again from scratch.

        Lastly, this routine only adds the appendage to the training basis matrix.
        However, it records the relevant information, so that when construct_basis_matrix()
        is called for the predictive basis matrix, the appendage will be automatically
        appended to it without having to pass to this routine.
        """
        vbr_utilities.append_training_basis_matrix( self, appendage, appendage_name=appendage_name )

    def do_linear_regression( self, convergence=0.00001 ):
        """
        Once the normalised training basis matrix and target data has been set up, this routine
        simply runs the VB linear regression algorithm. Values for the model parameters and
        hyperparameters are inferred, including for the linear weights of the model.
        """
        self.vb_convergence = convergence
        vbr_routines.do_linear_regression(self, convergence=convergence)

    def get_predictive_distribution( self, model_basis_inputs_pred=None ):
        """
        Computes the predictive distribution (i.e. means and standard deviations)
        inferred by the do_linear_regression() routine at the locations in input
        space specified by the model_basis_inputs_pred variable.
        """
        # If requested, evaluate the predictive distribution at
        # the user-specified locations:
        if model_basis_inputs_pred!=None:
            self.model_basis_inputs_pred = model_basis_inputs_pred
        # Otherwise, the default is to evaluate the predictive
        # distribution at the locations of the input training
        # data:
        else:
            self.model_basis_inputs_pred = self.model_basis_inputs_train
        vbr_routines.get_predictive_distribution(self)


    def disentangle_basis_contributions( self, make_plots=True, abcissa_values=None, abcissa_label=None, ordinate_label=None, y_zoom=False ):
        """
        Extracts the model contributions for each basis group separately, and plots
        them if requested. For example, suppose our linear model has the form:

         w1*phi1 + w2*phi2 + w3*phi3 = t

        then this routine will decompose the signal into:

          w1*phi1 = t - ( w2*phi2 + w3*phi3 )
          w2*phi1 = t - ( w1*phi2 + w3*phi3 )
          w3*phi1 = t - ( w1*phi2 + w2*phi3 )

        The intention is to isolate the individual contributions from each basis to
        the overall signal. However, the VB algorithm might not be suitable for inferring
        marginalised posterior distributions (see one of the Roberts & Penny papers).
        """
        vbr_routines.disentangle_basis_contributions( self, make_plots=make_plots, abcissa_values=abcissa_values, abcissa_label=abcissa_label, ordinate_label=ordinate_label, y_zoom=y_zoom )

    def basis_subset_contribution( self, make_plot=True ):
        """
        Similar to disentangle_basis_contributions(), except for an arbitrary combination
        of basis function groups. TO DO, and I'm not actually sure it's worth it.
        """
        vbr_routines.basis_subset_contribution( self, make_plot=make_plot )

    def get_marginalised_loglikelihood( self ):
        """
        Computes the marginalised likelihood of the model, which is a useful quantity
        for performing Bayesian comparisons of different models. 
        """
        vbr_routines.get_marginalised_loglikelihood( self )

    
        

        

