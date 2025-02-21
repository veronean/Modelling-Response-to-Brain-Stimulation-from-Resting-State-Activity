# Import Libraries and Data
import numpy as np
import scipy.io
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
import time
import os
import pickle
import gdown
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import erf
import pathlib as Path

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
torch.cuda.reset_accumulated_memory_stats()

import sys

#neuroimaging packages
import mne

# Suppression block
class Suppressor:
  def __enter__(self):
    self.stdout_original = sys.stdout
    sys.stdout = open('/dev/null', 'w')


  def __exit__(self, exc_type, exc_value, traceback):
    sys.stdout.close()
    sys.stdout = self.stdout_original

class par:
    def __init__(self, val, prior_mean=None, prior_var=None, fit_par=False, fit_hyper=False, 
                 use_heterogeneity=False, h_map=None, device='cuda'):
        '''
        Parameters
        ----------
        val : Float
            The parameter value
        prior_mean : Float or Tuple
            If use_heterogeneity is False: prior_mean for the parameter.
            If use_heterogeneity is True: (prior_min_mean, prior_scale_mean).
        prior_var : Float or Tuple
            If use_heterogeneity is False: prior_var for the parameter.
            If use_heterogeneity is True: (prior_min_var, prior_scale_var).
        fit_par : Bool
            Whether the parameter value should be set as a PyTorch Parameter
        fit_hyper : Bool
            Whether the prior mean and prior variance should be set as a PyTorch Parameter
        use_heterogeneity : Bool
            Whether this parameter should be computed using a heterogeneity map.
        h_map : Tensor (Optional)
            The heterogeneity map to be used (should match spatial dimensions of the parameter).
        '''
        self.device = device
        self.use_heterogeneity = use_heterogeneity
        self.h_map = torch.tensor(h_map, dtype=torch.float32, device=device) if h_map is not None else None

        if use_heterogeneity:
            if not isinstance(prior_mean, tuple) and not isinstance(prior_var, tuple):
                raise ValueError("For heterogeneity, prior_mean and prior_var must be tuples (min, scale)")
            if np.all(prior_mean != None) and np.all(prior_var != None):
                self.has_prior = True
            else:
                self.has_prior = False
                prior_mean = (0, 0)
                prior_var = (0, 0)
                       
            self.val_min = torch.tensor(val, dtype=torch.float32, device=device)
            self.val_scale = torch.tensor(0.05, dtype=torch.float32, device=device)
            
            self.prior_min_mean = torch.tensor(prior_mean[0], dtype=torch.float32, device=device)
            self.prior_scale_mean = torch.tensor(prior_mean[1], dtype=torch.float32, device=device)
            self.prior_min_var = torch.tensor(prior_var[0], dtype=torch.float32, device=device)
            self.prior_scale_var = torch.tensor(prior_var[1], dtype=torch.float32, device=device)
            
            self.fit_par = fit_par
            self.fit_hyper = fit_hyper

            if fit_par:
                self.val_min = nn.Parameter(self.val_min)
                self.val_scale = nn.Parameter(self.val_scale)
            if fit_hyper:
                self.prior_min_mean = nn.Parameter(self.prior_min_mean)
                self.prior_min_var = nn.Parameter(self.prior_min_var)
                self.prior_scale_mean = nn.Parameter(self.prior_scale_mean)
                self.prior_scale_var = nn.Parameter(self.prior_scale_var)
            
        else:
            if np.all(prior_mean != None) & np.all(prior_var != None):
                self.has_prior = True
            elif np.all(prior_mean != None) | np.all(prior_var != None):
                raise ValueError("prior_mean and prior_var must either be both None or both set")
            else:
                self.has_prior = False
                prior_mean = 0
                prior_var = 0

            self.val = torch.tensor(val, dtype=torch.float32).to(device)
            
            self.prior_mean = torch.tensor(prior_mean, dtype=torch.float32, device=device)
            self.prior_var = torch.tensor(prior_var, dtype=torch.float32, device=device)

            self.fit_par = fit_par
            self.fit_hyper = fit_hyper
            
            if fit_par:
                self.val = nn.Parameter(self.val)
            if fit_hyper:
                self.prior_mean = nn.Parameter(self.prior_mean)
                self.prior_var = nn.Parameter(self.prior_var)
     
    def value(self):
        '''
        Returns
        ---------
        Tensor of Value
            The parameter value(s) as a PyTorch Tensor
        '''
        if self.use_heterogeneity and self.h_map is not None:
            w = self.val_min + self.val_scale * self.h_map
            return w
        else:
            return self.val

    def npValue(self):
        '''
        Returns
        --------
        NumPy of Value
            The parameter value(s) as a NumPy Array
        '''
        return self.value().detach().clone().cpu().numpy()
        
    def to(self, device='cuda'):
        self.device = device

        if self.use_heterogeneity:
            if self.fit_par:
                self.val_min = nn.Parameter(self.val_min.detach().clone().to(device))
                self.val_scale = nn.Parameter(self.val_scale.detach().clone().to(device))
            else:
                self.val_min = self.val_min.to(device)
                self.val_scale = self.val_scale.to(device)

            if self.fit_hyper:
                self.prior_min_mean = nn.Parameter(self.prior_min_mean.detach().clone().to(device))
                self.prior_min_var = nn.Parameter(self.prior_min_var.detach().clone().to(device))
                self.prior_scale_mean = nn.Parameter(self.prior_scale_mean.detach().clone().to(device))
                self.prior_scale_var = nn.Parameter(self.prior_scale_var.detach().clone().to(device))
            else:
                self.prior_min_mean = self.prior_min_mean.to(device)
                self.prior_min_var = self.prior_min_var.to(device)
                self.prior_scale_mean = self.prior_scale_mean.to(device)
                self.prior_scale_var = self.prior_scale_var.to(device)
        else:
            if self.fit_par:
                self.val = nn.Parameter(self.val.detach().clone().to(device))
            else:
                self.val = self.val.to(device)

            if self.fit_hyper:
                self.prior_mean = nn.Parameter(self.prior_mean.detach().clone().to(device))
                self.prior_var = nn.Parameter(self.prior_var.detach().clone().to(device))
            else:
                self.prior_mean = self.prior_mean.to(device)
                self.prior_var = self.prior_var.to(device)

    def randSet(self):
        '''
        This method sets the initial value using the mean and variance of the priors.
        '''

        if self.use_heterogeneity:
            if self.has_prior:
                self.val_min = self.prior_min_mean.detach() + self.prior_min_var.detach() * torch.randn(1, device=self.device)
                self.val_scale = self.prior_scale_mean.detach() + self.prior_scale_var.detach() * torch.randn(1, device=self.device)
                if self.fit_par:
                    self.val_min = nn.Parameter(self.val_min)
                    self.val_scale = nn.Parameter(self.val_scale)
            else:
                raise ValueError("must have priors provided at par object initialization to use this method")
        else:
            if self.has_prior:
                self.var = self.prior_mean.detach() + self.prior_var.detach() * torch.randn(1, device=self.device)
                if self.fit_par:
                    self.val = torch.nn.parameter.Parameter(self.val)
            else:
                raise ValueError("must have priors provided at par object initialization to use this method")

    def __pos__(self):
        return self.value()

    def __neg__(self):
        return -self.value()

    def __add__(self, num):
        return self.value() + num

    def __radd__(self, num):
        return num + self.value()

    def __sub__(self, num):
        return self.value() - num

    def __rsub__(self, num):
        return num - self.value()

    def __mul__(self, num):
        return self.value() * num

    def __rmul__(self, num):
        return num * self.value()

    def __truediv__(self, num):
        return self.value() / num

    def __rtruediv__(self, num):
        return num / self.value()

class AbstractParams:
    def __init__(self, **kwargs):
        pass

    def to(self, device):
        vars_names = [a for a in dir(self) if not a.startswith('__')]
        for var_name in vars_names:
            var = getattr(self, var_name)
            if (type(var) == par):
                var.to(device)

class ParamsHP(AbstractParams):
    """
    Class for setting the parameters of the neural mass model.

    Attributes:
        a (par): The node's bifurcation parameter (s-1).
        omega (par): The intrinsic angular frequency (in rad.s-1).
        sig_omega (par): Variance of the angular frequency.
        std_in (par): The standard deviation of the input noise.
        g (par): Global Connectivity Scaling.
        v_d (par): Conduction Velocity.
        cy0 (par): Leadfield Matrix scaling parameter.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ParamsJR object.

        Args:
            **kwargs: Keyword arguments for the model parameters.

        Returns:
            None
        """
        param = {
            "a": par(-0.5),
            "omega": par(10.0),
            "sig_omega": par(3.0),
            "g": par(500),

            "std_in": par(100),

            "v_d": par(1.0),
            "cy0": par(50),
        }

        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])

class TrainingStats:
    '''
    This class is responsible for recording stats during training including:
        - The training and validation losses over time
        - The change in model parameters over time

    These are things typically recorded on a per epoch basis

    Attributes
    ------------
    model_info : Dict
        Information about model being tracked during training.
    track_params : List
        List of parameter names being tracked during training.
    loss : List
        A list of loss values over training.
    connectivity : List
        A list of connectivity values over training.
    leadfield : List
        A list of leadfield matrices over training.
    w_ll : List
        A list of connection gain matrices over training.
    fit_params : Dict
        A dictionary of lists where the key is the parameter name and the value is the list of parameter values over training.
    '''
    def __init__(self, model):
        '''
        Parameters
        -----------
        model : AbstractNMM
            A model for which stats will be recorded during training.

        '''
        self.model_info = model.info()
        self.track_params = model.track_params

        self.loss = []

        self.connectivity = []
        self.w_ll = []
        self.leadfield = []

        self.fit_params = {}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def reset(self):
        self.loss = []

        self.connectivity = []
        self.w_ll = []
        self.leadfield = []
        
        self.fit_params = {}

    def appendLoss(self, newValue):
        self.loss.append(newValue)

    def appendCONN(self, newValue):
        self.connectivity.append(newValue)

    def appendWll(self, newValue):
        self.w_ll.append(newValue)

    def appendLF(self, newValue):
        self.leadfield.append(newValue)

    def appendParam(self, newValues):
        '''
        Append parameter values to the fit_params dictionary.
        If a parameter uses heterogeneity, store the mean value or the entire tensor.

        Parameters
        -----------
        newValues : Dict
            A dictionary where keys are parameter names and values are the parameter values.
        '''
        if not self.fit_params:  # If fit_params is empty
            for name, value in newValues.items():
                # Skip sig_omega if omega is heterogeneous
                if name == 'sig_omega' and self.model.parmas.omega.use_heterogeneity:
                    continue
                if isinstance(value, torch.Tensor):
                    self.fit_params[name] = [value.detach().clone().cpu().numpy()]
                else:
                    self.fit_params[name] = [value]
        else:
            for name, value in newValues.items():
                # Skip sig_omega if omega is heterogeneous
                if name == 'sig_omega' and self.model.parmas.omega.use_heterogeneity:
                    continue
                if isinstance(value, torch.Tensor): 
                    self.fit_params[name].append(value.detach().clone().cpu().numpy())
                else:
                    self.fit_params[name].append(value)
        

class AbstractNMM(torch.nn.Module):
    def __init__(self):
        super(AbstractNMM, self).__init__() 

        self.track_params = [] 

        self.use_fit_gains = False # Whether to fit the Connectivity Matrix
        self.use_fit_lfm = False # Whether to fit the Lead Field Matrix

    def info(self):
        return {"track_params": self.track_params}

    def setModelParameters(self):
        pass

    def forward(self, batch_size=10):
        pass

class AbstractLoss:
    def __init__(self, device = 'cuda'):
        self.device = device

    def loss(self, simData, empData, method):
        pass

class AbstractFitting():
    def __init__(self, model: AbstractNMM, cost: AbstractLoss, device = 'cuda'):
        self.model = model
        self.cost = cost
        self.device = device

        self.trainingStats = TrainingStats(self.model)
        self.valStats = {}
        self.testStats = {}
        self.lastRec = None
        self.sim = None
        self.psd = None

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train():
        pass

    def validate():
        pass

    def test():
        pass

    def simulate():
        pass

    def PSD():
        pass

class CostsTS(AbstractLoss):
    def __init__(self):
        super(CostsTS, self).__init__()

    def loss(self, empData: torch.Tensor, simData: torch.Tensor, method: str = 'mse'):
        """
        Computes loss between simulated and empirical covariance matrices. 

        Loss methods:
        - 'pearson': Pearson Correlation.
        - 'mse': Mean Squared Error (Frobenius norm squared).
        - 'log_loss': Logarithmic loss (log of Frobenius norm).
        - 'log_fro': Log-domain Frobenius norm.

        The loss is used to measure similarity or dissimilarity between covariance matrices.
        """

        sim = simData
        emp = empData

        if method == 'pearson':
            eps = 1e-8  
            sim = sim.flatten()
            emp = emp.flatten()
            
            sim_mean = sim.mean()
            emp_mean = emp.mean()
            
            sim_std = sim.std(unbiased=False)
            emp_std = emp.std(unbiased=False)

            pearson_corr = ((sim - sim_mean) * (emp - emp_mean)).mean() / (sim_std * emp_std + eps)
            return -pearson_corr 

        elif method == 'mse':
            # Mean Squared Error (Frobenius norm squared)
            return torch.norm(sim - emp, p='fro').pow(2) 

        elif method == 'log_loss':
            # Log Frobenius norm to dampen large differences
            frobenius_norm = torch.norm(sim - emp, p='fro') + 1e-8 
            return torch.log(frobenius_norm)

        elif method == 'log_fro':
            # Log-domain Frobenius norm (better handling of multiplicative differences)
            log_sim = torch.log(torch.clamp(sim, min=1e-8)) 
            log_emp = torch.log(torch.clamp(emp, min=1e-8))
            return torch.norm(log_sim - log_emp, p='fro') 

        else:
            raise ValueError(f"Invalid method '{method}'. Choose from 'mse', 'log_loss' and 'log_fro'.")

class COVHOPF(AbstractNMM):
    """
    Module for forward model (HOPF) to simulate covariance matrix.

    Attibutes
    ---------
    node_size : int
        Number of ROI
    output_size : int
        Number of EEG channels.
    tr : float
        Sampling rate of the EEG signals
    sc: ndarray (node_size x node_size) of floats
        Structural connectivity
    dist: ndarray of floats
        Distance matrix
    freqs: ndarray float array
        Discretization of the frequancy space
    params: par
        Parameters of the model
    lm: ndarray of floats
        Leadfield matrix from source space to EEG space
    w_ll_init: ndarray of floats
        Connection gain matrix (for effective connectivity)
    use_fit_gains: bool
        Flag for fitting gains. 1: fit, 0: not fit
    use_fit_lfm: bool
        Flag for fitting the leadfield matrix. 1: fit, 0: not fit
    Methods
    -------
    setModelParameters(self):
        Sets the parameters of the model.
    forward(self, breq_chunk_size, debug_sim)
        Compute the model covariance matrix
    """
    def __init__(self, node_size: int,
                output_size: int, tr: float, sc: np.ndarray, dist: np.ndarray, freqs: np.ndarray, 
                params: ParamsHP, lm: np.ndarray, w_ll_init: np.ndarray = None,
                use_fit_gains: bool = True, use_fit_lfm: bool = True) -> None:
        
        """
        Parameters
        ----------
        node_size: int
            Number of ROIs
        step_size: float
            Integration step for forward model
        output_size : int
            Number of EEG channels.
        tr : float
            Sampling rate of the simulated EEG signals
        sc: ndarray node_size x node_size float array
            Structural connectivity
        lm: ndarray float array
            Leadfield matrix from source space to EEG space
        dist: ndarray float array
            Distance matrix
        freqs: ndarray float array
            Discretization of the frequancy space
        use_fit_gains: bool
            Flag for fitting gains. 1: fit, 0: not fit
        use_fit_lfm: bool
            Flag for fitting the leadfield matrix. 1: fit, 0: not fit
        params: ParamsJR
            Model parameters object.
        """

        super(COVHOPF, self).__init__()
        self.device = 'cuda'
        self.track_params = [] #Is populated during setModelParameters()

        self.tr = tr 
        self.node_size = node_size          # num of ROI
        self.output_size = output_size      # num of EEG channels
        self.sc = sc                        # matrix node_size x node_size structure connectivity
        self.dist = torch.tensor(dist, dtype=torch.float32, device=self.device)
        self.lm = lm
        self.freqs = torch.tensor(freqs, dtype=torch.float32, device=self.device)
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        self.use_fit_lfm = use_fit_lfm      # flag for fitting leadfield matrix
        self.params = params

        if w_ll_init is None:
            w_ll_init = np.zeros((self.node_size, self.node_size)) + 0.05
            print("w_ll_init not provided, using default initialization for w_ll.")
        
        self.w_ll = w_ll_init

        self.setModelParameters()

    def setModelParameters(self):
        """
        Sets the parameters of the model.
        """

        param_reg = []
        param_hyper = []

        # Set w_ll as attributes as type Parameter if use_fit_gains is True
        if self.use_fit_gains:
            self.w_ll = nn.Parameter(torch.tensor(self.w_ll, dtype=torch.float32, device=self.device))
            param_reg.append(self.w_ll)
            print("Fitting gains: w_ll is set as a Parameter.")
        else:
            self.w_ll = torch.tensor(self.w_ll, dtype=torch.float32, device=self.device)
            print("NOT fitting gains: w_ll is a fixed tensor.")

        # If use_fit_lfm is True, set lm as an attribute as type Parameter (containing variance information)
        if self.use_fit_lfm:
            self.lm = nn.Parameter(torch.tensor(self.lm, dtype=torch.float32, device=self.device))
            param_reg.append(self.lm)
            print("Fitting leadfield matrix: lm is set as a Parameter.")
        else:
            self.lm = torch.tensor(self.lm, dtype=torch.float32, device=self.device)
            print("NOT fitting leadfield matrix: lm is a fixed tensor.")

        var_names = [a for a in dir(self.params) if (type(getattr(self.params, a)) == par)]
        for var_name in var_names:
            var = getattr(self.params, var_name)

            # Skip sig_omega if omega is heterogeneous
            if var_name == 'sig_omega' and self.params.omega.use_heterogeneity:
                continue
            
            if var.use_heterogeneity:
                if var.fit_hyper:
                    var.randSet()
                    param_hyper.extend([var.prior_min_mean, var.prior_min_var, var.prior_scale_mean, var.prior_scale_var])
            else:
                if var.fit_hyper:
                    if var_name in ['lm', 'w_ll']:
                    # Initialize with prior mean and variance
                        init_value = torch.normal(mean=var.prior_mean, std=torch.sqrt(var.prior_var)).to(self.device)
                        var.val = nn.Parameter(init_value)
                        param_hyper.extend([var.prior_mean, var.prior_var])
                    else:
                        var.randSet()
                        param_hyper.extend([var.prior_mean, var.prior_var])
            
            if var.use_heterogeneity:
                if var.fit_par:
                    param_reg.extend([var.val_min, var.val_scale])
            else:
                if var.fit_par:
                    param_reg.append(var.val)

            if var.fit_par or var.fit_hyper:
                self.track_params.append(var_name)

            if var_name in ['lm', 'w_ll']:
                setattr(self, var_name, var.val)
                
        self.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}

    def forward(self, freq_chunk_size=20, debug_sim=False):
        '''
        Function that computes the covariance matrix of the Hopf Whole Brain Model.
        '''
        # Ensure device consistency
        device = self.device 

        n = self.node_size
        n_ch = self.output_size
        m = nn.ReLU()

        # Bounding Constant
        con_1 = torch.tensor(1.0, dtype=torch.float32, device=device)
        conduct_lb = 1.5  # lower bound for conduct velocity
        noise_std_lb = 20 # lower bound of std of noise
        lb = 0.01         # lower bound of local gains

        # Extract model parameters and ensure they are on the same device
        a0 = -m(-self.params.a.value()).to(device)                  # Node's bifurcation parameter (s^-1)
        a = a0 * torch.ones(n, device=device) if a0.dim() == 0 else a0
        if self.params.omega.use_heterogeneity:
            omega = self.params.omega.value().to(device)
        else:
            mean_omega = m(self.params.omega.value()).to(device)        # Intrinsic angular frequency (rad.s^-1)
            sig_omega = (self.params.sig_omega.value()).to(device)      # Variance of the angular frequency
            omega = m(torch.normal(mean=mean_omega.item(), std=sig_omega.item(), size=(n,))).to(device)

        g = (lb * con_1 + m(self.params.g.value())).to(device)                      # Global Connectivity Scaling (s^-1)
        std_in = (noise_std_lb * con_1 + m(self.params.std_in.value())).to(device)  # White noise standard deviation
        v_d = (conduct_lb * con_1 + m(self.params.v_d.value())).to(device)          # Conduction Velocity (or its inverse i don't remember)
        cy0 = self.params.cy0.value().to(device)                                    # Leadfield Matrix Scaling Parameter

        # Update the Laplacian based on the updated connection gains w_ll.
        w_l = torch.exp(self.w_ll) * torch.tensor(self.sc, dtype=torch.float32, device=device)
        w_n_l = (0.5 * (w_l + torch.transpose(w_l, 0, 1))) / torch.linalg.norm(0.5 * (w_l + torch.transpose(w_l, 0, 1)))
        self.sc_fitted = w_n_l
        dg_l = -torch.diag(torch.sum(w_n_l, dim=1))

        C = (w_n_l + dg_l)
        S = torch.sum(C, axis=1)
        self.gamma = (torch.tensor(self.dist, dtype=torch.float32, device=device) / (v_d) 
                      if v_d.ndimension() == 0 else torch.tensor(self.dist, dtype=torch.float32, device=device) / (v_d[None, :]))

        # EEG computation (Leadfield Matrix)
        lm_t = (self.lm.T / torch.sqrt(self.lm ** 2).sum(1)).T
        self.lm_t = (lm_t - 1 / n_ch * torch.matmul(torch.ones((1,n_ch), device=device), lm_t))

        Bxx = torch.diag(a - g * S)
        Bxy = torch.diag(omega)
        B = torch.zeros(2 * n, 2 * n, dtype=torch.float32, device=device)
        B[:n, :n] = Bxx
        B[:n, n:] = Bxy
        B[n:, :n] = -Bxy
        B[n:, n:] = Bxx

        Q = (std_in**2) * torch.eye(2 * n, dtype=torch.complex64, device=device)

        def C_exp(nu):
            exp_matrix = torch.exp(1j * 2 * np.pi * nu * self.gamma)
            C_block = torch.zeros(2 * n, 2 * n, dtype=torch.complex64, device=device)
            C_block[:n, :n] = C * exp_matrix
            C_block[n:, n:] = C * exp_matrix
            return C_block

        def U_nu(nu):
            Cexp = C_exp(nu)
            U_block = B + g * Cexp + 1j * 2 * np.pi * nu * torch.eye(2 * n, dtype=torch.complex64, device=device)
            return torch.linalg.inv(U_block)

        cov_model = torch.zeros((n_ch, n_ch), dtype=torch.float32, device=device)

        for i in range(0, len(self.freqs), freq_chunk_size):
            freq_batch = self.freqs[i:i+freq_chunk_size]

            psi_nu_sum = torch.zeros((freq_chunk_size, 2 * n, 2 * n), dtype=torch.complex64, device=device)

            for j, nu in enumerate(freq_batch):
                U = U_nu(nu)
                psi_nu_sum[j] = U @ Q @ U.conj().T

            batch_covariance = 2 * torch.trapz(torch.real(psi_nu_sum), x=freq_batch, dim=0)

            batch_cov_model = cy0**2 * self.lm_t @ batch_covariance[:n, :n] @ (self.lm_t).T
            cov_model += batch_cov_model
        
        def debug_matrix(matrix, name="Matrix"):
            print(f"{name}:")
            print(f"  Shape: {matrix.shape}")
            print(f"  Mean: {matrix.mean().item():.6f}")
            print(f"  Std: {matrix.std().item():.6f}")
            print(f"  Min: {matrix.min().item():.6f}")
            print(f"  Max: {matrix.max().item():.6f}")
            print(f"  Median: {matrix.median().item():.6f}")
            print()

        if debug_sim:
            debug_matrix(cov_model, name="simCOV")
        return cov_model

class Model_fitting(AbstractFitting):
    """
    Class with the purpose of fitting the model covariance matrix, validate and test it.
    In addition it can simulate the model covariance matrix and the PSD.

    Attributes
    ----------
    model: AbstractNMM
        Whole Brain Model to Simulate
    cost: AbstractLoss
        A particular objective function which the model will be optimized for.
    trainingStats: TrainingStats
        Information about objective function loss and parameter values over training windows/epochs
    lastRec: dict
        The last simulation of fitting()
    valStats: dict
        Validation Stats
    testStats: dict
        Test Stats
    sim: dict
        The last simulation of simulate()
    device : torch.device
        Whether the fitting is to run on CPU or GPU
    """

    def __init__(self, model: AbstractNMM, cost: AbstractLoss, device='cuda'):
        """
        Parameters
        ----------
        model: AbstractNMM
            Whole Brain Model to Simulate
        cost: AbstractLoss
            A particular objective function which the model will be optimized for.
        device : torch.device
            Whether the fitting is to run on CPU or GPU
        """
        
        self.model = model
        self.cost = cost

        self.device = device

        self.trainingStats = TrainingStats(self.model)
        self.lastRec = None 
        self.valStats = {}
        self.testStats = {}
        self.sim = {}
        self.psd = None

    def save(self, filename):
        """
        Parameters
        ----------
        filename: String
            filename to use when saving object to file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train(self, empCOV: torch.Tensor, valCOV: torch.Tensor = None,
              num_epochs: int = 120, learningrate: float = 0.01, lr_2ndLevel: float = 0.05, 
              lr_scheduler: bool = False, scheduler_type: str = 'ReduceLROnPlateau', loss_method: str = 'log_fro',
              run_val: bool = False, val_freq: int = 10,
              debug_loss: bool = False, debug_grad: bool = False):
        """
        Function to train the model (with optional validation).

        Parameters
        ----------
        empCOV: torch.Tensor
            Empirical Covariance Matrix for training.
        valCOV: torch.Tensor, optional
            Empirical Covariance Matrix for validation.
        num_epochs: int
            Number of training epochs.
        learningrate: float
            Learning rate for model parameters.
        lr_2ndLevel: float
            Learning rate for hyperparameters.
        lr_scheduler: bool
            Whether to use learning rate scheduling.
        scheduler_type: str
            Type of learning rate scheduler ('ReduceLROnPlateau' or 'OneCycleLR').
        loss_method: str
            Loss method for training and validation ('mse', 'log_loss' and 'log_fro'.).
        run_validation: bool
            Whether to run validation during training.
        validation_frequency: int
            Frequency (in epochs) of validation runs.
        """
        
        # Define two different optimizers for each group
        modelparameter_optimizer = optim.Adam(self.model.params_fitted['modelparameter'], lr=learningrate, betas=(0.9, 0.999), eps=1e-8)
        if 'hyperparameter' in self.model.params_fitted and self.model.params_fitted['hyperparameter']:
            hyperparameter_optimizer = optim.Adam(self.model.params_fitted['hyperparameter'], lr=lr_2ndLevel, betas=(0.9, 0.999), eps=1e-8)
        else:
            hyperparameter_optimizer = None  # No hyperparameters
        
        print('Scheduler Type: ', scheduler_type)
        # Define the learning rate schedulers for each group of parameters
        if lr_scheduler:
            total_steps = num_epochs

            if scheduler_type == 'OneCycleLR':
                # OneCycleLR setup
                if hyperparameter_optimizer: 
                    hyperparameter_scheduler = optim.lr_scheduler.OneCycleLR(hyperparameter_optimizer,
                                                                            10 * lr_2ndLevel,
                                                                            total_steps,
                                                                            anneal_strategy="cos")
                    hlrs = []
                modelparameter_scheduler = optim.lr_scheduler.OneCycleLR(modelparameter_optimizer,
                                                                        10*learningrate,
                                                                        total_steps,
                                                                        anneal_strategy="cos")
                mlrs = []

            elif scheduler_type == 'ReduceLROnPlateau':
                # ReduceLROnPlateau setup
                if hyperparameter_optimizer: 
                    hyperparameter_scheduler = optim.lr_scheduler.ReduceLROnPlateau(hyperparameter_optimizer,
                                                                                    mode='min',
                                                                                    factor=0.5,
                                                                                    patience=5,
                                                                                    verbose=True,
                                                                                    min_lr=1e-5)
                    hlrs = []
                modelparameter_scheduler = optim.lr_scheduler.ReduceLROnPlateau(modelparameter_optimizer, 
                                                                                mode='min', 
                                                                                factor=0.5, 
                                                                                patience=5, 
                                                                                verbose=True, 
                                                                                min_lr=1e-5)
                mlrs = []
            else:
                raise ValueError("Unsupported scheduler type. Use 'OneCycleLR' or 'ReduceLROnPlateau'.")

        # Set up device and mixed precision scaler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = torch.cuda.amp.GradScaler()

        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            loss_history = []

            def closure():
                if hyperparameter_optimizer:
                    hyperparameter_optimizer.zero_grad()
                modelparameter_optimizer.zero_grad()

                # Forward pass and loss computation
                with torch.cuda.amp.autocast():
                    simCOV = self.model(freq_chunk_size=20, debug_sim=False)
                    loss = self.cost.loss(empCOV, simCOV, loss_method, debug_loss)
                loss = loss.to(device)
                scaler.scale(loss).backward()
                print(f"Loss: {loss.item():.6f}")
                loss_history.append(loss.item())
                return loss, simCOV
            
            loss, simCOV = closure()

            # Perform validation
            if run_val and (epoch + 1) % val_freq == 0:
                if valCOV is not None:
                    val_loss, val_corr, val_cos_sim = self.validate(valCOV, loss_method)

                    self.valStats[epoch + 1] = {
                        "loss": val_loss,
                        "corr": val_corr,
                        "cosine_similarity": val_cos_sim
                    }
                    print(f"Validation Loss: {val_loss:.6f}, Corr: {val_corr:.4f}, Cosine Sim: {val_cos_sim:.4f}")
                else:
                    print("Validation data not provided. Skipping validation.")

            if hyperparameter_optimizer:
                scaler.step(hyperparameter_optimizer)
            scaler.step(modelparameter_optimizer)
            scaler.update()

            if debug_grad:
                if hasattr(self.model.params.w_ll.val, 'grad') and self.model.params.w_ll.val.grad is not None:
                    wll_grads = self.model.params.w_ll.val.grad
                    print(f"Gradients for w_ll: {wll_grads}")
                    print(f"Mean Gradient: {wll_grads.mean().item()}")
                    print(f"Max Gradient: {wll_grads.max().item()}")
                    print(f"Min Gradient: {wll_grads.min().item()}")

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.params_fitted['modelparameter'], max_norm=10.0)  
            torch.nn.utils.clip_grad_norm_(self.model.params_fitted['hyperparameter'], max_norm=10.0) 

            if lr_scheduler:
                if hyperparameter_optimizer:
                    hlrs.append(hyperparameter_optimizer.param_groups[0]["lr"])
                mlrs.append(modelparameter_optimizer.param_groups[0]["lr"])

                # scheduler step
                if scheduler_type == 'OneCycleLR':
                    if hyperparameter_optimizer:
                        hyperparameter_scheduler.step()
                    modelparameter_scheduler.step()
                elif scheduler_type == 'ReduceLROnPlateau':
                    if hyperparameter_optimizer:
                        hyperparameter_scheduler.step(loss.item())
                    modelparameter_scheduler.step(loss.item())

            # Calculate Pseudo FC Correlation and Cosine Similarity
            emp_flat = empCOV.detach().cpu().numpy().ravel()
            sim_flat = simCOV.detach().cpu().numpy().ravel()

            pseudo_fc_cor = np.corrcoef(emp_flat, sim_flat)[0, 1]
            cos_sim = cosine_similarity(emp_flat.reshape(1, -1), sim_flat.reshape(1, -1))[0, 0]

            # Log metrics
            print(f"Pseudo FC Corr: {pseudo_fc_cor:.4f}")
            print(f"cos_sim: {cos_sim:.4f}")

            if lr_scheduler:
                for param_group in modelparameter_optimizer.param_groups:
                    print('Modelparam_lr:', param_group['lr'])
                if hyperparameter_optimizer:
                    for param_group in hyperparameter_optimizer.param_groups:
                        print('Hyperparam_lr:', param_group['lr'])

            self.trainingStats.appendLoss(np.mean(loss_history))

            # Parameter info for the Epoch
            trackedParam = {}
            exclude_param = ['w_ll', 'gains_con', 'lm'] #This stores SC and LF which are saved seperately
            if(self.model.track_params):
                for par_name in self.model.track_params:
                    var = getattr(self.model.params, par_name)
                    if (var.fit_par):
                        trackedParam[par_name] = var.value().detach().cpu().numpy().copy()
                    if (var.fit_hyper):
                        if var.use_heterogeneity:
                            trackedParam[par_name + "_prior_min_mean"] = var.prior_min_mean.detach().cpu().numpy().copy()
                            trackedParam[par_name + "_prior_min_var"] = var.prior_min_var.detach().cpu().numpy().copy()
                            trackedParam[par_name + "_prior_scale_mean"] = var.prior_scale_mean.detach().cpu().numpy().copy()
                            trackedParam[par_name + "_prior_scale_var"] = var.prior_scale_var.detach().cpu().numpy().copy()
                        else:
                            trackedParam[par_name + "_prior_mean"] = var.prior_mean.detach().cpu().numpy().copy()
                            trackedParam[par_name + "_prior_var"] = var.prior_var.detach().cpu().numpy().copy()
            for key, value in self.model.state_dict().items():
                if key not in exclude_param:
                    trackedParam[key] = value.detach().cpu().numpy().ravel().copy()
            self.trainingStats.appendParam(trackedParam)
            # Saving the SC and/or Lead Field State at Every Epoch
            if self.model.use_fit_gains:
                self.trainingStats.appendWll(self.model.w_ll.detach().cpu().numpy())
                self.trainingStats.appendCONN(self.model.sc_fitted.detach().cpu().numpy())
            if self.model.use_fit_lfm:
                self.trainingStats.appendLF(self.model.lm.detach().cpu().numpy())
            
        # Save the last optimized covariance matrix
        self.lastRec = {}
        self.lastRec["simCOV"] = simCOV.detach().cpu().numpy()

    def validate(self, valCOV: torch.Tensor, loss_method: str = 'log_fro'):
        """
        Evaluate the model on the validation set.
        
        Parameters
        ----------
        valCOV: torch.Tensor
            Validation covariance matrix (empirical).
        loss_method: str
            Loss method to use during validation.
        
        Returns
        -------
        avg_loss: float
            Average validation loss.
        avg_corr: float
            Average correlation between validation and simulated covariance matrices.
        avg_cos_sim: float
            Average cosine similarity between validation and simulated covariance matrices.
        """
        self.model.eval()  # Put model in evaluation mode
        device = self.device

        with torch.no_grad():
            # Forward pass and loss computation
            simCOV = self.model(freq_chunk_size=20, debug_sim=False)
            loss = self.cost.loss(valCOV, simCOV, loss_method).to(device)

            val_flat = valCOV.detach().cpu().numpy().ravel()
            sim_flat = simCOV.detach().cpu().numpy().ravel()

            avg_corr = np.corrcoef(val_flat, sim_flat)[0, 1]
            avg_cos_sim = cosine_similarity(val_flat.reshape(1, -1), sim_flat.reshape(1, -1))[0, 0]

        return loss.item(), avg_corr, avg_cos_sim

    def test(self, testCOV: torch.Tensor, loss_method: str = 'log_fro'):
        """
        Evaluate the model on the test set and save results to an attribute.
        
        Parameters
        ----------
        testCOV: torch.Tensor
            Test covariance matrix (empirical).
        loss_method: str
            Loss method to use during testing.
        
        Returns
        -------
        test_loss: float
            Test loss value.
        corr: float
            Correlation between test and simulated covariance matrices.
        cos_sim: float
            Cosine similarity between test and simulated covariance matrices.
        """
        self.model.eval()  # Set model to evaluation mode
        device = self.device

        with torch.no_grad():
            # Forward pass and loss computation
            simCOV = self.model(freq_chunk_size=20, debug_sim=False)
            test_loss = self.cost.loss(testCOV, simCOV, loss_method).to(device)

            # Flatten the covariance matrices for comparison
            test_flat = testCOV.detach().cpu().numpy().ravel()
            sim_flat = simCOV.detach().cpu().numpy().ravel()

            # Compute correlation
            corr = np.corrcoef(test_flat, sim_flat)[0, 1]

            # Compute cosine similarity
            cos_sim = np.dot(test_flat, sim_flat) / (
                np.linalg.norm(test_flat) * np.linalg.norm(sim_flat)
            )

        # Save results to the attribute
        self.testStats = {
            "test_loss": test_loss.item(),
            "correlation": corr,
            "cosine_similarity": cos_sim
        }

        print(f"Test Loss: {test_loss.item():.6f}, Correlation: {corr:.4f}, Cosine Sim: {cos_sim:.4f}")

    def simulate(self, freq_chunk_size=20, debug_sim=False):
        """
        Simulate data using the model without optimization.

        Parameters
        ----------
        freq_chunk_size : int, optional
            Size of frequency chunks to use during simulation (default is 20).
        debug_sim : bool, optional
            Whether to enable debug mode for the simulation (default is False).

        """
        self.model.eval()  # Set the model to evaluation mode
        device = self.device

        with torch.no_grad():
            simCOV = self.model(freq_chunk_size=freq_chunk_size, debug_sim=debug_sim)

        self.sim = {
            "COV": simCOV.detach().cpu().numpy()
        }

        print("Simulation completed. Covariance matrix saved in '.sim['COV']'.")

    def PSD(self):
        '''
        Function that computes the Power Spectral Density of the Hopf Whole Brain Model.
        '''
        print('Computing PSD')
        # Ensure device consistency
        device = self.device 

        n = self.model.node_size
        n_ch = self.model.output_size
        m = nn.ReLU()

        # Bounding Constant
        con_1 = torch.tensor(1.0, dtype=torch.float32, device=device)
        conduct_lb = 1.5  # lower bound for conduct velocity
        noise_std_lb = 20 # lower bound of std of noise
        lb = 0.01         # lower bound of local gains

        a0 = -m(-self.model.params.a.value()).to(device)                  # Node's bifurcation parameter (s^-1)
        a = a0 * torch.ones(n, device=device) if a0.dim() == 0 else a0
        if self.model.params.omega.use_heterogeneity:
            omega = self.model.params.omega.value().to(device)
        else:
            mean_omega = m(self.model.params.omega.value()).to(device)        # Intrinsic angular frequency (rad.s^-1)
            sig_omega = (self.model.params.sig_omega.value()).to(device)      # Variance of the angular frequency
            omega = m(torch.normal(mean=mean_omega.item(), std=sig_omega.item(), size=(n,))).to(device)

        g = (lb * con_1 + m(self.model.params.g.value())).to(device)                      # Global Connectivity Scaling (s^-1)
        std_in = (noise_std_lb * con_1 + m(self.model.params.std_in.value())).to(device)  # White noise standard deviation
        v_d = (conduct_lb * con_1 + m(self.model.params.v_d.value())).to(device)          # Conduction Velocity (or its inverse i don't remember)
        cy0 = self.model.params.cy0.value().to(device)                                    # Leadfield Matrix Scaling Parameter

        # Update the Laplacian based on the updated connection gains w_ll.
        w_l = torch.exp(self.model.w_ll) * torch.tensor(self.model.sc, dtype=torch.float32, device=device)
        w_n_l = (0.5 * (w_l + torch.transpose(w_l, 0, 1))) / torch.linalg.norm(0.5 * (w_l + torch.transpose(w_l, 0, 1)))
        dg_l = -torch.diag(torch.sum(w_n_l, dim=1))

        C = (w_n_l + dg_l)
        S = torch.sum(C, axis=1)
        gamma = (torch.tensor(self.dist, dtype=torch.float32, device=device) / (v_d) 
                      if v_d.ndimension() == 0 else torch.tensor(self.dist, dtype=torch.float32, device=device) / (v_d[None, :]))

        # EEG computation (Leadfield Matrix)
        lm_t = (self.model.lm.T / torch.sqrt(self.model.lm ** 2).sum(1)).T
        lm_t = (lm_t - 1 / n_ch * torch.matmul(torch.ones((1,n_ch), device=device), lm_t))

        Bxx = torch.diag(a - g * S)
        Bxy = torch.diag(omega)
        B = torch.zeros(2 * n, 2 * n, dtype=torch.float32, device=device)
        B[:n, :n] = Bxx
        B[:n, n:] = Bxy
        B[n:, :n] = -Bxy
        B[n:, n:] = Bxx

        Q = (std_in**2) * torch.eye(2 * n, dtype=torch.complex64, device=device)

        def C_exp(nu):
            exp_matrix = torch.exp(1j * 2 * np.pi * nu * gamma)
            C_block = torch.zeros(2 * n, 2 * n, dtype=torch.complex64, device=device)
            C_block[:n, :n] = C * exp_matrix
            C_block[n:, n:] = C * exp_matrix
            return C_block

        def U_nu(nu):
            Cexp = C_exp(nu)
            U_block = B + g * Cexp + 1j * 2 * np.pi * nu * torch.eye(2 * n, dtype=torch.complex64, device=device)
            return torch.linalg.inv(U_block)
        
        # Compute the cross-spectrum over the range
        Psi_nu = []
        # Iterate over the frequencies.
        for nu in self.model.freqs:
            # Compute U_nu(nu)
            U = U_nu(nu)
            
            # Compute the cross-spectrum for the current frequency.
            cross_spectrum = U @ Q @ U.conj().T
            
            # Append the result to the list.
            Psi_nu.append(cross_spectrum)

        # Convert the list of results to a PyTorch tensor.
        Psi_nu = torch.stack(Psi_nu)
        
        # Compute the power spectrum in the space of channels
        psd = torch.zeros((n_ch, len(self.model.freqs)), dtype=torch.float32, device=device)
            
        for i, nu in enumerate(self.model.freqs):
            Z_xx = Psi_nu[i][:n, :n].real  # Extract the block corresponding to the real part x
            Y = cy0**2 * (lm_t) @ Z_xx @ (lm_t).T  # Transform to channel space
            psd[:, i] = Y.diagonal()  # Store the diagonal (PSD values)

        self.psd = {}
        self.psd['simPSD'] = psd.detach().cpu().numpy()
            
class CostsHP(AbstractLoss):
    def __init__(self, model):
        self.mainLoss = CostsTS()
        self.model = model
    
    def loss(self, empData: torch.Tensor, simData: torch.Tensor, loss_method: str = 'mse' ,debug_loss: bool = False):
        eps = 1e-8  # avoid dividing by very small values in the Normalization procedure
        sim = simData / (simData.norm(p='fro') + eps)
        emp = empData / (empData.norm(p='fro') + eps)

        model = self.model

        # define some constants
        lb = 0.001
        w_cost = 1.0
        reg_lambda = 0.01
        reg_scales = {
            'w_ll': 1e-3,  
            'lm': 1e-2,     
        }

        # define the relu function
        m = torch.nn.ReLU()

        exclude_param = []
        if not model.use_fit_gains:
            exclude_param.append('w_ll')
            exclude_param.append('gains_con')

        if not model.use_fit_lfm:
            exclude_param.append('lm')
        
        loss_main = self.mainLoss.loss(emp, sim, method=loss_method)
        # Print main loss contribution
        if debug_loss:
            print(f"Loss Main ({loss_method}): {loss_main.item()}")
            print("Regularization contributions:")

        loss_prior = []
        reg_term = 0

        variables_p = [a for a in dir(model.params) if (type(getattr(model.params, a)) == par)]
        for var_name in variables_p:
            var = getattr(model.params, var_name)

            if var_name == 'sig_omega' and model.params.omega.use_heterogeneity:
                continue

            if var_name in exclude_param:
                continue

            if var.use_heterogeneity:
                if var.has_prior:
                    prior_loss_min = torch.sum(reg_scales.get(var_name, 1.0) * (lb + m(var.prior_min_var)) * 
                                   (m(var.val_min) - m(var.prior_min_mean)) ** 2) \
                                   + torch.sum(-torch.log(lb + m(var.prior_min_var)))
                    
                    # Prior loss for val_scale
                    prior_loss_scale = torch.sum(reg_scales.get(var_name, 1.0) * (lb + m(var.prior_scale_var)) * 
                                      (m(var.val_scale) - m(var.prior_scale_mean)) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_scale_var)))
                    
                    # Combine prior losses
                    prior_loss = prior_loss_min + prior_loss_scale
                    loss_prior.append(prior_loss)
                    
                    # Print prior loss contribution for debugging
                    if debug_loss:
                        print(f"Prior loss for {var_name} (heterogeneity): {prior_loss.item()}")
                else:
                    # Regularization for val_min and val_scale (no prior)
                    reg_loss = reg_lambda * (torch.sum(var.val_min ** 2) + torch.sum(var.val_scale ** 2))
                    reg_term += reg_loss
                    
                    # Print regularization contribution
                    if debug_loss:
                        print(f"  {var_name} (heterogeneity): {reg_loss.item()}")
            else:
                # Handle non-heterogeneity parameters
                if var.has_prior:
                    # Prior loss for non-heterogeneity parameters
                    prior_loss = torch.sum(reg_scales.get(var_name, 1.0) * (lb + m(var.prior_var)) * 
                                 (m(var.val) - m(var.prior_mean)) ** 2) \
                                 + torch.sum(-torch.log(lb + m(var.prior_var)))
                    loss_prior.append(prior_loss)
                    
                    # Print prior loss contribution for debugging
                    if debug_loss:
                        print(f"Prior loss for {var_name}: {prior_loss.item()}")
                else:
                    # Regularization for non-heterogeneity parameters (no prior)
                    reg_loss = reg_lambda * torch.sum(var.val ** 2)
                    reg_term += reg_loss
                    
                    # Print regularization contribution
                    if debug_loss:
                        print(f"  {var_name}: {reg_loss.item()}")

        # Summing up all contributions
        loss_prior_sum = sum(loss_prior)
        total_loss = w_cost * loss_main + loss_prior_sum + reg_term

        if debug_loss:
            print(f"Total Loss: {total_loss.item()}")
        
        return total_loss
    
#######################################################################################################################################################################################################
# Running Code Start HERE
#######################################################################################################################################################################################################

def main(n_sub: int = 1, train_region: str = 'Premotor', 
         val_region: str = None, test_region: str = None, valFlag: bool = False):
    # Choose Subject
    subject_num = n_sub

    repo_root = Path(__file__).resolve().parent

    # Define the subfolders within your repo
    struct_data_folder = repo_root / 'struct_data'
    conn_data_folder = repo_root / 'conn_data'
    eegtms_data_folder = repo_root / 'eegtms_data'
    neuromaps_folder = repo_root / 'neuromaps'

    # Define the path to save the file
    eeg_tms_file = eegtms_data_folder / 'Experiment_1.mat'

    # If the file does not exist, download it from Google Drive
    if not os.path.exists(eeg_tms_file):
        print("EEG-TMS data not found. Downloading...")
        # Google Drive file ID
        file_id = '18fvsLjG1nmg43grW3dRpiXwEsRbYqRvS'
        gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', str(eeg_tms_file), quiet=False)
        print(f"Data downloaded to {eeg_tms_file}")
    else:
        print("EEG-TMS data already exists.")

    # Load data and layout information from .mat file
    mat_data = scipy.io.loadmat(eeg_tms_file)
    layout_mat_data = scipy.io.loadmat(eegtms_data_folder / 'chlocs_nexstim.mat')

    # Load NeuroMaps
    t1t2_map = np.load(neuromaps_folder / 'T1T2_parc.npy')
    T = erf(t1t2_map)
    T_max, T_min = T.max(), T.min()
    h_map = (T_max - T) / (T_max - T_min)

    n_channels = 60
    sampling_freq = 725  # in Hertz

    # Channel labels
    channel_labels = ['Fp1', 'Fpz', 'Fp2', 'AF1', 'AFz', 'AF2', 'F7', 'F5', 'F1', 'Fz', 'F2', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T3', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T4', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P8', 'P10', 'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2', 'Iz']
    ch_types = ["eeg"] * n_channels

    X, Y, Z = [np.ndarray(n_channels) for _ in range(3)]

    for i in range(n_channels):
        X[i] = layout_mat_data['chlocs']['Y'][0][i][0]/10
        Y[i] = layout_mat_data['chlocs']['X'][0][i][0]/10
        Z[i] = layout_mat_data['chlocs']['Z'][0][i][0]/10

    # Create MNE data structure
    ch_loc = np.concatenate((X[:, np.newaxis], Y[:, np.newaxis], Z[:, np.newaxis]), axis=1)
    ch_loc_dict = {}
    for label, coordinates in zip(channel_labels, ch_loc):
        ch_loc_dict[label] = coordinates

    custom_montage = mne.channels.make_dig_montage(ch_pos=ch_loc_dict ,coord_frame='head')
    info = mne.create_info(ch_names=channel_labels, ch_types= ch_types, sfreq=sampling_freq)
    info.set_montage(custom_montage)

    def prep_data(region_key, split_flag=False, split_indices=None):
        """
        Helper function to compute covariance matrices for a given region.
        """
        data_dict = {}
        region_data = mat_data['Experiment_1'][0, 0][f'Subject_{subject_num}'][0, 0][region_key]
        eeg_data_epochs = np.transpose(region_data, (2, 0, 1))
        epoched = mne.EpochsArray(eeg_data_epochs, info)
        evoked = epoched.average()

        fft_results = np.fft.fft(evoked.data, axis=1)
        freq_axis = np.fft.fftfreq(evoked.data.shape[1], 1 / sampling_freq)

        peak_frequencies = []
        for channel_fft in fft_results:
            peak_index = np.argmax(np.abs(channel_fft)[:len(channel_fft) // 2])
            peak_frequency = freq_axis[peak_index]
            peak_frequencies.append(peak_frequency)

        peaks = np.array(peak_frequencies)
        omega0 = np.mean(peaks)
        sig_omega0 = np.std(peaks)

        trials_array = epoched.get_data()
        num_trials, num_channels, _ = trials_array.shape
        t_start, t_end = 0.0, 0.5
        index_start, index_end = int(t_start * sampling_freq), int(t_end * sampling_freq)

        if split_flag and split_indices is not None:
            # Use provided indices for the subset
            trials_array = trials_array[split_indices]

        # Compute covariance matrices for the trials
        rho = np.zeros((trials_array.shape[0], num_channels, num_channels))
        for trial in range(trials_array.shape[0]):
            rho[trial] = np.cov(trials_array[trial, :, index_start:index_end])

        data_dict = {
            'cov': torch.tensor(np.mean(rho, axis=0), dtype=torch.float32, device='cuda'),
            'omega': omega0,
            'sig_omega': sig_omega0,
            'nTrials': num_trials
        }
        return data_dict

    print(f'Processing Subject {subject_num}, Region {train_region}')

    # Compute correlation on the resting state
    trainData = prep_data(train_region)
    trainCOV = trainData['cov']
    omega0 = trainData['omega']
    sig_omega0 = trainData['sig_omega']
    num_trials = trainData['nTrials']

    if valFlag:
        if val_region and test_region:
            print(f'Validation Region: {val_region}')
            valCOV = prep_data(val_region)['cov']

            print(f'Test Region: {test_region}')
            testCOV = prep_data(test_region)['cov']
        else:
            print("No specific validation or test region provided. Falling back to 60-30-10 split in the train region.")
            # Shuffle trials for randomness
            np.random.seed(42)
            shuffled_indices = np.random.permutation(num_trials)
            train_split = int(0.6 * num_trials)
            val_split = int(0.9 * num_trials)

            train_indices = shuffled_indices[:train_split]
            val_indices = shuffled_indices[train_split:val_split]
            test_indices = shuffled_indices[val_split:]

            # Prepare split datasets
            trainData = prep_data(train_region, split_flag=True, split_indices=train_indices)
            trainCOV = trainData['cov']

            valData = prep_data(train_region, split_flag=True, split_indices=val_indices)
            valCOV = valData['cov']

            testData = prep_data(train_region, split_flag=True, split_indices=test_indices)
            testCOV = testData['cov']
    else:
        empCOV = trainCOV

    # Load connectivity matrix
    atlas = pd.read_csv(conn_data_folder / 'atlas_data.csv')
    labels = atlas['ROI Name']
    coords = np.array([atlas['R'], atlas['A'], atlas['S']]).T

    dist = np.zeros((coords.shape[0], coords.shape[0]))
    for roi1 in range(coords.shape[0]):
        for roi2 in range(coords.shape[0]):
            dist[roi1, roi2] = np.sqrt(np.sum((coords[roi1, :] - coords[roi2, :]) ** 2, axis=0))

    sc_file = conn_data_folder / 'Schaefer2018_200Parcels_7Networks_count.csv'
    sc_df = pd.read_csv(sc_file, header=None, sep=' ')
    sc = sc_df.values
    sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

    fc_path = conn_data_folder / 'group_mean_fc.npy'
    group_fc = np.load(fc_path)
    fc_shifted = group_fc
    log_fc = np.log1p(fc_shifted)
    fc = log_fc / np.linalg.norm(log_fc)

    node_size = sc.shape[0]
    output_size = n_channels

    lm_name = struct_data_folder / f'Subject{subject_num}_{train_region}_leadfield.npy'
    lm0 = np.load(lm_name)

    wll_name = struct_data_folder / f'Subject{subject_num}_{train_region}_wll.npy'
    wll0 = np.load(wll_name)

    # Upload Starting w_ll
    #wll_regions = ['Prefrontal', 'Premotor', 'Motor']
    #wll_sum = None
    #wll_count = 0
    #for r in wll_regions:
    #    wll_name = os.path.join(hopf_path, 'struct_data', f'Subject{subject_num}_{r}_wll.npy')
    #    wll_region = np.load(wll_name)
        
    #    if wll_sum is None:
    #        wll_sum = np.zeros_like(wll_region)
        
    #    wll_sum += wll_region
    #    wll_count += 1
    
    #wll0 = wll_sum / wll_count

    # w_ll mediated through regions and subjects
    #wll_regions = ['Prefrontal', 'Premotor', 'Motor']

    #wll_sum = None
    #wll_count = 0

    #for s in range(1, 7): 
    #    for r in wll_regions:
    #        wll_name = os.path.join(hopf_path, 'struct_data', f'Subject{s}_{r}_wll.npy')
    #        wll_region = np.load(wll_name)

    #        if wll_sum is None:
    #            wll_sum = np.zeros_like(wll_region)
    #        
    #        wll_sum += wll_region
    #        wll_count += 1

    #wll0 = wll_sum / wll_count

    # Initialize w_ll prior
    #wll0 = np.zeros_like(sc)
    #eps = 1e-6
    #alpha = 0.5
    #blended = alpha * fc + (1 - alpha) * sc
    #wll0 = np.log(blended + eps)
    #wll0 = fc.copy()

    freqs = np.linspace(1, 80, 800, endpoint=False)
    params = ParamsHP(a=par(-0.5,(-0.5, 0.05), (1/4, 0.05), True, True, use_heterogeneity=True, h_map=h_map), 
                    omega=par(omega0, (omega0, 0.05), (1/2, 0.05), True, True, use_heterogeneity=True, h_map=h_map),
                    sig_omega=par(sig_omega0, sig_omega0, 1/4, True, True),
                    g=par(500,500, 10, True, True), std_in= par(0.3, 0.3, 1, True, True),
                    v_d = par(1., (1., 0.05), (0.4, 0.05), True, True, use_heterogeneity=True, h_map=h_map), 
                    cy0 = par(50, 50, 1, True, True),
                    lm=par(lm0), w_ll=par(wll0, wll0, np.ones_like(wll0), True, True))

    # Simulation start
    n_epochs = 120
    start_time = time.time()

    model = COVHOPF(node_size, output_size, sampling_freq, sc, dist, freqs, params, lm0, wll0, use_fit_gains=True, use_fit_lfm=False)
    ObjFun = CostsHP(model)
    F = Model_fitting(model, ObjFun)
    scheds = {'O': 'OneCycleLR', 'R': 'ReduceLROnPlateau'}
    sched_type = scheds['R']
    loss_method = 'log_fro'
    print(f"Loss Method: {loss_method}")

    if not valFlag:
        # Train
        F.train(empCOV = empCOV, num_epochs = n_epochs, 
                lr_scheduler = True, scheduler_type = sched_type, loss_method = loss_method, debug_loss = False)
    else:
        # Train with validation
        F.train(empCOV = trainCOV, valCOV = valCOV, num_epochs = n_epochs, lr_scheduler = True,
                scheduler_type = sched_type, loss_method = loss_method, 
                run_val = True, val_freq = 10, debug_loss = False)
        F.test(testCOV = testCOV, loss_method = loss_method)

    # Simulate the model after training
    F.simulate()
    # Compute the PSD with the optimized parameters
    F.PSD()

    # Save fitting results to a file using pickle
    store_filename = repo_root / 'Results' / f'Subject{subject_num}_{train_region}_{loss_method}_{sched_type}_neuromaps_fitting_results.pkl'
    with open(store_filename, 'wb') as file:
        pickle.dump(F, file)
    print("Results successfully saved to the file.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert elapsed time to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"Elapsed time: {minutes} minutes and {seconds} seconds")
    print(f'Finished processing Subject {subject_num}, Region {train_region}')

if __name__ == "__main__":
    #main(valFlag=True, train_region='Premotor', val_region='Prefrontal', test_region='Motor')
    main(n_sub = 1, train_region = 'Premotor')