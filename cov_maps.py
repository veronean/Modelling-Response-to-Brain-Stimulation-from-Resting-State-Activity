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
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import erf
from sklearn.metrics import r2_score
from pathlib import Path


warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
torch.cuda.reset_accumulated_memory_stats()

import sys

#neuroimaging packages
import mne

def Scaler(pred, target, eps=1e-8):
    """
    Apply rescaling of the predicted values to match the norm of the target values.
    This is useful when the model output is not scaled correctly.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        norm_target = target.norm(p='fro')
        rescaled_pred = pred * (norm_target / (pred.norm(p='fro') + eps))
    elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        norm_target = np.linalg.norm(target, 'fro')
        rescaled_pred = pred * (norm_target / (np.linalg.norm(pred, 'fro') + eps))
    else:
        raise TypeError("Both input arguments must be either PyTorch tensors or NumPy arrays, but not mixed.")
    
    return rescaled_pred

# Suppression block
class Suppressor:
  def __enter__(self):
    self.stdout_original = sys.stdout
    sys.stdout = open('/dev/null', 'w')


  def __exit__(self, exc_type, exc_value, traceback):
    sys.stdout.close()
    sys.stdout = self.stdout_original

class par:
    def __init__(self, val, prior_mean=None, prior_var=None, 
                 fit_par=False, fit_hyper=False, use_heterogeneity=False, 
                 h_maps=None, param_bounds=(0.0, 1.0),
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        '''
        Parameters
        ----------
        val : Float or Tuple
            If use heterogeneity is False: the parameter value
            If use heterogeneity is True: (val_min, val_max)
        prior_mean : Float or Tuple
            If use_heterogeneity is False: prior_mean for the parameter.
            If use_heterogeneity is True: (prior_min_mean, prior_max_mean).
        prior_var : Float or Tuple
            If use_heterogeneity is False: prior_var for the parameter.
            If use_heterogeneity is True: (prior_min_var, prior_max_var).
        fit_par : Bool
            Whether the parameter value should be set as a PyTorch Parameter
        fit_hyper : Bool
            Whether the prior mean and prior variance should be set as a PyTorch Parameter
        use_heterogeneity : Bool
            Whether this parameter should be computed using a heterogeneity map.
        h_map : List[Tensor] or Tensor (Optional)
            The heterogeneity maps to be used (should match spatial dimensions of the parameter).
            Can be a single tensor (for backward compatibility) or list of tensors for new version.
        param_bounds : Tuple (min, max)
            The minimum and maximum bounds for the parameter values when using heterogeneity.
            Default is (0.0, 1.0).
        '''
        self.device = device
        self.use_heterogeneity = use_heterogeneity
        self.fit_par = fit_par
        self.fit_hyper = fit_hyper
        self.param_bounds = param_bounds
        
        # Process heterogeneity maps
        if use_heterogeneity and h_maps is not None:
            if isinstance(h_maps, (list, tuple)):
                self.h_maps = [torch.tensor(m, dtype=torch.float32, device=device) for m in h_maps]
                self.num_maps = len(h_maps)
            else:
                self.h_maps = [torch.tensor(h_maps, dtype=torch.float32, device=device)]
                self.num_maps = 1
        else:
            self.h_maps = None
            self.num_maps = 0

        if use_heterogeneity:
            if np.all(prior_mean != None) and np.all(prior_var != None):
                self.has_prior = True
            else:
                self.has_prior = False
                prior_mean = (0, 0)
                prior_var = (0, 0)
                       
            self.val_min = torch.tensor(val[0], dtype=torch.float32, device=device)
            self.val_max = torch.tensor(val[1], dtype=torch.float32, device=device)
            
            self.map_weights = torch.ones(self.num_maps, dtype=torch.float32, device=device) / self.num_maps
            
            self.prior_min_mean = torch.tensor(prior_mean[0], dtype=torch.float32, device=device)
            self.prior_max_mean = torch.tensor(prior_mean[1], dtype=torch.float32, device=device)
            self.prior_min_var = torch.tensor(prior_var[0], dtype=torch.float32, device=device)
            self.prior_max_var = torch.tensor(prior_var[1], dtype=torch.float32, device=device)
            
            if fit_par:
                self.val_min = nn.Parameter(self.val_min)
                self.val_max = nn.Parameter(self.val_max)
                self.map_weights = nn.Parameter(self.map_weights)
                
            if fit_hyper:
                self.prior_min_mean = nn.Parameter(self.prior_min_mean)
                self.prior_min_var = nn.Parameter(self.prior_min_var)
                self.prior_max_mean = nn.Parameter(self.prior_max_mean)
                self.prior_max_var = nn.Parameter(self.prior_max_var)
            
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

            if fit_par:
                self.val = nn.Parameter(self.val)
            if fit_hyper:
                self.prior_mean = nn.Parameter(self.prior_mean)
                self.prior_var = nn.Parameter(self.prior_var)

    def value(self):
        if self.use_heterogeneity and self.h_maps is not None:
            normalized_weights = torch.tanh(self.map_weights)
            
            if len(self.h_maps) == 1:
                combined_map = self.h_maps[0]
            else:
                combined_map = sum(w * m for w, m in zip(normalized_weights, self.h_maps))
            
            # Apply constraints to val_min and val_max using the user-specified bounds
            param_min, param_max = self.param_bounds
            val_min = torch.sigmoid(self.val_min) * (param_max - param_min) + param_min
            val_max = torch.sigmoid(self.val_max) * (param_max - param_min) + param_min
            
            # Ensure val_max > val_min
            val_max = torch.max(val_min + 1e-6, val_max)
            
            if len(self.h_maps) == 1:
                return val_min + normalized_weights[0] * combined_map * (val_max - val_min)
            else:
                v_min = combined_map.min()
                v_max = combined_map.max()
                scale = (val_max - val_min) / (v_max - v_min + 1e-6)
                offset = val_min - scale * v_min
                return scale * combined_map + offset
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
        
    def to(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        if self.use_heterogeneity:
            if self.fit_par:
                self.val_min = nn.Parameter(self.val_min.detach().clone().to(device))
                self.val_max = nn.Parameter(self.val_max.detach().clone().to(device))
                self.map_weights = nn.Parameter(self.map_weights.detach().clone().to(device))
            else:
                self.val_min = self.val_min.to(device)
                self.val_max = self.val_max.to(device)
                self.map_weights = self.map_weights.to(device)

            if self.fit_hyper:
                self.prior_min_mean = nn.Parameter(self.prior_min_mean.detach().clone().to(device))
                self.prior_min_var = nn.Parameter(self.prior_min_var.detach().clone().to(device))
                self.prior_max_mean = nn.Parameter(self.prior_max_mean.detach().clone().to(device))
                self.prior_max_var = nn.Parameter(self.prior_max_var.detach().clone().to(device))
            else:
                self.prior_min_mean = self.prior_min_mean.to(device)
                self.prior_min_var = self.prior_min_var.to(device)
                self.prior_max_mean = self.prior_max_mean.to(device)
                self.prior_max_var = self.prior_max_var.to(device)
                
            # Move maps to device
            self.h_maps = [m.to(device) for m in self.h_maps]
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
                self.val_max = self.prior_max_mean.detach() + self.prior_max_var.detach() * torch.randn(1, device=self.device)
                self.map_weights = torch.randn(self.num_maps, device=self.device) * 0.1  # Small random weights
                
                if self.fit_par:
                    self.val_min = nn.Parameter(self.val_min)
                    self.val_max = nn.Parameter(self.val_max)
                    self.map_weights = nn.Parameter(self.map_weights)
            else:
                raise ValueError("must have priors provided at par object initialization to use this method")
        else:
            if self.has_prior:
                self.val = self.prior_mean.detach() + self.prior_var.detach() * torch.randn(1, device=self.device)
                if self.fit_par:
                    self.val = torch.nn.parameter.Parameter(self.val)
            else:
                raise ValueError("must have priors provided at par object initialization to use this method")

    # Keep all the operator overloads the same
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
    wll : List
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
        self.model = model
        self.model_info = model.info()
        self.track_params = model.track_params

        self.loss = []

        self.connectivity = []
        self.wll = []
        self.leadfield = []

        self.fit_params = {}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def reset(self):
        self.loss = []

        self.connectivity = []
        self.wll = []
        self.leadfield = []
        
        self.fit_params = {}

    def appendLoss(self, newValue):
        self.loss.append(newValue)

    def appendCONN(self, newValue):
        self.connectivity.append(newValue)

    def appendWll(self, newValue):
        self.wll.append(newValue)

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
                if name == 'sig_omega' and self.model.params.omega.use_heterogeneity:
                    continue
                if isinstance(value, torch.Tensor):
                    self.fit_params[name] = [value.detach().clone().cpu().numpy()]
                else:
                    self.fit_params[name] = [value]
        else:
            for name, value in newValues.items():
                # Skip sig_omega if omega is heterogeneous
                if name == 'sig_omega' and self.model.params.omega.use_heterogeneity:
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
    def __init__(self, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def loss(self, simData, empData, method):
        pass

class AbstractFitting():
    def __init__(self, model: AbstractNMM, cost: AbstractLoss, 
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):
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
    wll_init: ndarray of floats
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
                params: ParamsHP, lm: np.ndarray, wll_init: np.ndarray = None,
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        if wll_init is None:
            wll_init = np.zeros((self.node_size, self.node_size)) + 0.05
            print("wll_init not provided, using default initialization for wll.")
        
        self.wll = wll_init

        self.setModelParameters()

    def setModelParameters(self):
        """
        Sets the parameters of the model.
        """

        param_reg = []
        param_hyper = []

        # Set wll as attributes as type Parameter if use_fit_gains is True
        if self.use_fit_gains:
            self.wll = nn.Parameter(torch.tensor(self.wll, dtype=torch.float32, device=self.device))
            param_reg.append(self.wll)
            print("Fitting gains: wll is set as a Parameter.")
        else:
            self.wll = torch.tensor(self.wll, dtype=torch.float32, device=self.device)
            print("NOT fitting gains: wll is a fixed tensor.")

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
                    param_hyper.extend([var.prior_min_mean, var.prior_min_var, var.prior_max_mean, var.prior_max_var])
            else:
                if var.fit_hyper:
                    if var_name in ['lm', 'wll']:
                    # Initialize with prior mean and variance
                        init_value = torch.normal(mean=var.prior_mean, std=torch.sqrt(var.prior_var)).to(self.device)
                        var.val = nn.Parameter(init_value)
                        param_hyper.extend([var.prior_mean, var.prior_var])
                    else:
                        var.randSet()
                        param_hyper.extend([var.prior_mean, var.prior_var])
            
            if var.use_heterogeneity:
                if var.fit_par:
                    param_reg.extend([var.val_min, var.val_max, var.map_weights])
            else:
                if var.fit_par:
                    param_reg.append(var.val)

            if var.fit_par or var.fit_hyper:
                self.track_params.append(var_name)

            if var_name in ['lm', 'wll']:
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
            mean_omega = m(self.params.omega.value()).to(device)  # Intrinsic angular frequency (rad.s^-1)
            raw_sig_omega = (self.params.sig_omega.value()).to(device)  # Raw parameter
            sig_omega = torch.nn.functional.softplus(raw_sig_omega)  # Always positive
            omega = m(torch.normal(mean=mean_omega.item(), std=sig_omega.item(), size=(n,))).to(device)

        g = (lb * con_1 + m(self.params.g.value())).to(device)                      # Global Connectivity Scaling (s^-1)
        std_in = (noise_std_lb * con_1 + m(self.params.std_in.value())).to(device)  # White noise standard deviation
        v_d = (conduct_lb * con_1 + m(self.params.v_d.value())).to(device)          # Conduction Velocity (or its inverse i don't remember)
        cy0 = self.params.cy0.value().to(device)                                    # Leadfield Matrix Scaling Parameter

        # Update the Laplacian based on the updated connection gains wll.
        w_l = torch.exp(self.wll) * torch.tensor(self.sc, dtype=torch.float32, device=device)
        w_n_l = w_l / torch.linalg.norm(w_l)
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
        Bxy = torch.diag(2 * torch.pi * omega)
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
              num_epochs: int = 120, learningrate: float = 0.05, lr_2ndLevel: float = 0.05, 
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
                    val_loss, val_corr, val_cos_sim, val_r2 = self.validate(valCOV, loss_method)

                    self.valStats[epoch + 1] = {
                        "loss": val_loss,
                        "corr": val_corr,
                        "cosine_similarity": val_cos_sim,
                        "r2": val_r2
                    }
                    print(f"Validation Loss: {val_loss:.6f}, Corr: {val_corr:.4f}, Cosine Sim: {val_cos_sim:.4f}")
                else:
                    print("Validation data not provided. Skipping validation.")

            if hyperparameter_optimizer:
                scaler.step(hyperparameter_optimizer)
            scaler.step(modelparameter_optimizer)
            scaler.update()

            if debug_grad:
                if hasattr(self.model.params.wll.val, 'grad') and self.model.params.wll.val.grad is not None:
                    wll_grads = self.model.params.wll.val.grad
                    print(f"Gradients for wll: {wll_grads}")
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
            eps = 1e-8
            empData = empCOV.detach().cpu().numpy()
            simData = simCOV.detach().cpu().numpy()

            # Normalize using Frobenius norm (i.e., divide by the Frobenius norm of each matrix)
            emp_norm = empData / (np.linalg.norm(empData, 'fro') + eps)
            sim_norm = simData / (np.linalg.norm(simData, 'fro') + eps)

            sim_rescaled = Scaler(pred=sim_norm, target=emp_norm)

            emp_flat = emp_norm.flatten().reshape(1, -1)
            sim_flat = sim_rescaled.flatten().reshape(1, -1)

            pseudo_fc_cor = np.corrcoef(emp_flat[0], sim_flat[0])[0, 1]
            cos_sim = cosine_similarity(emp_flat, sim_flat)[0, 0]
            r2 = r2_score(emp_flat[0], sim_flat[0])

            # Log metrics
            print(f"Pseudo FC Corr: {pseudo_fc_cor:.4f}")
            print(f"cos_sim: {cos_sim:.4f}")
            print(f"R²: {r2:.4f}")

            if lr_scheduler:
                for param_group in modelparameter_optimizer.param_groups:
                    print('Modelparam_lr:', param_group['lr'])
                if hyperparameter_optimizer:
                    for param_group in hyperparameter_optimizer.param_groups:
                        print('Hyperparam_lr:', param_group['lr'])

            self.trainingStats.appendLoss(np.mean(loss_history))

            # Parameter info for the Epoch
            trackedParam = {}
            exclude_param = ['wll', 'gains_con', 'lm'] #This stores SC and LF which are saved seperately
            if(self.model.track_params):
                for par_name in self.model.track_params:
                    var = getattr(self.model.params, par_name)
                    if (var.fit_par):
                        trackedParam[par_name] = var.value().detach().cpu().numpy().copy()
                        if hasattr(var, 'map_weights'):
                            trackedParam[par_name + "_map_weights"] = var.map_weights.detach().cpu().numpy().copy()
                    if (var.fit_hyper):
                        if var.use_heterogeneity:
                            trackedParam[par_name + "_prior_min_mean"] = var.prior_min_mean.detach().cpu().numpy().copy()
                            trackedParam[par_name + "_prior_min_var"] = var.prior_min_var.detach().cpu().numpy().copy()
                            trackedParam[par_name + "_prior_max_mean"] = var.prior_max_mean.detach().cpu().numpy().copy()
                            trackedParam[par_name + "_prior_max_var"] = var.prior_max_var.detach().cpu().numpy().copy()
                        else:
                            trackedParam[par_name + "_prior_mean"] = var.prior_mean.detach().cpu().numpy().copy()
                            trackedParam[par_name + "_prior_var"] = var.prior_var.detach().cpu().numpy().copy()
            for key, value in self.model.state_dict().items():
                if key not in exclude_param:
                    trackedParam[key] = value.detach().cpu().numpy().ravel().copy()
            self.trainingStats.appendParam(trackedParam)
            # Saving the SC and/or Lead Field State at Every Epoch
            if self.model.use_fit_gains:
                self.trainingStats.appendWll(self.model.wll.detach().cpu().numpy())
                self.trainingStats.appendCONN(self.model.sc_fitted.detach().cpu().numpy())
            if self.model.use_fit_lfm:
                self.trainingStats.appendLF(self.model.lm.detach().cpu().numpy())
            
        # Save the last optimized covariance matrix
        self.lastRec = {}
        self.lastRec["simCOV"] = sim_rescaled

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

            eps = 1e-8
            valData = valCOV.detach().cpu().numpy()
            simData = simCOV.detach().cpu().numpy()

            val_norm = valData / (np.linalg.norm(valData, 'fro') + eps)
            sim_norm = simData / (np.linalg.norm(simData, 'fro') + eps)

            sim_rescaled = Scaler(pred=sim_norm, target=val_norm)

            val_flat = val_norm.flatten().reshape(1, -1)
            sim_flat = sim_rescaled.flatten().reshape(1, -1)

            avg_corr = np.corrcoef(val_flat[0], sim_flat[0])[0, 1]
            avg_cos_sim = cosine_similarity(val_flat, sim_flat)[0, 0]
            r2 = r2_score(val_flat[0], sim_flat[0])

        return loss.item(), avg_corr, avg_cos_sim, r2

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

            eps = 1e-8

            testData = testCOV.detach().cpu().numpy()
            simData = simCOV.detach().cpu().numpy()

            test_norm = testData / (np.linalg.norm(testData, 'fro') + eps)
            sim_norm = simData / (np.linalg.norm(simData, 'fro') + eps)

            sim_rescaled = Scaler(pred=sim_norm, target=test_norm)

            test_flat = test_norm.flatten().reshape(1, -1)
            sim_flat = sim_rescaled.flatten().reshape(1, -1)

            corr = np.corrcoef(test_flat[0], sim_flat[0])[0, 1]
            cos_sim = cosine_similarity(test_flat, sim_flat)[0, 0]
            r2 = r2_score(test_flat[0], sim_flat[0])

        # Save results to the attribute
        self.testStats = {
            "test_loss": test_loss.item(),
            "correlation": corr,
            "cosine_similarity": cos_sim,
            "r2": r2
        }

        print(f"Test Loss: {test_loss.item():.6f}, Correlation: {corr:.4f}, Cosine Sim: {cos_sim:.4f}")

    def simulate(self, empCOV: torch.Tensor, freq_chunk_size=20, debug_sim=False):
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

        eps = 1e-8
        sim = simCOV / (simCOV.norm(p='fro') + eps)
        emp = empCOV / (empCOV.norm(p='fro') + eps)

        sim_rescaled = Scaler(pred=sim, target=emp)

        self.sim = {
            "simCOV": sim_rescaled.detach().cpu().numpy(),
            "empCOV": empCOV.detach().cpu().numpy()
        }

        print("Simulation completed. Covariance matrix saved in '.sim['simCOV']'.")
            
class CostsHP(AbstractLoss):
    def __init__(self, model):
        self.mainLoss = CostsTS()
        self.model = model
    
    def loss(self, empData: torch.Tensor, simData: torch.Tensor, loss_method: str = 'mse' ,debug_loss: bool = False):
        eps = 1e-8  # avoid dividing by very small values in the Normalization procedure
        sim = simData / (simData.norm(p='fro') + eps)
        emp = empData / (empData.norm(p='fro') + eps)

        sim_rescaled = Scaler(pred=sim, target=emp)

        model = self.model

        # define some constants
        lb = 0.001
        w_cost = 1.0
        reg_lambda = 0.01
        weight_reg_scale = 0.01
        reg_scales = {
            'wll': 1e-3,  
            'lm': 1e-2,     
        }

        # define the relu function
        m = torch.nn.ReLU()

        exclude_param = []
        if not model.use_fit_gains:
            exclude_param.append('wll')
            exclude_param.append('gains_con')

        if not model.use_fit_lfm:
            exclude_param.append('lm')
        
        loss_main = self.mainLoss.loss(emp, sim_rescaled, method=loss_method)
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
                    
                    prior_loss_max = torch.sum(reg_scales.get(var_name, 1.0) * (lb + m(var.prior_max_var)) * 
                                   (m(var.val_max) - m(var.prior_max_mean)) ** 2) \
                                   + torch.sum(-torch.log(lb + m(var.prior_max_var)))
                    
                    loss_prior.append(prior_loss_min + prior_loss_max)
                    
                    if debug_loss:
                        print(f"Prior loss for {var_name} min/max: {(prior_loss_min + prior_loss_max).item()}")
                
                if var.fit_par:
                    weight_reg = weight_reg_scale * torch.sum(var.map_weights ** 2)
                    reg_term += weight_reg
                    
                    if debug_loss:
                        print(f"  {var_name} weights regularization: {weight_reg.item()}")

                # Regularization for val_min/val_max (if no prior exists)
                if not var.has_prior:
                    val_reg = reg_lambda * (torch.sum(var.val_min ** 2) + torch.sum(var.val_max ** 2))
                    reg_term += val_reg
                    
                    if debug_loss:
                        print(f"  {var_name} val_min/max regularization: {val_reg.item()}")
            else:
                # Original non-heterogeneous parameter handling
                if var.has_prior:
                    prior_loss = torch.sum(reg_scales.get(var_name, 1.0) * (lb + m(var.prior_var)) * 
                                 (m(var.val) - m(var.prior_mean)) ** 2) \
                                 + torch.sum(-torch.log(lb + m(var.prior_var)))
                    loss_prior.append(prior_loss)
                    
                    if debug_loss:
                        print(f"Prior loss for {var_name}: {prior_loss.item()}")
                else:
                    reg_loss = reg_lambda * torch.sum(var.val ** 2)
                    reg_term += reg_loss
                    
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

def main(n_sub: int = 1, train_region: str = 'Premotor'):
    # Choose Subject
    subject_num = n_sub

    repo_root = Path(__file__).resolve().parent

    # Define the subfolders within your repo
    struct_data_folder = repo_root / 'struct_data'
    conn_data_folder = repo_root / 'conn_data'
    data_folder = repo_root / 'cov_data'
    neuromaps_folder = repo_root / 'neuromaps'

    # Load NeuroMaps
    t1t2_map = np.load(neuromaps_folder / 'T1T2_parc.npy')
    T = erf(t1t2_map)
    T_max, T_min = T.max(), T.min()
    h_map = (T_max - T) / (T_max - T_min)

    # Channel labels and sampling frequency
    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    sampling_freq = metadata[f"SUB{n_sub}"]["sampling_freq"]
    channel_labels = metadata[f"SUB{n_sub}"]["channel_labels"]
    n_channels = len(channel_labels)    

    print(f'Processing Subject {subject_num}')

    # Load correlation on the resting state
    with open(data_folder / f'SUB{n_sub}_RES.pkl', "rb") as f:
        trainData = pickle.load(f)

    empCOV = torch.tensor(trainData['cov_raw'], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

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

    node_size = sc.shape[0]
    output_size = n_channels

    lm_name = os.path.join(struct_data_folder, f'Subject{subject_num}_{train_region}_leadfieldR.npy')
    lm0 = np.load(lm_name)

    wll_name = os.path.join(struct_data_folder, f'Subject{subject_num}_{train_region}_wllR02.npy')
    wll0 = np.load(wll_name)

    omega0 = 10.0
    a0 = -0.5
    sig_omega0 = 0.02

    freqs = np.linspace(1, 80, 800, endpoint=False)
    params = ParamsHP(a=par((a0, 0.0), (a0, 0.0), (0.01*a0, 1e-3), fit_par=True, fit_hyper=False, 
                            use_heterogeneity=True, h_maps=h_map, param_bounds=(1.5*a0, 0.0)), 
                    omega=par(omega0, omega0, 0.02*omega0, fit_par=True, fit_hyper=True),
                    sig_omega=par(sig_omega0, sig_omega0, 0.02*sig_omega0, True, True),
                    g=par(500, 500, 10, True, True), std_in= par(0.3, 0.3, 0.5, True, True),
                    v_d = par(2., 2., 0.05, fit_par=True, fit_hyper=True), 
                    cy0 = par(50, 50, 1, True, True),
                    lm=par(lm0), 
                    wll=par(wll0, wll0, 0.02*np.ones_like(wll0), True, True))

    # Simulation start
    n_epochs = 120
    start_time = time.time()

    model = COVHOPF(node_size, output_size, sampling_freq, sc, dist, freqs, params, lm=lm0, wll_init= wll0, 
                    use_fit_gains=True, use_fit_lfm=False)
    ObjFun = CostsHP(model)
    F = Model_fitting(model, ObjFun)
    scheds = {'O': 'OneCycleLR', 'R': 'ReduceLROnPlateau'}
    sched_type = scheds['R']
    loss_method = 'log_fro'
    print(f"Loss Method: {loss_method}")

    # Train
    F.train(empCOV = empCOV, num_epochs = n_epochs, 
            lr_scheduler = True, scheduler_type = sched_type, loss_method = loss_method, debug_loss = False)

    # Simulate the model after training
    F.simulate(empCOV = empCOV) 

    # Save fitting results to a file using pickle
    results_folder = repo_root / 'Results_Neuromaps'
    results_folder.mkdir(exist_ok=True)
    store_filename = results_folder / f'Subject{subject_num}_{train_region}_{loss_method}_{sched_type}_fitting_results.pkl'
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
    regions = ['Prefrontal', 'Premotor']
    for subj in range(6):
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        main(n_sub = subj+1, train_region = regions[1])
    for subj in range(6):
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        main(n_sub = subj+1, train_region = regions[0])
