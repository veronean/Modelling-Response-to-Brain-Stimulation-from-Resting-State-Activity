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
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from pathlib import Path

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
torch.cuda.reset_accumulated_memory_stats()

import sys

#neuroimaging packages
import mne

repo_root = Path(__file__).resolve().parent

# Define the subfolders within your repo
struct_data_folder = repo_root / 'struct_data'
conn_data_folder = repo_root / 'conn_data'
data_folder = repo_root / 'cov_data'
tms_data_folder = repo_root / 'eegtms_data'
rest_fitpar_folder = repo_root / 'rest_fitpar'
neuromaps_folder = repo_root / 'neuromaps'

# Suppression block
class Suppressor:
    def __enter__(self):
        self.stdout_original = sys.stdout
        sys.stdout = open('/dev/null', 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.stdout_original

class par:
    def __init__(self, val, prior_mean = None, prior_var = None, 
                 fit_par = False, fit_hyper = False, device = 'cuda'):
        '''
        Parameters
        ----------
        val : Float (or Array)
            The parameter value (or an array of node specific parameter values)
        prior_mean : Float
            Prior mean of the data value
        prior_var : Float
            Prior variance of the value
        fit_par: Bool
            Whether the parameter value should be set to as a PyTorch Parameter
        fit_hyper : Bool
            Whether the parameter prior mean and prior variance should be set as a PyTorch Parameter
        device: torch.device
            Whether to run on CPU or GPU
        '''

        self.device = device

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
            self.val = torch.nn.parameter.Parameter(self.val)
        if fit_hyper:
            self.prior_mean = torch.nn.parameter.Parameter(self.prior_mean)
            self.prior_var = torch.nn.parameter.Parameter(self.prior_var)

    def value(self):
        '''
        Returns
        ---------
        Tensor of Value
            The parameter value(s) as a PyTorch Tensor
        '''
        return self.val

    def npValue(self):
        '''
        Returns
        --------
        NumPy of Value
            The parameter value(s) as a NumPy Array
        '''
        return self.val.detach().clone().cpu().numpy()

    def to(self, device='cuda'):
        self.device = device

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
        if self.has_prior:
            self.var = self.prior_mean.detach() + self.prior_var.detach() * torch.randn(1, device=self.device)
            if self.fit_par:
                self.val = torch.nn.parameter.Parameter(self.val)
        else:
            raise ValueError("must have priors provided at par object initialization to use this method")

    def __pos__(self):
        return self.val

    def __neg__(self):
        return -self.val

    def __add__(self, num):
        return self.val + num

    def __radd__(self, num):
        return num + self.val

    def __sub__(self, num):
        return self.val - num

    def __rsub__(self, num):
        return num - self.val

    def __mul__(self, num):
        return self.val * num

    def __rmul__(self, num):
        return num * self.val

    def __truediv__(self, num):
        return self.val / num

    def __rtruediv__(self, num):
        return num / self.val

class Recording():
    '''
    This class is responsible for holding timeseries of empirical and simulated data. It is:
        - Part of the input and output of Model_fitting.

    Attributes
    -------------
    data : Numpy Array or Tensor of dimensions num_regions x ts_length
        The time series data, either empirical or simulated
    step_size : Float
        The step size of the time points in the data class
    modality : String
        The name of the modality of the time series
    numNodes : Int
        The number of nodes it time series.
    length : Int
        The number of time points in the time series.
    '''

    def __init__(self, data, step_size, modality = "", device = 'cuda'):
        '''
        Parameters
        -----------
        data : Numpy Array or Tensor of dimensions num_regions x ts_length
            The time series data, either empirical or simulated
        step_size : Float
            The step size of the time points in the data class
        modality : String
            The name of the modality of the time series
        '''

        self.device = device
        if not(torch.is_tensor(data)):
            data = torch.tensor(data, device=device) # Store as Tensor

        self.data = data
        self.step_size = step_size
        self.modality = modality
        self.numNodes = self.data.shape[0]
        self.length = self.data.shape[1]

    def pyTS(self):
        '''
        Returns
        --------
        Tensor of num_regions x ts_length

        '''
        return self.data

    def npTS(self):
        '''
        Returns
        ---------
        Numpy Array of num_regions x ts_length

        '''
        return self.data.cpu().numpy()

    def npNodeByTime(self):
        '''
        Returns
        ---------
        Numpy Array of num_regions x ts_length

        '''
        return self.data.cpu().numpy()

    def npTimeByNodes(self):
        '''
        Returns
        ---------
        Numpy Array of ts_length x num_regions

        '''
        return self.data.cpu().numpy().T

    def length(self):
        '''
        Returns
        ---------
        The time series length

        '''
        return self.length

    def windowedTensor(self, TPperWindow):
        '''
        This method is called by the Model_fitting Class during training 
        to reshape the data into windowed segments (adds another dimension).

        Parameters
        -----------
        TPperWindow : Int
            The number of time points in the window that will be back propagated

        Returns
        ---------
        Tensor: num_windows x num_regions x window_length
            The time series data in a windowed format
        '''

        node_size = self.data.shape[0]
        length_ts = self.data.shape[1]
        num_windows = length_ts // TPperWindow

        data_out = torch.zeros((num_windows, node_size, TPperWindow), device=self.data.device)

        for i_win in range(num_windows):
            data_out[i_win, :, :] = self.data[:, i_win * TPperWindow:(i_win + 1) * TPperWindow]

        return data_out

class AbstractNMM(torch.nn.Module):
    def __init__(self):
        super(AbstractNMM, self).__init__()

        self.state_names = ["None"]
        self.output_names = ["None"]
        self.track_params = []

        self.use_fit_gains = False
        self.use_fit_lfm = False

    def info(self):
        return {"state_names": self.state_names,
                "output_names": self.output_names,
                "track_params": self.track_params}

    def setModelParameters(self):
        pass

    def createIC(self, ver):
        pass

    def createDelayIC(self, ver):
        return torch.tensor(1.0) 
    
    def forward(self, external, hx, hE):
        pass


class AbstractParams:
    def __init__(self, **kwargs):
        pass

    def to(self, device):
        vars_names = [a for a in dir(self) if not a.startswith('__')]
        for var_name in vars_names:
            var = getattr(self, var_name)
            if (type(var) == par):
                var.to(device)

class AbstractLoss:
    def __init__(self, simKey = None, device = 'cuda'):
        self.simKey = simKey
        self.device = device

    def loss(self, simData, empData):
        pass

class TrainingStats:
    '''
    This class is responsible for recording stats during training including:
        - The training and validation losses over time
        - The change in model parameters over time

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
        self.model_info = model.info()
        self.model = model
        self.track_params = model.track_params

        self.loss = []

        self.connectivity = []
        self.wll = []
        self.leadfield = []

        self.fit_params = {}

    def save(self, filename):
        '''
        Parameters
        ------------
        filename : String
            The filename to use to save the TrainingStats as a pickle object.

        '''

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def reset(self):
        '''
        Resets the attributes of the model to a pre-training state.

        '''

        self.loss = []
        self.connectivity = []
        self.wll = []
        self.leadfield = []
        self.fit_params = {}

    def appendLoss(self, newValue):
        """
        Append Trainig Loss

        Parameters
        -----------
        newValue : Float
            The loss value of objective function being tracked.

        """
        self.loss.append(newValue)

    def appendSC(self, newValue):
        """
        Append Network Connections

        Parameters
        -----------
        newValue : Array
            Current state of the structural connectivity being tracked.

        """
        self.connectivity.append(newValue)

    def appendConnGain(self, newValue):
        """
        Append Connection Gain Loss

        Parameters
        -----------
        newValue : Array
            Current state of connection gain matrices being tracked.

        """
        self.wll.append(newValue['wll'])

    def appendLF(self, newValue):
        """
        Append Lead Field Loss

        Parameters
        -----------
        newValue : Array
            Current state of a lead field matrix being tracked.

        """
        self.leadfield.append(newValue)

    def appendParam(self, newValues):
        """
        Append Fit Parameters

        Parameters
        ----------
        newValues : Dict
            Dictionary with current states of each model parameter being tracked.

        """
        if (self.fit_params == {}):
            for name in newValues.keys():
                self.fit_params[name] = [newValues[name]]
        else:
            for name in newValues.keys():
                self.fit_params[name].append(newValues[name])

class AbstractFitting():
    def __init__(self, model: AbstractNMM, cost: AbstractLoss, device = torch.device('cpu')):

        self.model = model
        self.cost = cost
        self.device = device

        self.trainingStats = TrainingStats(self.model)
        self.lastRec = None

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train():
        pass

    def evaluate():
        pass

    def simulate():
        pass


class ParamsHP(AbstractParams):
    """
    A class for setting the parameters of a neural mass model for EEG data fitting.

    Attributes:
        a (par): The node's bifurcation parameter (s-1)
        omega (par): The intrinsic angular frequency (in rad.s-1)
        std_in (par): The standard deviation of the input noise.
        g (par): The strength of the connections
        mu (par): Conduction Velocity.
        cy0 (par): Leadfield Slope.
        y0 (par): Leadfield Intercept.
        k (par): TMS impulse global gain.
        ki (par): TMS impulse spatial input.
    """

    def __init__(self, **kwargs):
        param = {
            "a": par(-0.5),
            "omega": par(10.0),
            "sig_omega": par(1.0),
            "g": par(500),
            "std_in": par(100),
            "mu": par(.5),

            "cy0": par(5),
            "y0": par(2),

            "k": par(5),
            "ki": par(1)
        }

        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])


class CostsTS(AbstractLoss):
    def __init__(self, simKey):
        super(CostsTS, self).__init__(simKey)
        self.simKey = simKey

    def loss(self, simData: dict, empData: torch.Tensor, method: str = 'RMSE'):
        """
        Calculate the Loss Function between the simFC and empFC.

        Parameters
        ----------
        simData: dict of tensor with node_size X datapoint
            simulated EEG
        empData: tensor with node_size X datapoint
            empirical EEG
        method: str that specify which loss function to use between RMSE, Pearson and Cosine
        """

        sim = simData[self.simKey]
        emp = empData

        if method == 'RMSE':
            return torch.sqrt(torch.mean((sim - emp) ** 2))

        elif method == 'Pearson':
            sim = sim - sim.mean(dim=1, keepdim=True)
            emp = emp - emp.mean(dim=1, keepdim=True)

            sim_std = sim.std(dim=1, unbiased=False, keepdim=True) + 1e-8
            emp_std = emp.std(dim=1, unbiased=False, keepdim=True) + 1e-8

            sim = sim / sim_std
            emp = emp / emp_std

            return -torch.mean(torch.sum(sim * emp, dim=1))

        elif method == 'Cosine':
            sim_norm = torch.norm(sim, dim=1, keepdim=True) + 1e-8
            emp_norm = torch.norm(emp, dim=1, keepdim=True) + 1e-8

            sim = sim / sim_norm
            emp = emp / emp_norm

            return -torch.mean(torch.sum(sim * emp, dim=1))

class RNNHOPF(AbstractNMM):
    """
    A module for forward model (HOPF WHOLE BRAIN) to simulate EEG signals

    Attibutes
    ---------
    state_size : int
        Number of states in the Hopf model
    output_size : int
        Number of EEG channels.
    node_size: int
        Number of ROIs
    hidden_size: int
        Number of step_size for each sampling step
    step_size: float
        Integration step for forward model
    tr : float
        Sampling rate of the simulated EEG signals
    TRs_per_window: int
        Number of EEG signals to simulate
    sc: ndarray (node_size x node_size) of floats
        Structural connectivity
    lm: ndarray of floats
        Leadfield matrix from source space to EEG space
    dist: ndarray of floats
        Distance matrix
    use_fit_gains: bool
        Flag for fitting gains. 1: fit, 0: not fit
    use_fit_lfm: bool
        Flag for fitting the leadfield matrix. 1: fit, 0: not fit
    params: ParamsHP
        Model parameters object.

    Methods
    -------
    createIC(self, ver):
        Creates the initial conditions for the model.
    createDelayIC(self, ver):
        Creates the initial conditions for the delays.
    setModelParameters(self):
        Sets the parameters of the model.
    forward(input, noise_out, hx)
        Forward pass for generating a number of EEG signals with current model parameters

    """

    def __init__(self, node_size: int,
                 TRs_per_window: int, step_size: float, output_size: int, tr: float, sc: np.ndarray, 
                 lm: np.ndarray, dist: np.ndarray, params: ParamsHP, conn_gains: dict = None, 
                 use_fit_gains: bool = False, use_fit_lfm: bool = False) -> None:
        """
        Parameters
        ----------
        node_size: int
            Number of ROIs
        TRs_per_window: int 
            Number of EEG signals to simulate
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
        use_fit_gains: bool
            Flag for fitting gains. 1: fit, 0: not fit
        use_fit_lfm: bool
            Flag for fitting the leadfield matrix. 1: fit, 0: not fit
        params: ParamsHP
            Model parameters object.
        """
       
        super(RNNHOPF, self).__init__()
        self.device = 'cuda'
        self.state_names = ['X', 'Y']
        self.output_names = ["eeg"]
        self.track_params = []                  #Is populated during setModelParameters()

        self.model_name = "Hopf"
        self.state_size = 2                     # 2 states Hopf model
        self.tr = tr                            # tr ms (integration step 0.1 ms)
        self.step_size = torch.tensor(step_size, dtype=torch.float32, device=self.device) 
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window    # size of the batch used at each step
        self.node_size = node_size              # num of ROI
        self.output_size = output_size          # num of EEG channels
        self.sc = torch.tensor(sc, dtype=torch.float32, device=self.device)  # matrix node_size x node_size structure connectivity
        self.dist = torch.tensor(dist, dtype=torch.float32, device=self.device)
        self.lm = lm
        self.use_fit_gains = use_fit_gains      # flag for fitting gains
        self.use_fit_lfm = use_fit_lfm
        self.params = params
        self.output_size = lm.shape[0]          # number of EEG channels

        if conn_gains is None:
            conn_gains = {
                'wll': np.zeros((self.node_size, self.node_size)) + 0.05,
            }
            print("Connection gains not provided, using default initialization.")
    
        self.wll = conn_gains['wll']

        self.setModelParameters()

    def info(self):
        """
        Returns a dictionary with the names of the states and the output.

        Returns
        -------
        Dict[str, List[str]]
        """

        return {"state_names": ['X', 'Y'], "output_names": ["eeg"]}

    def createIC(self, ver):
        """
        Creates the initial conditions for the model.

        Parameters
        ----------
        ver : int 
            Initial condition version.

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, state_size) with random values between `state_lb` and `state_ub`.
        """

        state_lb = -0.5
        state_ub = 0.5

        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size, self.state_size)),
                             dtype=torch.float32, device=self.device)

    def createDelayIC(self, ver):
        """
        Creates the initial conditions for the delays.

        Parameters
        ----------
        ver : int
            Initial condition version.

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, delays_max) with random values between `state_lb` and `state_ub`.
        """

        delays_max = 500
        state_ub = 0.5
        state_lb = -0.5

        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size, delays_max)), dtype=torch.float32, device=self.device)

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
        else:
            self.wll = torch.tensor(self.wll, dtype=torch.float32, device=self.device)

        # If use_fit_lfm is True, set lm as an attribute as type Parameter
        if self.use_fit_lfm:
            self.lm = nn.Parameter(torch.tensor(self.lm, dtype=torch.float32, device=self.device))
            param_reg.append(self.lm)
        else:
            self.lm = torch.tensor(self.lm, dtype=torch.float32, device=self.device)

        var_names = [a for a in dir(self.params) if (type(getattr(self.params, a)) == par)]
        for var_name in var_names:
            var = getattr(self.params, var_name)
            if (var.fit_hyper):
                if var_name in ['lm', 'wll']:
                    init_val = torch.normal(mean=var.prior_mean, std=torch.sqrt(var.prior_var)).to(self.device)
                    var.val = nn.Parameter(init_val)
                    param_hyper.append(var.prior_mean)
                    param_hyper.append(var.prior_var)
                elif (var != 'std_in'):
                    var.randSet() 
                    param_hyper.append(var.prior_mean)
                    param_hyper.append(var.prior_var)

            if (var.fit_par):
                param_reg.append(var.val) 

            if (var.fit_par | var.fit_hyper):
                self.track_params.append(var_name) #NMM Parameters

            if var_name in ['lm', 'wll']:
                setattr(self, var_name, var.val)

        self.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}


    def forward(self, external, hx, hEx, hEy):
        """
        This function carries out the forward Euler integration method for the Hopf Whole Brain model,
        with time delays, connection gains, and external inputs considered. The function
        updates the state of each neural population and computes the EEG signals at each time step.
        """

        # Generate the ReLU module
        m = torch.nn.ReLU()
        n = self.node_size

        # Define some constants
        con_1 = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        conduct_lb = 1.5    # lower bound for conduct velocity
        u_sys_ub = 500      # the bound of the input for second order system
        noise_std_lb = 20   # lower bound of std of noise
        lb = 0.01           # lower bound of local gains
        k_lb = 0.5          # lower bound of coefficient of external inputs
        omega_ub = 2.0
        eps = torch.randn(n, device=self.device)

        # Defining NMM Parameters
        a0 = self.params.a.value().to(self.device)
        a = a0 * torch.ones(n, device=self.device) if a0.dim() == 0 else a0
        
        mean_omega_raw = m(self.params.omega.value()).to(self.device)   # Intrinsic angular frequency (rad.s^-1)
        mean_omega = torch.sigmoid(mean_omega_raw) * omega_ub           # Apply upper bound to mean
        sig_omega = m(self.params.sig_omega.value()).to(self.device)    # Variance of the angular frequency
        omega_raw = mean_omega + sig_omega * eps
        omega = 2 * torch.pi * torch.clamp(omega_raw, max=omega_ub)

        g = (lb * con_1 + m(self.params.g.value())).to(self.device)
        std_in = (noise_std_lb * con_1 + m(self.params.std_in.value())).to(self.device)
        y0 = self.params.y0.value().to(self.device)
        mu = (conduct_lb * con_1 + m(self.params.mu.value())).to(self.device)
        k = (k_lb * con_1 + m(self.params.k.value())).to(self.device)
        cy0 = self.params.cy0.value().to(self.device)
        ki = self.params.ki.value().to(self.device)

        next_state = {}

        X = hx[:,0:1]
        Y = hx[:,1:2]

        dt = self.step_size

        w_l = torch.exp(self.wll) * self.sc
        w_n_l = w_l / torch.linalg.norm(w_l)
        self.sc_fitted = w_n_l
        dg_l = -torch.diag(torch.sum(w_n_l, dim=1))

        self.delays = ((self.dist / mu).type(torch.int64)).to(self.device)

        # Placeholder for the updated current state
        current_state = torch.zeros_like(hx, device=self.device)

        eeg_window = []
        X_window = []
        Y_window = []

        # Use the forward model to get EEG signal at the i-th element in the window.
        for i_window in range(self.TRs_per_window):
            for step_i in range(self.steps_per_TR):
                Edx = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32, device=self.device)  
                Edy = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32, device=self.device) 
                hEx_new = hEx.clone()
                hEy_new = hEy.clone()
                Edx = hEx_new.gather(1, self.delays)
                Edy = hEy_new.gather(1, self.delays)
                LEd_x = torch.reshape(torch.sum(w_n_l * torch.transpose(Edx, 0, 1), 1),
                                      (self.node_size, 1))
                LEd_y = torch.reshape(torch.sum(w_n_l * torch.transpose(Edy, 0, 1), 1),
                                    (self.node_size, 1))

                # TMS input
                u_tms = external[:, step_i:step_i + 1, i_window]
                rX = k * ki * u_tms + std_in * torch.randn(self.node_size, 1, device=self.device) + \
                    1 * g * (LEd_x + 1 * torch.matmul(dg_l, X)) 
                rY = std_in * torch.randn(self.node_size, 1, device=self.device) + \
                    1 * g * (LEd_y + 1 * torch.matmul(dg_l, Y))
                
                # Update the states with every step size.
                dX = X + dt * ((a.view(-1, 1) - X**2 - Y**2)*X - omega.view(-1, 1)*Y + u_sys_ub * torch.tanh(rX / u_sys_ub))
                dY = Y + dt * ((a.view(-1, 1) - X**2 - Y**2)*Y + omega.view(-1, 1)*X + u_sys_ub * torch.tanh(rY / u_sys_ub))

                # Calculate the saturation for model states (for stability and gradient calculation).
                X = 1000*torch.tanh(dX/1000)
                Y = 1000*torch.tanh(dY/1000)

                # Update placeholders for buffer
                hEx[:, 0] = X[:, 0]
                hEy[:, 0] = Y[:, 0]

            # Capture the states at every tr in the placeholders for checking them visually.
            X_window.append(X)
            Y_window.append(Y)

            hEx = torch.cat([X, hEx[:, :-1]], dim=1)  
            hEy = torch.cat([Y, hEy[:, :-1]], dim=1) 

            # Capture the states at every tr in the placeholders which is then used in the cost calculation.
            lm_t = (self.lm.T / torch.sqrt(self.lm ** 2).sum(1)).T
            self.lm_t = (lm_t - 1 / self.output_size * torch.matmul(torch.ones((1, self.output_size), device=self.device), lm_t))
            temp = cy0 * torch.matmul(self.lm_t, X[:200, :]) - 1 * y0
            eeg_window.append(temp)

        # Update the current state.
        current_state = torch.cat([X, Y], dim=1)
        next_state['current_state'] = current_state
        next_state['eeg'] = torch.cat(eeg_window, dim=1)
        next_state['X'] = torch.cat(X_window, dim=1)
        next_state['Y'] = torch.cat(Y_window, dim=1)

        return next_state, hEx, hEy
    
class Model_fitting(AbstractFitting):
    """
    This Model_fitting class is able to fit evoked potential data
    for which the input training data is empty or some stimulus to one or more NMM nodes,
    and the label is an associated empirical neuroimaging recording.

    Attributes
    ----------
    model: AbstractNMM
        Whole Brain Model to Simulate
    cost: AbstractLoss
        A particular objective function which the model will be optimized for.
    trainingStats: TrainingStats
        Information about objective function loss and parameter values over training windows/epochs
    lastRec: Recording
        The last simulation of fitting(), evaluation(), or simulation()
    device : torch.device
        Whether the fitting is to run on CPU or GPU
    """

    def __init__(self, model: AbstractNMM, cost: AbstractLoss, device = 'cuda'):
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

    def save(self, filename):
        """
        Parameters
        ----------
        filename: String
            filename to use when saving object to file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train(self, u, empRecs: list,
              num_epochs: int, TPperWindow: int, learningrate: float = 0.05, 
              lr_2ndLevel: float = 0.05, lr_scheduler: bool = False):
        """
        Parameters
        ----------
        u: type
           This stimulus is the ML "Training Input"
        empRec: list of Recording
            This is the ML "Training Labels"
        num_epochs: int
            the number of times to go through the entire training data set
        TPperWindow: int
            Number of Empirical Time Points per window. model.forward does one window at a time.
        learningrate: float
            rate of gradient descent
        lr_2ndLevel: float
            learning rate for priors of model parameters, and possibly others
        lr_scheduler: bool
            Whether to use the learning rate scheduler
        """

        # Define two different optimizers for each group
        modelparameter_optimizer = optim.Adam(self.model.params_fitted['modelparameter'], 
                                              lr=learningrate, eps=1e-7)
        use_hyper = len(self.model.params_fitted['hyperparameter']) > 0
        if use_hyper:
            hyperparameter_optimizer = optim.Adam(
                self.model.params_fitted['hyperparameter'], 
                lr=lr_2ndLevel, eps=1e-7
        )
        else:
            hyperparameter_optimizer = None

        # Define the learning rate schedulers for each group of parameters
        if lr_scheduler:
            total_steps = 0
            for empRec in empRecs:
                total_steps += int(empRec.length/TPperWindow)*num_epochs

            # total_steps = self.num_windows*num_epochs
            if use_hyper:
                hyperparameter_scheduler = optim.lr_scheduler.OneCycleLR(hyperparameter_optimizer,
                                                                        lr_2ndLevel,
                                                                        total_steps,
                                                                        anneal_strategy = "cos")
                hlrs = []
            modelparameter_scheduler = optim.lr_scheduler.OneCycleLR(modelparameter_optimizer,
                                                                     learningrate,
                                                                     total_steps,
                                                                     anneal_strategy = "cos")
            mlrs = []

        # initial state
        X = self.model.createIC(ver = 0)
        # initials of history of E
        hEx = self.model.createDelayIC(ver = 0)
        hEy = self.model.createDelayIC(ver = 0)

        # define masks for getting lower triangle matrix indices
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # LOOP 1/4: Number of Training Epochs
        for i_epoch in range(num_epochs):

            # TRAINING_STATS: placeholders for the history of trainingStats
            loss_his = []  # loss placeholder to take the average for the epoch at the end of the epoch

            print(f"Epoch:  {i_epoch+1}/{num_epochs}")

            # LOOP 2/4: Number of Recordings in the Training Dataset
            for empRec in empRecs:
                windowedTS = empRec.windowedTensor(TPperWindow)

                # TIME SERIES: Create placeholders for the simulated states and outputs of entire time series corresponding to one recording
                windListDict = {} # A Dictionary with a List of windowed time series
                for name in set(self.model.state_names + self.model.output_names):
                    windListDict[name] = []

                # initial the external inputs
                external = torch.tensor(
                    np.zeros([self.model.node_size, self.model.steps_per_TR, self.model.TRs_per_window]),
                    dtype=torch.float32, device=self.device)

                # LOOP 3/4: Number of windowed segments for the recording
                for win_idx in range(windowedTS.shape[0]):

                    # Reset the gradient to zeros after update model parameters.
                    if use_hyper:
                        hyperparameter_optimizer.zero_grad()
                    modelparameter_optimizer.zero_grad()

                    # if the external not empty
                    if not isinstance(u, int):
                        external = torch.tensor(
                            (u[:, :, win_idx * self.model.TRs_per_window:(win_idx + 1) * self.model.TRs_per_window]),
                            dtype=torch.float32, device=self.device)

                    # LOOP 4/4: The loop within the forward model (numerical solver), which is number of time points per windowed segment
                    next_window, hEx_new, hEy_new = self.model(external, X, hEx, hEy)

                    # Get the batch of empirical signal.
                    ts_window = torch.tensor(windowedTS[win_idx, :, :], dtype=torch.float32, device=self.device)

                    # calculating loss
                    loss = self.cost.loss(next_window, ts_window)

                    # TIME SERIES: Put the window of simulated forward model.
                    for name in set(self.model.state_names + self.model.output_names):
                        windListDict[name].append(next_window[name].detach().cpu().numpy())

                    # TRAINING_STATS: Adding Loss for every training window (corresponding to one backpropagation)
                    loss_his.append(loss.detach().cpu().numpy())

                    # Calculate gradient using backward (backpropagation) method of the loss function.
                    loss.backward(retain_graph=True)
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                    # Optimize the model based on the gradient method in updating the model parameters.
                    if use_hyper:
                        hyperparameter_optimizer.step()

                    modelparameter_optimizer.step()

                    if lr_scheduler:
                        #appending (needed to plot learning rate)
                        if use_hyper:
                            hlrs.append(hyperparameter_optimizer.param_groups[0]["lr"])
                        mlrs.append(modelparameter_optimizer.param_groups[0]["lr"])

                        # schedular step
                        if use_hyper:
                            hyperparameter_scheduler.step()
                        modelparameter_scheduler.step()

                    X = next_window['current_state'].detach().clone()
                    hEx = hEx_new.detach().clone()
                    hEy = hEy_new.detach().clone()

                ts_emp = torch.cat(list(windowedTS), dim=1).detach().cpu().numpy()
                fc = np.corrcoef(ts_emp)

                # TIME SERIES: Concatenate all windows together to get one recording
                for name in set(self.model.state_names + self.model.output_names):
                        windListDict[name] = np.concatenate(windListDict[name], axis=1)
                
                ts_sim = windListDict[self.model.output_names[0]]
                fc_sim = np.corrcoef(ts_sim[:, 10:])
                
                cos_sim = np.mean([
                    np.dot(ts_sim[i], ts_emp[i]) / 
                    (np.linalg.norm(ts_sim[i]) * np.linalg.norm(ts_emp[i]) + 1e-12)
                    for i in range(ts_sim.shape[0])
                ])

                pseudo_fc_corr = np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1]
                r2 = r2_score(fc[mask_e], fc_sim[mask_e])

                print('loss:', loss.detach().cpu().numpy())
                print(f'Pseudo FC_cor: {pseudo_fc_corr:.4f}')
                print(f'cos_sim:  {cos_sim:.4f}')
                print(f'R2: {r2:.4f}')
                print()
                
                if lr_scheduler:
                    print('Modelparam_lr: ', modelparameter_scheduler.get_last_lr()[0])
                    if use_hyper:
                        print('Hyperparam_lr: ', hyperparameter_scheduler.get_last_lr()[0])

            # TRAINING_STATS: Put the updated model parameters into the history placeholders at the end of every epoch.
            # Additing Mean Loss for the Epoch
            self.trainingStats.appendLoss(np.mean(loss_his))
            # NMM/Other Parameter info for the Epoch (a list where a number is recorded every window of every record)
            trackedParam = {}
            exclude_param = ['wll', 'gains_con', 'lm'] #This stores SC and LF which are saved seperately
            if(self.model.track_params):
                for par_name in self.model.track_params:
                    var = getattr(self.model.params, par_name)
                    if (var.fit_par):
                        trackedParam[par_name] = var.value().detach().cpu().numpy().copy()
                    if (var.fit_hyper):
                        trackedParam[par_name + "_prior_mean"] = var.prior_mean.detach().cpu().numpy().copy()
                        trackedParam[par_name + "_prior_var"] = var.prior_var.detach().cpu().numpy().copy()
            for key, value in self.model.state_dict().items():
                if key not in exclude_param:
                    trackedParam[key] = value.detach().cpu().numpy().ravel().copy()
            self.trainingStats.appendParam(trackedParam)
            # Saving the SC and/or Lead Field State at Every Epoch
            if self.model.use_fit_gains:
                self.trainingStats.appendConnGain(
                    {
                        'wll': self.model.wll.detach().cpu().numpy(),
                    }
                )
                self.trainingStats.appendSC(self.model.sc_fitted.detach().cpu().numpy())
            if self.model.use_fit_lfm:
                self.trainingStats.appendLF(self.model.lm.detach().cpu().numpy())

        # Saving the last recording of training as a Model_fitting attribute
        self.lastRec = {}
        for name in set(self.model.state_names + self.model.output_names):
            self.lastRec[name] = Recording(windListDict[name], step_size = self.model.step_size)

    def evaluate(self, u, empRec: list, TPperWindow: int, base_window_num: int = 0, transient_num: int = 10):
        """
        Parameters
        ----------
        u : int or Tensor
            external or stimulus
        empRec: list of Recording
            This is the ML "Training Labels"
        TPperWindow: int
            Number of Empirical Time Points per window. model.forward does one window at a time.
        base_window_num : int
            length of num_windows for resting
        transient_num : int
            The number of initial time points to exclude from some metrics
        -----------
        """

        self.model.eval()

        # initial state
        X = self.model.createIC(ver = 1)
        # initials of history of E
        hEx = self.model.createDelayIC(ver = 1)
        hEy = self.model.createDelayIC(ver = 1)

        # define mask for getting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # Create placeholders for the simulated states and outputs of entire time series corresponding to one recording
        windListDict = {} # A Dictionary with a List of windowed time series
        for name in set(self.model.state_names + self.model.output_names):
            windListDict[name] = []

        num_windows = int(empRec.length/TPperWindow)
        u_hat = np.zeros(
            (self.model.node_size,self.model.steps_per_TR,
             base_window_num*self.model.TRs_per_window + num_windows*self.model.TRs_per_window))
        u_hat[:, :, base_window_num * self.model.TRs_per_window:] = u

        # LOOP 1/2: The number of windows in a recording
        with torch.no_grad():
            for win_idx in range(num_windows + base_window_num):

                # Get the input and output noises for the module.
                external = torch.tensor(
                    (u_hat[:, :, win_idx * self.model.TRs_per_window:(win_idx + 1) * self.model.TRs_per_window]),
                    dtype=torch.float32, device=self.device)

                # LOOP 2/2: The loop within the forward model (numerical solver), which is number of time points per windowed segment
                next_window, hEx_new, hEy_new = self.model.forward(external, X, hEx, hEy)

                # TIME SERIES: Put the window of simulated forward model.
                if win_idx > base_window_num - 1:
                    for name in set(self.model.state_names + self.model.output_names):
                        windListDict[name].append(next_window[name].detach().cpu().numpy())

                X = next_window['current_state'].detach().clone() # dtype=torch.float32
                hEx = hEx_new.detach().clone() #dtype=torch.float32
                hEy = hEy_new.detach().clone() #dtype=torch.float32

        windowedTS = empRec.windowedTensor(TPperWindow)
        ts_emp = torch.cat(list(windowedTS),dim=1).detach().cpu().numpy()
        fc = np.corrcoef(ts_emp)

        # TIME SERIES: Concatenate all windows together to get one recording
        for name in set(self.model.state_names + self.model.output_names):
            windListDict[name] = np.concatenate(windListDict[name], axis=1)

        ts_sim = windListDict[self.model.output_names[0]]
        fc_sim = np.corrcoef(ts_sim[:, transient_num:])

        cos_sim = np.mean([
            np.dot(ts_sim[i], ts_emp[i]) / 
            (np.linalg.norm(ts_sim[i]) * np.linalg.norm(ts_emp[i]) + 1e-12)
            for i in range(ts_sim.shape[0])
        ])

        pseudo_fc_corr = np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1]
        r2 = r2_score(fc[mask_e], fc_sim[mask_e])

        print(' === Evaluation === ')
        print(f'Pseudo FC_cor:  {pseudo_fc_corr:.4f}')
        print(f'cos_sim:  {cos_sim:.4f}')
        print(f'R2: {r2:.4f}')

        # Saving the last recording of training as a Model_fitting attribute
        self.lastRec = {}
        for name in set(self.model.state_names + self.model.output_names):
            self.lastRec[name] = Recording(windListDict[name], step_size = self.model.step_size)

    def simulate(self, u, numTP: int, base_window_num: int = 0, transient_num: int = 10):
        """
        Parameters
        ----------
        u : int or Tensor
            external or stimulus
        numTP : int
            The number of time points to simulate
        base_window_num : int
            length of num_windows for resting
        transient_num : int
            The number of initial time points to exclude from some metrics
        -----------
        """

        self.model.eval()
        num_windows = int(numTP/self.model.TRs_per_window)
        TPperWindow = numTP 

        # initial state
        X = self.model.createIC(ver = 1)
        # initials of history of E
        hEx = self.model.createDelayIC(ver = 1)
        hEy = self.model.createDelayIC(ver = 1)

        # define mask for getting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # Create placeholders for the simulated states and outputs of entire time series corresponding to one recording
        windListDict = {} # A Dictionary with a List of windowed time series
        for name in set(self.model.state_names + self.model.output_names):
            windListDict[name] = []

        u_hat = np.zeros(
            (self.model.node_size,self.model.steps_per_TR,
             base_window_num*self.model.TRs_per_window + num_windows*self.model.TRs_per_window))
        u_hat[:, :, base_window_num * self.model.TRs_per_window:] = u

        # LOOP 1/2: The number of windows in a recording
        with torch.no_grad():
            for win_idx in range(num_windows + base_window_num):

                # Get the input and output noises for the module.
                external = torch.tensor(
                    (u_hat[:, :, win_idx * self.model.TRs_per_window:(win_idx + 1) * self.model.TRs_per_window]),
                    dtype=torch.float32, device=self.device)

                # LOOP 2/2: The loop within the forward model (numerical solver), which is number of time points per windowed segment
                next_window, hEx_new, hEy_new = self.model.forward(external, X, hEx, hEy)

                # TIME SERIES: Put the window of simulated forward model.
                if win_idx > base_window_num - 1:
                    for name in set(self.model.state_names + self.model.output_names):
                        windListDict[name].append(next_window[name].detach().cpu().numpy())

                X = next_window['current_state'].detach().clone() # dtype=torch.float32
                hEx = hEx_new.detach().clone() #dtype=torch.float32
                hEy = hEy_new.detach().clone() #dtype=torch.float32

        # TIME SERIES: Concatenate all windows together to get one recording
        for name in set(self.model.state_names + self.model.output_names):
            windListDict[name] = np.concatenate(windListDict[name], axis=1)

        ts_sim = windListDict[self.model.output_names[0]]
        fc_sim = np.corrcoef(ts_sim[:, transient_num:])

        # Saving the last recording of training as a Model_fitting attribute
        self.lastRec = {}
        for name in set(self.model.state_names + self.model.output_names):
            self.lastRec[name] = Recording(windListDict[name], step_size = self.model.step_size)

class CostsJR(AbstractLoss):
    def __init__(self, model):
        self.mainLoss = CostsTS("eeg")
        self.simKey = "eeg"
        self.model = model

    def loss(self, simData: dict, empData: torch.Tensor):

        sim = simData
        emp = empData

        model = self.model

        # define some constants
        lb = 0.001

        w_cost = 1
        w_prior = 3e-4

        # define the relu function
        m = torch.nn.ReLU()

        exclude_param = []
        if not model.use_fit_gains:
            exclude_param.append('gains_con')
            exclude_param.append('wll')

        if not model.use_fit_lfm:
            exclude_param.append('lm') 

        loss_main = self.mainLoss.loss(sim, emp)

        loss_EI = 0
        loss_prior = []

        variables_p = [a for a in dir(model.params) if (type(getattr(model.params, a)) == par)]

        for var_name in variables_p:
            var = getattr(model.params, var_name)
            if var.has_prior and var_name not in ['std_in'] and \
                        var_name not in exclude_param:
                loss_prior.append(torch.sum(((m(var.val) - m(var.prior_mean)) ** 2) * 1/(lb + m(var.prior_var))) \
                                  + torch.sum(-torch.log(lb + m(var.prior_var))))
        # total loss
        loss =  w_cost * loss_main + w_prior * sum(loss_prior) + 1 * loss_EI
        return loss

def prepare_static():
    mat_data = scipy.io.loadmat(tms_data_folder / 'Experiment_1.mat')
    layout_mat_data = scipy.io.loadmat(tms_data_folder / 'chlocs_nexstim.mat')

    n_channels = 60
    sfreq = 725
    channel_labels = ['Fp1', 'Fpz', 'Fp2', 'AF1', 'AFz', 'AF2', 
                      'F7', 'F5', 'F1', 'Fz', 'F2', 'F6', 'F8', 
                      'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 
                      'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T3', 
                      'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 
                      'T4', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 
                      'P9', 'P7', 'P3', 'P1', 'Pz', 'P2', 'P4', 
                      'P8', 'P10', 'PO3', 'POz', 'PO4', 'O1', 
                      'Oz', 'O2', 'Iz']
    ch_types = ["eeg"] * n_channels

    X, Y, Z = [np.ndarray(n_channels) for _ in range(3)]
    for i in range(n_channels):
        X[i] = -layout_mat_data['chlocs']['Y'][0][i][0] / 10
        Y[i] =  layout_mat_data['chlocs']['X'][0][i][0] / 10
        Z[i] =  layout_mat_data['chlocs']['Z'][0][i][0] / 10

    ch_loc = np.c_[X, Y, Z]
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(channel_labels, ch_loc)),
        coord_frame='head'
    )
    info = mne.create_info(ch_names=channel_labels, 
                           ch_types=ch_types, 
                           sfreq=sfreq)
    info.set_montage(montage)

    # --- Atlas / SC ---
    atlas = pd.read_csv(conn_data_folder / 'atlas_data.csv')
    coords = atlas[['R', 'A', 'S']].values

    dist = np.zeros((coords.shape[0], coords.shape[0]))
    dist = cdist(coords, coords)

    sc = pd.read_csv(
        conn_data_folder / 'Schaefer2018_200Parcels_7Networks_count.csv',
        header=None, sep=' '
    ).values
    sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

    return dict(
        mat_data=mat_data,
        info=info,
        dist=dist,
        sc=sc,
        sfreq=sfreq
    )

def main(static, n_sub: int = 1, 
         train_region: str = 'Premotor', 
         trial_fraction: float = 0.2):
    
    torch.manual_seed(42)
    np.random.seed(42)

    mat_data = static['mat_data']
    info = static['info']
    sc = static['sc']
    dist = static['dist']
    sfreq = static['sfreq']

    # ---- data extraction ----
    subject_data = mat_data['Experiment_1'][0,0][f'Subject_{n_sub}'][0,0]
    eeg_epochs = np.transpose(subject_data[train_region], (2,0,1))
    epoched = mne.EpochsArray(eeg_epochs, info)

    n_trials = len(epoched)
    train_trials = np.random.choice(n_trials, size=int(trial_fraction * n_trials), replace=False)
    test_trials = np.setdiff1d(np.arange(n_trials), train_trials)

    # === SAFE FULL-DATA MODE ===
    if len(test_trials) == 0:
        print("WARNING: trial_fraction=1 → using TRAIN as TEST")
        test_trials = train_trials.copy()

    evoked_train = epoched[train_trials].average()
    evoked_test  = epoched[test_trials].average()

    pre_pulse = 0.6
    ts_args = dict(xlim=[pre_pulse-0.1,pre_pulse+0.3])

    eeg_data = evoked_train.data.copy()

    time_start = np.where(abs(evoked_train.times-ts_args['xlim'][0])<=1e-3)[0][0]
    time_end = np.where(abs(evoked_train.times-ts_args['xlim'][1])<=1e-3)[0][0]
    eeg_data = eeg_data[:,time_start:time_end]/np.abs(eeg_data).max()*4

    st_file = tms_data_folder / ('stim_weights_' + train_region + '.npy')
    stim_weights = np.load(st_file)
    stim_weights_thr = stim_weights.copy()
    ki0 = stim_weights_thr[:,np.newaxis]

    # Load rest fit parameters estimate (averaged over last 10 epochs)
    fitted_params = np.load(rest_fitpar_folder / f"Sub{n_sub}_params.npz")

    a0 = fitted_params['a']
    omega0 = fitted_params['omega']
    sig_omega0 = fitted_params['sig_omega']
    g0 = fitted_params['g']
    std_in0 = fitted_params['std_in']
    mu0 = fitted_params['v_d']
    cy0 = fitted_params['cy0']
    wll0 = fitted_params['wll']
    lm0 = fitted_params['lm']

    conn_gains = {
        'wll': wll0
    }

    params = ParamsHP(a = par(a0), omega = par(omega0), sig_omega = par(sig_omega0), 
                      g = par(g0), std_in = par(std_in0), 
                      mu = par(mu0), k = par(2.5, fit_par=False, fit_hyper=False), 
                      cy0 = par(cy0), y0 = par(0), 
                      ki = par(ki0),
                      wll = par(wll0, wll0, 1.8*np.ones_like(wll0), fit_par=True, fit_hyper=False),
                      lm = par(lm0))

    # Simulation start
    start_time = time.time()
    node_size = sc.shape[0]
    output_size = eeg_data.shape[0]
    batch_size = 29
    step_size = 1e-4
    num_epoches = 200
    tr = 1/sfreq
    time_dim = eeg_data.shape[1]
    hidden_size = int(tr/step_size)

    # prepare data structure of the model
    data_mean = Recording(eeg_data, num_epoches, batch_size)

    # ==== TEST DATA ====
    eeg_data_test = evoked_test.data[:,time_start:time_end]
    eeg_data_test = eeg_data_test / np.abs(eeg_data_test).max() * 4
    data_test = Recording(eeg_data_test, num_epoches, batch_size)

    model = RNNHOPF(node_size, batch_size, step_size, output_size, tr, sc, 
                    lm0, dist, params, conn_gains, 
                    use_fit_gains=True, use_fit_lfm=False)

    # create objective function
    ObjFun = CostsJR(model)

    # call model fit
    F = Model_fitting(model, ObjFun)

    # model training
    u = np.zeros((node_size, hidden_size, time_dim))
    ts, te =  0.0965, 0.1103
    pulse_start = int(sfreq * ts)
    pulse_end = int(sfreq * te)
    u[:,:,pulse_start:pulse_end] = 1000
    F.train(u=u, empRecs = [data_mean], num_epochs = num_epoches, TPperWindow = batch_size)

    # ==== TRAIN SIMULATION ====
    F.evaluate(u=u, empRec=data_mean, TPperWindow=batch_size, base_window_num=20)
    sim_train = F.lastRec['eeg'].npTS().copy()

    # ==== TEST SIMULATION ====
    F.evaluate(u=u, empRec=data_test, TPperWindow=batch_size, base_window_num=20)
    sim_test = F.lastRec['eeg'].npTS().copy()

    results_dict = {
        'fitting_results': F,
        'TTrial_idxs': test_trials,
        'sim_train': sim_train,
        'sim_test': sim_test
    }

    # Save fitting results to a file using pickle
    results_folder = repo_root / 'Results_TEP'
    results_folder.mkdir(exist_ok=True)
    store_filename = results_folder / f'Subject{n_sub}_{train_region}_fitting_results.pkl'

    with open(store_filename, 'wb') as file:
        pickle.dump(results_dict, file)
    print("Results successfully saved to the file.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert elapsed time to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"Elapsed time: {minutes} minutes and {seconds} seconds")
    print(f'Finished processing Subject {n_sub}, Region {train_region}')

def exe_main():
    static = prepare_static()
    stim_sites = ['Premotor', 'Prefrontal']
    #stim_sites = ['Premotor']

    for site in stim_sites:
        for sub in range(1, 7):
            print(f'\n=== Subject {sub} | {site} ===')
            torch.cuda.empty_cache()
            main(
                static=static,
                n_sub=sub,
                train_region=site,
                trial_fraction=0.2
            )

if __name__ == "__main__":
    exe_main()
