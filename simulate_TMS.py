import numpy as np
import pandas as pd
import torch
import json
import pickle
import warnings
from pathlib import Path
import os
import io

warnings.filterwarnings('ignore')

repo_root = Path(__file__).resolve().parent

from COVHOPF import *

results_folder = repo_root / 'Results_Neuromaps'

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_res(subject_num, region_key):
  scheds = {'O': 'OneCycleLR', 'R': 'ReduceLROnPlateau'}
  sched_type = scheds['R']
  loss_method = 'log_fro'
  #Load saved covariance fitting results using pickle
  store_filename1 = os.path.join(results_folder, 
                                f'Subject{subject_num}_{region_key}_{loss_method}_{sched_type}_fitting_results.pkl')
  with open(store_filename1, 'rb') as file:
    F = CPU_Unpickler(file).load()

  return F

class Recording:
    def __init__(self, data, step_size):
        self.data = data
        self.step_size = step_size

    def npTS(self):
        return self.data

def forward(
    external, hs, hEx, hEy, model_params, hyperparams, dist
):
    """
    Simulates one window using forward Euler integration for the Hopf model.

    Parameters
    ----------
    external : torch.Tensor
        External input for the window.
    hs : torch.Tensor
        Initial hidden state tensor.
    hEx, hEy : torch.Tensor
        Initial delay states.
    model_params : dict
        Dictionary containing model parameters (a, omega, g, etc.).
    hyperparams : dict
        Dictionary containing hyperparameters (node_size, output_size, etc.).

    Returns
    -------
    next_state : dict
        The updated state after one window.
    hEx : torch.Tensor
        Updated delay state for real part.
    hEy : torch.Tensor
        Updated delay state for imaginary part.
    """

    node_size = hyperparams['node_size']
    output_size = hyperparams['output_size']
    dt = torch.tensor(hyperparams['step_size'], dtype=torch.float32)
    dist = torch.tensor(dist, dtype=torch.float32)

    # Generate the ReLU module
    m = torch.nn.ReLU()

    # Define some constants
    con_1 = torch.tensor(1.0, dtype=torch.float32) # Define constant 1 tensor
    conduct_lb = 1.5  # lower bound for conduct velocity
    u_sys_ub = 500  # the bound of the input for the system
    noise_std_lb = 20  # lower bound of std of noise
    lb = 0.01  # lower bound of local gains
    k_lb = 0.5  # lower bound of coefficient of external inputs

    # Check if parameters are heterogeneous (i.e., given as vectors)
    heterogeneous = 'a' in model_params and model_params['a'].shape[0] > 1

    a = 0 * con_1 - m(-model_params['a'].detach())
    mean_omega = model_params['mean_omega'].detach()
    sig_omega_raw = model_params['sig_omega'].detach()
    sig_omega = torch.nn.functional.softplus(sig_omega_raw)
    omega = (0 * con_1 + m(torch.normal(mean=mean_omega, std=sig_omega, size=(node_size,))))
    g = (lb * con_1 + m(model_params['g'].detach()))
    std_in = (noise_std_lb * con_1 + m(model_params['std_in'].detach()))
    v_d = (conduct_lb * con_1 + m(model_params['v_d'].detach()))
    k = (k_lb * con_1 + m(model_params['k']))
    cy0 = model_params['cy0'].detach()
    ki = model_params['ki']
    if isinstance(model_params['lm'], torch.nn.Parameter):
      lm = model_params['lm'].detach()
    else:
      lm = torch.tensor(model_params['lm'], dtype = torch.float32)

    # Initialize state components
    X = hs[:, 0:1]
    Y = hs[:, 1:2]

    # Lateral Weights
    w_n_l = torch.tensor(model_params['sc_fitted'], dtype=torch.float32)
    dg_l = -torch.diag(torch.sum(w_n_l, dim=1))
    
    # Calculate delays
    delays = (dist / v_d).type(torch.int64)

    # Placeholder for the updated current state
    current_state = torch.zeros_like(hs)

    eeg_window = []
    X_window = []
    Y_window = []

    for i_window in range(hyperparams['TRs_per_window']):
        for step_i in range(hyperparams['steps_per_TR']):
          # Compute delayed interactions
          Edx = torch.tensor(np.zeros((node_size, node_size)), dtype=torch.float32)
          Edy = torch.tensor(np.zeros((node_size, node_size)), dtype=torch.float32)
          hEx_new = hEx.clone()
          hEy_new = hEy.clone()

          # Gather delayed states
          Edx = hEx_new.gather(1, delays)
          Edy = hEy_new.gather(1, delays)

          # Lateral component
          LEd_x_l = torch.reshape(torch.sum(w_n_l * torch.transpose(Edx, 0, 1), 1), (node_size, 1))
          LEd_y_l = torch.reshape(torch.sum(w_n_l * torch.transpose(Edy, 0, 1), 1), (node_size, 1))

          # External input
          u_tms = external[:, step_i:step_i + 1, i_window]

          # Sum the components, scaled by their respective gains
          rX = (
              k * ki * u_tms +
              std_in * torch.randn(node_size, 1) + g * (
              (LEd_x_l + torch.matmul(dg_l, X))
              ))

          rY = (
              std_in * torch.randn(node_size, 1) + g * (
              (LEd_y_l + torch.matmul(dg_l, Y))
              ))

          # Update state dynamics with the new rX and rY that incorporate all connections
          if heterogeneous:
              dX = X + dt * ((a.view(-1,1) - X**2 - Y**2) * X - omega.view(-1, 1) * Y + u_sys_ub * torch.tanh(rX / u_sys_ub))
              dY = Y + dt * ((a.view(-1,1) - X**2 - Y**2) * Y + omega.view(-1, 1) * X + u_sys_ub * torch.tanh(rY / u_sys_ub))
          else:
              dX = X + dt * ((a.view(-1,1) - X**2 - Y**2) * X - omega.view(-1, 1) * Y + u_sys_ub * torch.tanh(rX / u_sys_ub))
              dY = Y + dt * ((a.view(-1,1) - X**2 - Y**2) * Y + omega.view(-1, 1) * X + u_sys_ub * torch.tanh(rY / u_sys_ub))

          # Saturation to keep values within range
          X = 1000 * torch.tanh(dX / 1000)
          Y = 1000 * torch.tanh(dY / 1000)

          # Update history buffers
          hEx[:, 0] = X[:, 0]
          hEy[:, 0] = Y[:, 0]

        X_window.append(X)
        Y_window.append(Y)

        hEx = torch.cat([X, hEx[:, :-1]], dim=1)
        hEy = torch.cat([Y, hEy[:, :-1]], dim=1)

        # EEG computation
        lm_t = (lm.T / torch.sqrt(lm ** 2).sum(1)).T
        lm_t = (lm_t - 1 / output_size * torch.matmul(torch.ones((1,output_size)), lm_t))
        temp = cy0 * torch.matmul(lm_t, X[:200, :])
        eeg_window.append(temp)

    next_state = {
        'current_state': torch.cat([X, Y], dim=1),
        'eeg': torch.cat(eeg_window, dim=1),
        'X': torch.cat(X_window, dim=1),
        'Y': torch.cat(Y_window, dim=1)
    }

    return next_state, hEx, hEy

def simulate(
    u, numTP, model_params, hyperparams, sc, dist,
    base_window_num=0, transient_num=10, initial_S = None,
    lin=False
):
    """
    Simulates the system using parameters directly.

    Parameters
    ----------
    u : np.array
        External input or stimulus.
    numTP : int
        The number of time points to simulate.
    model_params : dict
        Dictionary containing model parameters (a, omega, g, etc.).
    hyperparams : dict
        Dictionary containing hyperparameters (node_size, output_size, etc.).
    sc : np.array
        Structural connectivity matrix.
    dist : np.array
        Distance matrix between nodes.
    lm : np.array
        Leadfield matrix.
    base_window_num : int, optional
        Length of resting window, by default 0.
    transient_num : int, optional
        The number of initial time points to exclude from metrics, by default 10.
    initial_S : torch.Tensor, optional
        Custom initial state tensor. If None, random values will be used.

    Returns
    -------
    ts_sim : np.array
        Simulated time series for the EEG signal.
    fc_sim : np.array
        Functional connectivity matrix from the simulated time series.
    lastRec : dict
        Dictionary containing the last recorded EEG signal.
    """

    node_size = hyperparams['node_size']
    state_size = hyperparams['state_size']
    output_size = hyperparams['output_size']
    steps_per_TR = hyperparams['steps_per_TR']
    TRs_per_window = hyperparams['TRs_per_window']
    step_size = hyperparams['step_size']

    num_windows = int(numTP / TRs_per_window)

    # Initialize lists to store the results for each window
    windListDict = {
        'X': [], 'Y': [], 'eeg': []
    }

    lastRec = {}

    # Initialize states
    S = torch.tensor(initial_S, dtype=torch.float32) if initial_S is not None else torch.tensor(
        np.random.uniform(-0.5, 0.5, (node_size, state_size)), dtype=torch.float32)
    hEx = torch.tensor(np.random.uniform(-0.5, 0.5, (node_size, 500)), dtype=torch.float32)
    hEy = torch.tensor(np.random.uniform(-0.5, 0.5, (node_size, 500)), dtype=torch.float32)

    u_hat = np.zeros((node_size, steps_per_TR,
                      base_window_num * TRs_per_window + num_windows * TRs_per_window))
    u_hat[:, :, base_window_num * TRs_per_window:] = u

    # Iterate over windows
    for win_idx in range(num_windows + base_window_num):
        # External input for the current window
        external = torch.tensor(u_hat[:, :, win_idx * TRs_per_window:(win_idx + 1) * TRs_per_window], dtype=torch.float32)

        next_window, hEx_new, hEy_new = forward(
            external, S, hEx, hEy, model_params, hyperparams, sc, dist
        )

        # Store the results (only for windows beyond the resting period)
        if win_idx > base_window_num - 1:
            windListDict['X'].append(next_window['X'].detach().cpu().numpy())
            windListDict['Y'].append(next_window['Y'].detach().cpu().numpy())
            windListDict['eeg'].append(next_window['eeg'].detach().cpu().numpy())

        # Update the current state
        S = next_window['current_state'].detach().clone()
        hEx = hEx_new.detach().clone()
        hEy = hEy_new.detach().clone()

    # Concatenate all windows into one continuous time series
    for key in windListDict:
        windListDict[key] = np.concatenate(windListDict[key], axis=1)

    # Time series and functional connectivity
    ts_sim = windListDict['eeg']
    fc_sim = np.corrcoef(ts_sim[:, transient_num:])

    # Store only the EEG data in lastRec
    lastRec['eeg'] = Recording(ts_sim, step_size=hyperparams['step_size'])
    lastRec['X'] = Recording(windListDict['X'],step_size=hyperparams['step_size'])

    return ts_sim, fc_sim, lastRec

# =========================================================================== #
#                          Main Simulation Script                             #
# =========================================================================== #

# Set up paths
struct_data_folder = repo_root / 'struct_data'
conn_data_folder = repo_root / 'conn_data'
data_folder = repo_root / 'cov_data'
neuromaps_folder = repo_root / 'neuromaps'
eegtms_data_folder = repo_root / 'eegtms_data'

ch_file = eegtms_data_folder / 'chlocs_nexstim.mat'
region = ['Prefrontal', 'Premotor']

# Select subject and region
n_sub = 1
region_key = region[1]

# Channel labels and sampling frequency
with open("metadata.json", "r") as f:
    metadata = json.load(f)

sampling_freq = metadata[f"SUB{n_sub}"]["sampling_freq"]
channel_labels = metadata[f"SUB{n_sub}"]["channel_labels"]
n_channels = len(channel_labels) 

# Load Connectivity Matrix
atlas = pd.read_csv(conn_data_folder / 'atlas_data.csv')
labels = atlas['ROI Name']
coords = np.array([atlas['R'], atlas['A'], atlas['S']]).T

dist = np.zeros((coords.shape[0], coords.shape[0]))

for roi1 in range(coords.shape[0]):
  for roi2 in range(coords.shape[0]):
    dist[roi1, roi2] = np.sqrt(np.sum((coords[roi1,:] - coords[roi2,:])**2, axis=0))
    dist[roi1, roi2] = np.sqrt(np.sum((coords[roi1,:] - coords[roi2,:])**2, axis=0))

sc_file = conn_data_folder / 'Schaefer2018_200Parcels_7Networks_count.csv'
sc_df = pd.read_csv(sc_file, header=None, sep=' ')
sc = sc_df.values
sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

#Load TMS pulse
st_file = eegtms_data_folder / f'stim_weights_{region_key}.npy'
stim_weights = np.load(st_file)
stim_weights_thr = stim_weights.copy()
labels[np.where(stim_weights_thr>0)[0]]
ki0 =stim_weights_thr[:,np.newaxis]

# Load fitted parameters
F = load_res(subject_num=n_sub, region_key=region_key)

# Define parameter cases
homogeneous_params = ['a', 'omega', 'sig_omega', 'v_d', 'g', 'std_in', 'cy0']
heterogeneous_params = ['a']

all_param_cov = {}

# Get all parameters that exist in fit_params (regardless of heterogeneity)
available_params = [p for p in homogeneous_params if p in F.trainingStats.fit_params]

# Use all available parameters
param_keys = available_params

# Find the maximum length needed for padding
max_length = max(len(np.array(F.trainingStats.fit_params[key][-1:]).flatten()) for key in param_keys)

for key in param_keys:
    last_values = np.array(F.trainingStats.fit_params[key][-1:]).flatten()
    
    # Pad the values to max_length
    if last_values.shape[0] == 1:
        last_values = last_values.tolist()
    padded_values = np.pad(last_values, (0, max_length - len(last_values)), constant_values=np.nan)
    all_param_cov[key] = padded_values

# Convert to DataFrame
pd.set_option('display.max_rows', None)
df = pd.DataFrame(all_param_cov)

lm_1 = F.model.lm.detach().numpy()
ec_1 = F.model.sc_fitted.detach().numpy()

sc_fitted = ec_1

pre_pulse = 0.6
step_size = 1e-4
ts_args = dict(xlim=[pre_pulse-0.1,pre_pulse+0.3])
time_start = 362
time_end = 652
time_dim = 290

n_nodes = sc.shape[0]
n_ch = len(channel_labels)
tr = 1/sampling_freq
hidden_size = int(tr/step_size)

numTP = time_dim

k0_dict = np.load(eegtms_data_folder / 'k_values.npy', allow_pickle=True).item()
k0 = k0_dict[region_key][n_sub - 1]

model_params = {
    'a': torch.tensor(all_param_cov['a'], dtype=torch.float32),
    'g': torch.tensor(all_param_cov['g'][0], dtype=torch.float32),
    'std_in': torch.tensor(all_param_cov['std_in'][0], dtype=torch.float32),
    'v_d': torch.tensor(all_param_cov['v_d'][0], dtype=torch.float32),
    'k': torch.tensor(k0, dtype=torch.float32),
    'cy0': torch.tensor(all_param_cov['cy0'][0], dtype=torch.float32),
    'ki': torch.tensor(ki0, dtype=torch.float32),
    'sc_fitted': sc_fitted,
    'lm': lm_1
}


model_params['mean_omega'] = torch.tensor(all_param_cov['omega'][0], dtype=torch.float32)
model_params['sig_omega'] = torch.tensor(all_param_cov['sig_omega'][0], dtype=torch.float32)


# Hyperparameters
hyperparams = {
    'node_size': n_nodes,
    'output_size': n_ch,
    'state_size': 2,
    'steps_per_TR': hidden_size,
    'TRs_per_window': 29,
    'step_size': step_size
}

u = np.zeros((n_nodes, hidden_size, time_dim))

ts, te =  0.0965, 0.1103
pulse_start = int(sampling_freq * ts)
pulse_end = int(sampling_freq * te)
u[:,:,pulse_start:pulse_end]= 1000

# Simulate
ts_sim, fc_sim, lastRec = simulate(u, numTP, model_params, hyperparams, sc, dist, base_window_num = 20)