# Modelling Response to Brain Stimulation from Resting-State Activity

## Project Overview

This project focuses on fitting the covariance matrix of resting-state activity using TMS-EEG data. The optimized parameters obtained from this fit can be further used to model and predict brain network responses to external stimulation, although this repository primarily implements the covariance matrix fitting process.

## Repository Structure

The repository is organized into the following directories:

### 1. `struct_data`

Contains structural information used to define effective connectivity, including:

- **Leadfield matrix**: Maps sources to scalp-level EEG measurements.
- **Connection gain matrix**: Represents effective connectivity between brain regions.

### 2. `conn_data`

Holds parcellation, structural, and functional connectivity data:

- **Parcellation**: Brain region definitions for the analysis.
- **Structural connectivity (SC)**: Data describing physical connections between regions.

### 3. `eegtms_data`

Includes information on TMS stimulation and EEG channel positions:

- **Scalp channel position file**: Contains electrode positions.
- **Stimulation reconstruction**: Describes the applied TMS stimulation patterns.

### 4. `cov_data`

Contains the resting-state covariance matrices for each subject. These matrices are used to fit the model to the observed brain activity.

### 5. `neuromaps`

Holds a set of neurotransmitter receptor maps that encode regional heterogeneity in the model, allowing the simulation to account for differences in local neurochemistry.


### Code

- `cov_main.py`: The main script for:
  - Fitting the covariance matrix of resting-state activity.
  - Saving the optimized parameters for further analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/veronean/TMS-EEG-code.git
   cd TMS-EEG-code
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
`requirements.txt`:

1. The dependencies required for the project are:
    ```bash
    gdown
    scipy
    pandas
    torch
    sklearn
    mne
    numpy
    ```

## Usage

1. Prepare input data:
   - Ensure that structural, connection, and TMS data are in their respective directories.
2. Run the main script:
   ```bash
   python cov_main.py
   ```
3. Outputs:
   - Optimized parameters for the neural mass model.

## Code Structure

The `cov_main.py` script performs the following steps:

1. **Load Data**: Reads structural and functional connectivity data, as well as TMS-related files.
2. **Compute Covariance**: Fits the covariance matrix using resting-state activity data.
3. **Optimize Parameters**: Uses optimization techniques to fit the model to the covariance matrix.
4. **Save Results**: Stores the optimized parameters for further use in simulations.

## Technologies Used

- **Python**: Core language for modeling and simulation.
- **NumPy, SciPy**: Mathematical and statistical computations.
- **Matplotlib**: Visualization of results.
- **MNE-Python**: EEG/TMS data processing.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description of your changes.