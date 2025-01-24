# Modelling Response to Brain Stimulation from Resting-State Activity

## Project Overview

This project focuses on fitting the covariance matrix of resting-state activity using TMS-EEG data. The optimized parameters obtained from this fit can be further used to model and predict brain network responses to external stimulation, although this repository primarily implements the covariance matrix fitting process.

## Repository Structure

The repository is organized into the following directories:

### 1. `structural_data`

Contains structural information used to define effective connectivity, including:

- **Leadfield matrix**: Maps sources to scalp-level EEG measurements.
- **Connection gain matrix**: Represents effective connectivity between brain regions.

### 2. `connection_data`

Holds parcellation, structural, and functional connectivity data:

- **Parcellation**: Brain region definitions for the analysis.
- **Structural connectivity (SC)**: Data describing physical connections between regions.
- **Functional connectivity (FC)**: Correlation-based connectivity derived from resting-state activity.

### 3. `tms_data`

Includes information on TMS stimulation and EEG channel positions:

- **Scalp channel position file**: Contains electrode positions.
- **Stimulation reconstruction**: Describes the applied TMS stimulation patterns.

### Code

- `cov_main.py`: The main script for:
  - Fitting the covariance matrix of resting-state activity.
  - Saving the optimized parameters for further analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/brain-stimulation-model.git
   cd brain-stimulation-model
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

## Contributors

- **Andrea Veronese**: Lead developer and researcher.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description of your changes.

## Acknowledgments

Special thanks to collaborators and advisors, including Michele Allegra, Samir Suweis, Davide Momi, Simone Sarasso, and Maurizio Corbetta, for their insights and guidance.

