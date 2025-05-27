# SBIND: Spatiotemporal Behavior modeling in Imaging Neural Data

This repository provides the implementation of SBIND (Spatiotemporal Behavior modeling in Imaging Neural Data)
## Publication

This work is currently under review. The original work that this repository implements is:
_Dynamical Modeling of Behaviorally Relevant Spatiotemporal Patterns in Neural Imaging Data_


## Usage Examples

The following notebook contains usage examples of SBIND for several use-cases:

*   [source/tutorial.ipynb](./tutorial.ipynb)


## Key Classes

The following are the key classes used to implement the SBIND formulation as explained in the preprint:

*   **`CONVSBIND`** (`./sbind/convsbind.py`): This class implements the full ConvRNN model.

*   **`SBINDTrainer`** (`./sbind/sbind_trainer.py`): Contains the trainer to fit, predict, and validation functions.
