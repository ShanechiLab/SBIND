



# Publication
This repository provides the implementation of SBIND (Spatiotemporal Behavior modeling in Imaging Neural Data), a deep learning framework for modeling raw neural imaging data.

Mohammad Hosseini and Maryam M. Shanechi. _Dynamical Modeling of Behaviorally Relevant Spatiotemporal Patterns in Neural Imaging Data_. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025.

# Usage Examples
The following notebook contains usage example for SBIND:

* `./tutorial.ipynb`

# Key Classes
The following are the key classes used to implement the SBIND model based on the formulation explained in the paper.


*   **`CONVSBIND`** (`./sbind/convsbind.py`): This is the main SBIND model class. It integrates the two ConvRNN modules (ConvRNN1 for behaviorally relevant dynamics and ConvRNN2 for other neural dynamics) and implements the full two-phase learning process described in the paper.


*   **`SBINDTrainer`** (`./sbind/sbind_trainer.py`): This class is a utility trainer that contains the functions to fit the SBIND model, generate predictions on new data, and run validation. It handles the training loops, optimization, and saving/loading of the model.



# License
Copyright (c) 2025 University of Southern California

Mohammad Hosseini and Maryam M. Shanechi

Shanechi Lab, University of Southern California
