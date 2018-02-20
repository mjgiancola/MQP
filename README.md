Repository of code to supplement the submission of "Permutation-Invariant Consensus over Crowdsourced Labels" by Anonymous to HCOMP 2018.

## Installation
The dependency list for our model is in requirements.txt. If you are on a system with pip, the easiest way to install the dependencies is by running "sudo pip install -r requirements.txt"

## Running Simulations
To run the standard simulation, execute "python simulation.py ..." within the Simulation/ directory.
To run the image segmentation experiments, execute "python runsegmentation.py ..." within the Simulation/ directory.

## Directory Listing

### Simulation
Contains all files related to running our model, including scripts, the model implementation, and test data and results.
### Data Generator
Contains code for generating simulation data
### ADE20K
Contains scripts and data from the ADE20K project that were used in simulations.
### Archive
Graveyard of old code/results
