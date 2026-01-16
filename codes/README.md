## Directory with the codes

This directory contains codes implementing Model Agnostic Meta Learning MAML and its attentive variant (Att-MAML).

created in 2026 by Karol Baran
karol.baran[at]pg.edu.pl

## Files and data

- run_experiments.py - main file that should be run to reproduce the experiments with proper configuration and hyperparameters
- attmaml.py - implementation of Att-MAML
- learners.py - file containing architecture of neural network
- utils_codes.py - file with codes mostly regarding preparing and pre-processing data
- envdep.yaml - file with information on dependencies

## Dependencies

- deepchem
- torch
- rdkit
- lightgbm
- numpy
- sklearn
- matplotlib
- pandas

The detailed list available in envdep.yaml