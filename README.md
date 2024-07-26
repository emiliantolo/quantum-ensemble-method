# Quantum Ensemble Method

This repository contains the code for the paper "A quantum ensemble method for quantum binary classifiers".

Datasets and code are based on: https://github.com/emiliantolo/ensembles-quantum-classifiers.

## Project structure

- classification/
    - classifiers/ - classical and quantum classifiers
    - classifiers/ens_weight_quantum_* - quantum implementation of the weighted ensemble
    - classifiers/ens_weight_classical_* - classical simulation of the weighted ensemble
- data/
    - dataset/ - datasets
    - folds/ - data folding used for tests
- test_ens.py - example testing script

## Run

### Setup environment

Tested on EndeavourOS, Python 3.10.

#### Get code
    git clone https://github.com/emiliantolo/quantum-ensemble-method.git
    cd quantum-ensemble-method

#### Install dependencies
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### Run experiments

    python test_ens.py --data 0 --fold 0 

Optional arguments are ```--data (-d)```, ```--fold (-f)```, and ```--classifier (-c)```, with the slicing or indexing of the related lists defined in the script, with format: ```(-)?[0-9]*(:)?(-)?[0-9]*```.

    python test_ens.py          # run all experiments 
    python test_ens.py -d 1:-1  # skip the first and last datasets
    python test_ens.py -f :3    # run the first 3 folds
    python test_ens.py -c 0     # run only the first classifier
