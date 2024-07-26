#!/bin/bash

# ex. ./run.sh test_ens.py

inpt=$@
name=${inpt%.*}

mkdir -p runs
source venv/bin/activate
nohup python3 -u $inpt > runs/$name.out 2> runs/$name.err < /dev/null &
