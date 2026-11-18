#!/bin/bash
unset R_HOME
set -e

# Set BLAS threading limits before calling R
# export OPENBLAS_NUM_THREADS=1

echo ""
echo "#### Starting hidden-Potts simulation study (Section 3.3) ####"
echo ""

if [[ -z "${quick_str+x}" ]]; then
  echo "Do you wish to use a very low number of parameter configurations and epochs to quickly establish that the code is working? (y/n)"
  read quick_str
fi

if [[ $quick_str == "y" ||  $quick_str == "Y" ]]; then
    quick=--quick
elif [[ $quick_str == "n" ||  $quick_str == "N" ]]; then
    quick=""
else
    echo "Please re-run and type y or n"
    exit 1
fi

if [[ -z "${TRAIN+x}" ]]; then
  echo "Do you wish to retrain the neural networks (y), or use the trained networks included in the repository (n)? (y/n)"
  read TRAIN
fi

if ! [[ $TRAIN == "y" || $TRAIN == "Y" || $TRAIN == "n" || $TRAIN == "N" ]]; then
    echo "Please re-run and type y or n"
    exit 1
fi

# Simulate the training, validation, test and scenario data (Julia, HiddenMarkovRandomFields.jl)
julia --project=. --threads=auto src/HiddenPotts/SimulateData.jl $quick

# Train the NBEs, unless the trained networks included in the repository are used
if [[ $TRAIN == "y" || $TRAIN == "Y" ]]; then
  julia --project=. --threads=auto src/HiddenPotts/Train.jl $quick
fi

# ABC, with the training data as its reference table (R)
OPENBLAS_NUM_THREADS=1 Rscript src/HiddenPotts/ABC.R

# Assess with missing data 
julia --project=. --threads=auto src/HiddenPotts/Experiment.jl

# Generate results
Rscript src/HiddenPotts/Results.R

# Delete extraneous files
find . -type f -name "network_epoch*" -exec rm {} +  

# Convert pdf images to png
find img/HiddenPotts/ -type f -iname "*.pdf" -exec sh -c '
  for f do
    out="${f%.pdf}.png"
    gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r600 -o "$out" "$f" >/dev/null 2>&1
  done
' sh {} +

