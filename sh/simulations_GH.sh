#!/bin/bash
unset R_HOME
set -e

echo ""
echo "#### Starting GH simulation study (Section S4) ####"
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

# Train the NBEs (this also computes the reference table of ABC), unless the trained networks included in the
# repository are used
if [[ $TRAIN == "y" || $TRAIN == "Y" ]]; then
  julia --threads=auto --project=. src/GH/Train.jl $quick
fi

# Assess the estimators with missing data
julia --threads=auto --project=. src/GH/Experiment.jl $quick

# ABC, whose reference table (the summary statistics of the training data) is computed by src/GH/Train.jl
if [[ -f intermediates/GH/ABC/sumstats_train.csv ]]; then
  OPENBLAS_NUM_THREADS=1 Rscript src/GH/ABC.R
else
  echo "Skipping ABC: its reference table (intermediates/GH/ABC/sumstats_train.csv) is computed by src/GH/Train.jl, which runs when the networks are retrained"
fi

# Generate results
Rscript src/GH/Results.R

# Delete extraneous files
find . -type f -name "network_epoch*" -exec rm {} +  

# Convert pdf images to png
find img/GH/ -type f -iname "*.pdf" -exec sh -c '
  for f do
    out="${f%.pdf}.png"
    gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r600 -o "$out" "$f" >/dev/null 2>&1
  done
' sh {} +

