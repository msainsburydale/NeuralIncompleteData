#!/bin/bash
unset R_HOME

set -e

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

Rscript src/application/sea_ice/Preprocessing.R
Rscript src/application/sea_ice/Train.R $quick 
julia --threads=auto --project=. src/application/sea_ice/Inference.jl $quick 
Rscript src/application/sea_ice/Results.R