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

Rscript src/Potts/Train.R $quick
find . -type f -name "network_epoch*" -exec rm {} +  # delete extraneous files
Rscript src/TrainingTime.R --model=Potts             # compute total runtime
julia --threads=auto --project=. src/Potts/AssessMissing.jl
Rscript src/Potts/Results.R
