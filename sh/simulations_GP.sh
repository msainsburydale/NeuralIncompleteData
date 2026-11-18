#!/bin/bash
unset R_HOME
set -e

# Set BLAS threading limits before calling R
# export OPENBLAS_NUM_THREADS=1

echo ""
echo "#### Starting GP simulation study (Section 3.2) ####"
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

train=""
if [[ $TRAIN == "y" || $TRAIN == "Y" ]]; then train=--train; fi # otherwise, the trained networks included in the repository are used

# R implementation (GpGp); this also saves the incomplete data used by the Julia implementation
OPENBLAS_NUM_THREADS=1 Rscript src/GP/Experiment.R $quick $train
Rscript src/GP/Results.R

# Julia implementation (Vecchia.jl), with the same networks and data
julia --project=. --threads=32 src/GP/Experiment.jl $quick
Rscript src/GP/Results.R --julia

find . -type f -name "network_epoch*" -exec rm {} +  # delete extraneous files

# Convert pdf images to png
find img/GP/ -type f -iname "*.pdf" -exec sh -c '
  for f do
    out="${f%.pdf}.png"
    gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r600 -o "$out" "$f" >/dev/null 2>&1
  done
' sh {} +