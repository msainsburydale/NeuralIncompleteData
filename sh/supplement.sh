#!/bin/bash
unset R_HOME
set -e

echo ""
echo "#### Starting experiment on ensembles of NBEs (Section S5) ####"
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

Rscript src/supplement/Ensembles.R $quick

echo ""
echo "#### Starting experiment on neural MAP estimation with univariate data (response) ####"
echo ""

julia --threads=auto --project=. src/supplement/univariate_example/Experiment.jl 
Rscript src/supplement/univariate_example/Results.R

# Convert pdf images to png
find img/Ensemble/ -type f -iname "*.pdf" -exec sh -c '
  for f do
    out="${f%.pdf}.png"
    gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r600 -o "$out" "$f" >/dev/null 2>&1
  done
' sh {} +
