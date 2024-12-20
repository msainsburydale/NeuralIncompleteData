#!/bin/bash
unset R_HOME

set -e

echo ""
echo "#### Starting example with univariate neural MAP estimator (Figures S9 and S10) ####"
echo ""

julia --threads=auto --project=. src/supplement/univariate_example/Experiment.jl 
Rscript src/supplement/univariate_example/Results.R
