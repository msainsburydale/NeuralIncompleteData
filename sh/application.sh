#!/bin/bash
unset R_HOME

set -e

Rscript src/application/sea_ice/Preprocessing.R
Rscript src/application/sea_ice/Train.R $quick
julia --threads=auto --project=. src/application/sea_ice/Inference.jl
Rscript src/application/sea_ice/Results.R