#!/bin/bash
unset R_HOME

set -e

Rscript src/TrainingInfo.R

# Convert risk pdf to to png
find img -type f -iname "risk_profiles*.pdf" -exec sh -c '
  for f do
    out="${f%.pdf}.png"
    gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r600 -o "$out" "$f" >/dev/null 2>&1
  done
' sh {} +
