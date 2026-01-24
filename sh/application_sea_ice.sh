#!/bin/bash
unset R_HOME
set -e

echo ""
echo "#### Starting application study (Section 4) ####"
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

# Run scripts
Rscript src/application/sea_ice/Preprocessing.R
Rscript src/application/sea_ice/Train.R $quick --domain=full
Rscript src/application/sea_ice/Train.R $quick --domain=sub
julia --project=. --threads=auto src/application/sea_ice/Inference.jl $quick --domain=sub
julia --project=. --threads=64 src/application/sea_ice/Inference.jl $quick --domain=full
Rscript src/application/sea_ice/Results.R

# Delete extraneous files
find . -type f -name "network_epoch*" -exec rm {} +  

# Convert pdf images to png, excluding raw_data directory
find img/application/sea_ice/ -type f -iname "*.pdf" -not -path "*/raw_data/*" -exec sh -c '
  for f do
    out="${f%.pdf}.png"
    gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r600 -o "$out" "$f" >/dev/null 2>&1
  done
' sh {} +

# Convert only the sea_ice_1993.pdf file in raw_data directory
gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r600 \
   -o "img/application/sea_ice/raw_data/sea_ice_1993.png" \
   "img/application/sea_ice/raw_data/sea_ice_1993.pdf"