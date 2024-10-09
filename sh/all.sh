#!/bin/bash
unset R_HOME

set -e

echo ""
echo "##### Setting up #####"
echo ""

echo "Do you wish to use a very low number of parameter configurations and epochs to quickly establish that the code is working? (y/n)"
read quick_str

if ! [[ $quick_str == "y" ||  $quick_str == "Y" || $quick_str == "n" ||  $quick_str == "N" ]]; then
    echo "Please re-run and type y or n"
    exit 1
fi

source sh/simulations_Potts.sh   # Section 3.3
source sh/application.sh         # Section 4
source sh/simulations_GP.sh      # Section 3.2 

# Remove intermediate networks to reduce the folder’s memory size
find . -type f -name "network_epoch*" -exec rm {} +

echo ""
echo "##### Finished! #####"
echo ""
