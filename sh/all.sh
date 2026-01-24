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

echo ""
echo "#### Starting main-text simulation studies (Section 3) ####"
echo ""
source sh/simulations_GP.sh             # Section 3.2 
source sh/simulations_HiddenPotts.sh    # Section 3.3

# source sh/application_sea_ice.sh        # Section 4

echo ""
echo "#### Starting supplementary experiments ####"
echo ""
source sh/simulations_GH.sh             # Section S4
source sh/training_info.sh              # Training run time and risk profiles
source sh/supplement.sh                 # Other supplementary results

# Remove intermediate networks to reduce the folder’s memory size
find . -type f -name "network_epoch*" -exec rm {} +

echo ""
echo "##### Finished! #####"
echo ""