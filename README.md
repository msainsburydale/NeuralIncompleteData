# Source code for "Neural Parameter Estimation with Incomplete Data"

![Figure 2: The inference stage of the neural EM algorithm.](/img/schematic.png?raw=true)

> _**Figure 2**: The inference stage of the neural EM algorithm. Data with missing entries are completed by simulating conditionally on the incomplete data and the current estimates. The conditionally-independent replicates are then input to a neural Bayes estimator trained to approximate the MAP estimator, with the resulting estimates used for conditional simulation in the next iteration of the algorithm._

This repository contains the source code for reproducing the results in "Neural Parameter Estimation with Incomplete Data" (Sainsbury-Dale, Zammit-Mangion, Cressie, Huser, 2024+).

The methodology described in the manuscript has been incorporated into the Julia package [NeuralEstimators.jl](https://github.com/msainsburydale/NeuralEstimators.jl). In particular, see the example given [here](https://msainsburydale.github.io/NeuralEstimators.jl/dev/workflow/advancedusage/#Missing-data). The code in this repository is therefore made available primarily for reproducibility purposes, and we encourage readers seeking to implement this methodology to explore the package and its documentation. Users are also invited to contact the package maintainer for assistance. 

## Instructions

First, download this repository and navigate to its top-level directory within the command line (i.e., `cd` to wherever you installed the repository).

### Software dependencies

Before installing the software dependencies, users may wish to set up a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) environment, so that the dependencies of this repository do not affect the user's current installation. To create a conda environment, run the following command in terminal:

```
conda create -n NeuralIncompleteData -c conda-forge julia=1.10.1 r-base nlopt
```

Then activate the conda environment with:

```
conda activate NeuralIncompleteData
```

The above conda environment installs Julia and R automatically. If you do not wish to use a conda environment, you will need to install Julia and R manually if they are not already on your system:  

- Install [Julia 1.10.1](https://julialang.org/downloads/).
- Install [R >= 4.0.0](https://www.r-project.org/).

Once Julia and R are setup, install the Julia and R package dependencies (given in `Project.toml` and `Manifest.toml`, and `dependencies.txt`, respectively) by running the following commands from the top-level of the repository:

```
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```
```
Rscript dependencies_install.R
```

### Hardware requirements

The fast construction of neural Bayes estimators requires graphical processing units (GPUs). Hence, although the code in this repository will run on the CPU, we recommend that the user run this code on a workstation with a GPU. Note that running the "quick" version of the code (see below) is still fast even on the CPU.

### Reproducing the results

The replication script is `sh/all.sh`, invoked using `bash sh/all.sh` from the top level of this repository. For all studies, the replication script will automatically train the neural estimators, generate estimates from both the neural and likelihood-based estimators, and populate the `img` folder with the figures and results of the manuscript.

The nature of our experiments means that the run time for reproducing the results of the manuscript can be substantial (up to a day of computing time, depending on the computational resources available to the user). When running the replication script, the user will be prompted with an option to quickly establish that the code is working by training the neural networks with a small data set and a small number of epochs. Our envisioned workflow is to establish that the code is working with this "quick" option, clear the populated folders by entering `bash sh/clear.sh`, and then run the code in full (possibly over the weekend). **NB:** under this "quick" option, very few training samples and epochs are used when training the neural Bayes estimators, and the results produced will therefore not be meaningful and should not be interpreted.  

Note that the replication script is clearly presented and commented; hence, one may easily "comment out" sections to produce a subset of the results. (Comments in `.sh` files are made with `#`.)


#### Minor reproducibility difficulties

When training neural networks on the GPU, there is some unavoidable non-determinism: see [here](https://discourse.julialang.org/t/flux-reproducibility-of-gpu-experiments/62092). This does not significantly affect the "story" of the final results, but there may be some slight differences each time the code is executed.
