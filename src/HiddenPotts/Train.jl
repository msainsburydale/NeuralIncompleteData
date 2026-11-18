# Hidden Potts model (Section 3.3): training of the EM NBE and the Masking NBE with the training and validation data
# simulated by src/HiddenPotts/SimulateData.jl. The estimators are assessed by src/HiddenPotts/Experiment.jl.
#
# Usage (from the top level of the repository, after src/HiddenPotts/SimulateData.jl):
#   julia --project=. --threads=auto src/HiddenPotts/Train.jl [--quick]

quick = "--quick" in ARGS

using NeuralEstimators
using NeuralEstimators: estimate
using Flux
using CUDA, cuDNN # cuDNN: Flux convolutions on NVIDIA GPUs
using BSON: @save
using CSV
using DelimitedFiles
using Distributions: Uniform
using HDF5
using LinearAlgebra: BLAS
using Random
Threads.nthreads() > 1 && BLAS.set_num_threads(1) # see src/HiddenPotts/Experiment.jl

include(joinpath(pwd(), "src", "Architecture.jl"))
include(joinpath(pwd(), "src", "HiddenPotts", "Simulation.jl"))
int_path = joinpath("intermediates", "HiddenPotts")

Random.seed!(1) # the initial weights of the networks, and the missing data of the Masking NBE's training data

# ---- Data simulated by src/HiddenPotts/SimulateData.jl ----

prior_lower_bound = vec(readdlm(joinpath(int_path, "prior_lower_bound.csv"), ','))
prior_upper_bound = vec(readdlm(joinpath(int_path, "prior_upper_bound.csv"), ','))
θ_train, Z_train, θ_val, Z_val = h5open(joinpath(int_path, "training_data.h5")) do f
    read(f, "theta_train"), as_datasets(read(f, "Z_train")), read(f, "theta_val"), as_datasets(read(f, "Z_val"))
end
θ_test = readdlm(joinpath(int_path, "theta_test.csv"), ',')
Z_test = as_datasets(h5read(joinpath(int_path, "test_data.h5"), "Z_test"))

# ---- Training ----

epochs = quick ? 3 : 100
savenetwork(estimator, folder) = (model_state = Flux.state(Flux.cpu(estimator)); @save joinpath(int_path, folder, "best_estimator.bson") model_state)
d = size(θ_train, 1)

@info "Training the EM NBE..."
neuralMAP = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 1)
neuralMAP = train(neuralMAP, θ_train, θ_val, Z_train, Z_val; epochs = epochs, savepath = joinpath(int_path, "runs_EM"))
savenetwork(neuralMAP, "runs_EM")
assessment = assess(neuralMAP, θ_test, Z_test; estimator_name = "EM NBE")
CSV.write(joinpath(int_path, "rmse_complete_EM.csv"), rmse(assessment))

@info "Training the Masking NBE..."
# Each training data set has its own proportion of missing values, uniform on (0.01, 0.4)
masked(Zs) = encodedata.([removedata(z, rand(Uniform(0.01, 0.4))) for z in Zs])
maskedestimator = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 2)
maskedestimator = train(maskedestimator, θ_train, θ_val, masked(Z_train), masked(Z_val); epochs = epochs, savepath = joinpath(int_path, "runs_masking"))
savenetwork(maskedestimator, "runs_masking")
assessment = assess(maskedestimator, θ_test, encodedata.(remove_complex_missingness.(Z_test)); estimator_name = "Masking NBE")
CSV.write(joinpath(int_path, "rmse_MB_masking.csv"), rmse(assessment))
