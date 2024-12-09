using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)
quick       = parsed_args["quick"]

m = 3000
K_train = 100000
use_gpu = true
batchsize = 32

include(joinpath(pwd(), "src", "application", "crypto", "Model.jl"))
include(joinpath(pwd(), "src", "application", "crypto", "Architecture.jl"))
int_path = joinpath("intermediates", "application", "crypto")
if !isdir(int_path) mkpath(int_path) end
savepath = joinpath(int_path, "runs_")

epochs = quick ? 20 : 200 # the maximum number of epochs used during training
epochs_per_θ_refresh = 3 # how often to refresh the training parameters
epochs_per_Z_refresh = 3 # how often to refresh the training data

if quick K_train = K_train ÷ 100 end

# ---- Train the neural MAP estimator for use in the EM algorithm ----

@info "Training the neural MAP estimator for use in the neural EM algorithm..."
θ̂ = architecture(ξ; input_channels = 1)
θ̂ = train(θ̂, Parameters, simulate; m = m, savepath = savepath * "EM", ξ = ξ, K = K_train, epochs = epochs, epochs_per_θ_refresh = epochs_per_θ_refresh, epochs_per_Z_refresh = epochs_per_Z_refresh, use_gpu = use_gpu, batchsize = batchsize)
# θ̂ = train(θ̂, Parameters, simulate; m = m, savepath = savepath * "EM", ξ = ξ, K = K_train, epochs = epochs, epochs_per_θ_refresh = epochs_per_θ_refresh, epochs_per_Z_refresh = epochs_per_Z_refresh, loss = (ŷ, y) -> tanhloss(ŷ, y, 0.1f0), use_gpu = use_gpu, batchsize = batchsize)

# ---- Train the one-hot-encoding-based neural estimator  ----

@info "Training the masked neural Bayes estimator..."

# Variable missingness proportion p = (p₁, p₂, p₃) subject to ∑pᵢ = 1
function variableproportions(d::Integer, K::Integer; sum_to_one::Bool = false)
	map(1:K) do _
		proportions = rand(d)
		if sum_to_one
			proportions = proportions/sum(proportions)
		end
		proportions
	end
end

augmentdata(Z, x; kwargs...) = encodedata.(removedata.(Z, x; kwargs...))

function simulatemissing(parameters, m)
	d = size(parameters.chols, 1) # number of elements in the complete-data vector
	K = size(parameters, 2)       # number of parameter vectors in this set
	augmentdata(simulate(parameters, m), variableproportions(d, K))
end
θ̂ = architecture(ξ; input_channels = 2)
train(θ̂, Parameters, simulatemissing; m = m, savepath = savepath * "encoding", ξ = ξ, K = K_train, epochs = epochs, epochs_per_θ_refresh = epochs_per_θ_refresh, epochs_per_Z_refresh = epochs_per_Z_refresh, use_gpu = use_gpu, batchsize = batchsize)
#train(θ̂, Parameters, simulatemissing; m = m, savepath = savepath * "encoding", ξ = ξ, K = K_train, epochs = epochs, epochs_per_θ_refresh = epochs_per_θ_refresh, epochs_per_Z_refresh = epochs_per_Z_refresh, loss = (ŷ, y) -> tanhloss(ŷ, y, 0.1f0), use_gpu = use_gpu, batchsize = batchsize)
