using Turing.RandomMeasures
using Turing

@model function infiniteGMM(x)
    # Hyper-parameters, i.e. concentration parameter and parameters of H.
    α = 1.0
    μ0 = 0.0
    σ0 = 1.0
    
    # Define random measure, e.g. Dirichlet process.
    rpm = DirichletProcess(α)
    
    # Define the base distribution, i.e. expected value of the Dirichlet process.
    H = Normal(μ0, σ0)
    
    # Latent assignment.
    z = tzeros(Int, length(x))
        
    # Locations of the infinitely many clusters.
    μ = tzeros(Float64, 0)
    
    for i in 1:length(x)
        
        # Number of clusters.
        K = maximum(z)
        nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

        # Draw the latent assignment.
        z[i] ~ ChineseRestaurantProcess(rpm, nk)
        
        # Create a new cluster?
        if z[i] > K
            push!(μ, 0.0)

            # Draw location of new cluster.
            μ[z[i]] ~ H
        end
                
        # Draw observation.
        x[i] ~ Normal(μ[z[i]], 1.0)
    end
end

using Plots, Random

# Generate some test data.
Random.seed!(1)
data = vcat(randn(10), randn(10) .- 5, randn(10) .+ 10)
data .-= mean(data)
data /= std(data);

# MCMC sampling
Random.seed!(2)
iterations = 1000
model_fun = infiniteGMM(data);
chain = sample(model_fun, SMC(), iterations);

# Extract the number of clusters for each sample of the Markov chain.
k = map(
    t -> length(unique(vec(chain[t, MCMCChains.namesingroup(chain, :z), :].value))),
    1:iterations
);

# # Visualize the number of clusters.
# plot(k, xlabel = "Iteration", ylabel = "Number of clusters", label = "Chain 1")
# savefig("test/images/06-infinite-mixture-model-chain-smc.png")

# histogram(k, xlabel = "Number of clusters", legend = false)
# savefig("test/images/06-infinite-mixture-model-histogram-smc.png")

# Using iMCMC as the sampler
using DynamicPPL, LinearAlgebra, InvolutiveMCMC

# generate the log likelihood of model
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(model_fun, spl) # If you want the log joint
model_loglikelihood = Turing.Inference.gen_logπ(log_joint, spl, model_fun)
first_sample = log_joint[spl]

