using Distributions, StatsPlots, Random
using Turing, MCMCChains
using DynamicPPL, AbstractMCMC
using InvolutiveMCMC

# Set a random seed.
Random.seed!(3)

# Construct 30 data points for each cluster.
N = 30

# Parameters for each cluster, we assume that each cluster is Gaussian distributed in the example.
μs = [-3.5, 0.0]

# Construct the data points.
data = mapreduce(c -> rand(MvNormal([μs[c], μs[c]], 1.), N), hcat, 1:2)

# Visualization.
scatter(data[1,:], data[2,:], legend = false, title = "Synthetic Dataset")

# Turn off the progress monitor.
Turing.setprogress!(false)

@model GaussianMixtureModel(data) = begin

    D, N = size(data)

    # Draw the parameters for cluster 1.
    μ1 ~ Normal()

    # Draw the parameters for cluster 2.
    μ2 ~ Normal()

    μ = [μ1, μ2]

    # Uncomment the following lines to draw the weights for the K clusters
    # from a Dirichlet distribution.

    # α = 1.0
    # w ~ Dirichlet(2, α)

    # Comment out this line if you instead want to draw the weights.
    w = [0.5, 0.5]

    # Draw assignments for each datum and generate it from a multivariate normal.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ Categorical(w)
        data[:,i] ~ MvNormal([μ[k[i]], μ[k[i]]], 1.)
    end
    return k
end

gmm_model = GaussianMixtureModel(data);

gmm_sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ1, :μ2))
tchain = sample(gmm_model, gmm_sampler, 100);

ids = findall(map(name -> occursin("μ", string(name)), names(tchain)));
p=plot(tchain[:, ids, :], legend=true, labels = ["Mu 1" "Mu 2"], colordim=:parameter)
savefig("test/images/01-gaussian-mixture-model-gibbs.png")

# Using iMCMC as the sampler

# generate the log likelihood of model
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(gmm_model, spl) # If you want the log joint
model_loglikelihood = Turing.Inference.gen_logπ(log_joint, spl, gmm_model)

# define the iMCMC model
mh = Involution(s->s[Int(end/2)+1:end],s->s[1:Int(end/2)])
kernel = ProductAuxKernel(vcat([
  # proposal distribution for μ1
  μ1 -> Normal(μ1,1),
  # proposal distribution for μ2
  μ2 -> Normal(μ2,1)],
  # proposal distribution for k
  fill(k -> Categorical([0.5,0.5]),size(data)[2])
))
model = iMCMCModel(mh,kernel,model_loglikelihood,fill(1,2+size(data)[2]))

# generate chain
rng = MersenneTwister(1)
chn = Chains(sample(rng,model,iMCMC(),100;discard_initial=10))
plot(chn[:,1:2,:])
savefig("test/images/01-gaussian-mixture-model-imcmc.png")
