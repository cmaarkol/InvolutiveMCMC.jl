using InvolutiveMCMC
using Distributions, StatsPlots, Random
using Turing, MCMCChains
using DynamicPPL, AbstractMCMC

filename = "01-gaussian-mixture-model"
imagepath = "test/images/"

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

# Sampler
inf = "gibbs&hmc"
rng1 = MersenneTwister(1)
gmm_sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ1, :μ2))
tchain = sample(rng1, gmm_model, gmm_sampler, 100);
ids = findall(map(name -> occursin("μ", string(name)), names(tchain)));
plot(tchain[:, ids, :], legend=true, labels = ["Mu 1" "Mu 2"], colordim=:parameter)
savefig(join([imagepath, filename, "-", inf, ".png"]))

# Using iMCMC as the sampler
inf = "imcmc"
ADMH = ADInvolution(
    (x, v)->(v, x),
    s->(s[1:Int(end/2)], s[Int(end/2)+1:end])
)
kernel = PointwiseAuxKernel(
    (n, t) -> (n == 1 || n == 2) ?
    Normal(t[n], 10) :
    Categorical([0.5, 0.5])
)
model = iMCMCModel(ADMH, kernel, gmm_model)

# generate chain
rng2 = MersenneTwister(2)
imcmcsamples = sample(rng2, model, iMCMC(), 10000)
imcmcchn = Chains(convert(Matrix{Float64}, reduce(hcat, imcmcsamples)')[:,1:2,:], [:μ1, :μ2])
plot(imcmcchn)
savefig(join([imagepath, filename, "-", inf, ".png"]))
