using InvolutiveMCMC
using AbstractMCMC
using Distributions
using Random
using MCMCChains
using LinearAlgebra
using StatsPlots

# random seed
rng = MersenneTwister(1)

# sampler
sampler = iMCMC()

# 2d Gaussian mixture model
μ1 = [0, 0]; Σ1 = I; w1 = 0.5
μ2 = [2.5,2.5]; Σ2 = I; w2 = 0.5
function loglikelihood(x)
    density = w1*pdf(MvNormal(μ1,Σ1),x) + w2*pdf(MvNormal(μ2,Σ2),x)
    return log(density)
end

# involution
mh = Involution(s->s[2],s->s[1])

# auxiliary kernel
kernel(x) = MvNormal(x,1)

# prior
prior = MvNormal(μ1,Σ1)

# iMCMC Model
model = iMCMCModel(mh,kernel,loglikelihood,prior)

# sample
chn = Chains(sample(rng,model,sampler,1000))

describe(chn)

plot(chn)
