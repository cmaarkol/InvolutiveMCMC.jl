using InvolutiveMCMC
using AbstractMCMC
using Distributions
using Random
using MCMCChains
using LinearAlgebra
using StatsPlots

# Implementation of Mixture Proposal MCMC

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
# Φ(x,a,v) = (v,a,x)
# newx(x,a,v) = v
# newv(x,a,v) = (a,x)
mixture_inv = Involution(
    s->s[2*Int(end/3)+1:end],
    s->vcat(s[Int(end/3)+1:2*Int(end/3)],s[1:Int(end/3)])
)

# auxiliary kernel
mixture_kernels = [x -> MvNormal(x,I), a -> Dirichlet(abs.(a))]

# prior
prior = MvNormal(μ1,Σ1)

# iMCMC Model
model = iMCMCModel(mixture_inv,mixture_kernels,loglikelihood,prior)

# sample
chn = Chains(sample(rng,model,sampler,1000))
