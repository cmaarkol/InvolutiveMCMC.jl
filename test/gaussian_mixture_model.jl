using InvolutiveMCMC
using AbstractMCMC
using Distributions
using Random
using MCMCChains

# random seed
rng = MersenneTwister(1)

# sampler
sampler = iMCMC()

# Gaussian mixture model
μ1 = 0; σ1 = 2; w1 = 0.5
μ2 = 5; σ2 = 2; w2 = 0.5
function loglikelihood(x)
    density = w1*Distributions.pdf(Distributions.Normal(μ1,σ1),x) + w2*Distributions.pdf(Distributions.Normal(μ2,σ2),x)
    return log(density)
end

# involution
inv(x,v) = (v,x)

# auxiliary kernel
kernel(x) = Distributions.Normal(x,1)

# prior
prior = Distributions.Normal(μ1,σ1)

# iMCMC Model
model = iMCMCModel(inv,kernel,loglikelihood,prior)

# sample
chn = Chains(AbstractMCMC.sample(rng,model,sampler,100000))

plot(chn)
