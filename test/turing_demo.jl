using Turing
using DynamicPPL
using InvolutiveMCMC
using AbstractMCMC
using Distributions
using Random
using MCMCChains
using LinearAlgebra
using StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
  s² ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s²))
  x ~ Normal(m, sqrt(s²))
  y ~ Normal(m, sqrt(s²))
end

#  Run sampler, collect results
rng2 = MersenneTwister(2)
hmcchn = sample(rng2,gdemo(1.5, 2), HMC(0.1, 5), 1000)
hmcp = plot(hmcchn)
savefig("test/images/gdemo-plot-hmc.png")

# generate the log likelihood of model
m = gdemo(1.5, 2)
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(m, spl) # If you want the log joint
model_loglikelihood = Turing.Inference.gen_logπ(log_joint, spl, m)

# define the iMCMC model
mh = Involution(s->s[Int(end/2)+1:end],s->s[1:Int(end/2)])
kernel(x) = product_distribution([
  # proposal distribution for s²
  Truncated(Normal(x[1],1), 0, 10),
  # proposal distribution for m
  Normal(x[2],1)
])
prior = product_distribution([Truncated(Normal(2,1), 0, 10), Normal(0,1)])
model = iMCMCModel(mh,kernel,model_loglikelihood,prior)

# generate chain
rng1 = MersenneTwister(1)
imcmcchn = Chains(sample(rng1,model,iMCMC(),100000))
imcmcp = plot(imcmcchn)
savefig("test/images/gdemo-imcmc.png")
