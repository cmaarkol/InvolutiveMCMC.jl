using InvolutiveMCMC
using AbstractMCMC
using Distributions
using Bijectors
using Random

# random seed
rng = MersenneTwister(1)

# continuous iMCMCModel
inv(x,v) = (v,x)
kernel(x) = Distributions.Normal(x,1)
program = Distributions.Normal()
loglikelihood(x) = Distributions.logpdf(program,x)
conmodel = iMCMCModel(inv,kernel,loglikelihood)
