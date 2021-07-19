using InvolutiveMCMC
using AbstractMCMC
using Distributions
using Bijectors
using Random

# random seed
rng = MersenneTwister(1)

# sampler
sampler = iMCMC()

# continuous iMCMCModel
coninv(x,v) = (v,x)
conkernel(x) = Distributions.Normal(x,1)
conprogram = Distributions.Normal()
conloglikelihood(x) = Distributions.logpdf(conprogram,x)
conmodel = iMCMCModel(coninv,conkernel,conloglikelihood)

# discrete iMCMCModel
disinv(x,v) = (v,x)
diskernel(x) = Distributions.Bernoulli(0.4)
disprogram = Distributions.Bernoulli(0.2)
disloglikelihood(x) = Distributions.logpdf(disprogram,x)
dismodel = iMCMCModel(disinv,diskernel,disloglikelihood)
