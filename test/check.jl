using InvolutiveMCMC
using AbstractMCMC
using Distributions
using Bijectors
using Random

# form a iMCMCModel
inv(x,v) = (v,x)
d = Distributions.Normal()
l(x) = Distributions.pdf(d,x)
model = iMCMCModel(inv,d,l)

# random seed
rng = MersenneTwister(1) 