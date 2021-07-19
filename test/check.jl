using InvolutiveMCMC
using AbstractMCMC
using Distributions
using Bijectors
using Random
using LinearAlgebra

# random seed
rng = MersenneTwister(1)

# sampler
sampler = iMCMC()

# continuous iMCMCModel
coninv(x,v) = (v,x)
conkernel(x) = Distributions.Normal(x,1)
conprior = Distributions.Normal()
conprogram = conprior
conloglikelihood(x) = Distributions.logpdf(conprogram,x)
conmodel = iMCMCModel(coninv,conkernel,conloglikelihood,conprior)

# discrete iMCMCModel
disinv(x,v) = (v,x)
diskernel(x) = Distributions.Bernoulli(0.4)
disprior = Distributions.Bernoulli(0.2)
disprogram = disprior
disloglikelihood(x) = Distributions.logpdf(disprogram,x)
dismodel = iMCMCModel(disinv,diskernel,disloglikelihood,disprior)

# continuous multivariate iMCMCModel with dimension mcon
mcon = 5
mconinv(x,v) = (v,x)
mconkernel(x) = Distributions.MvNormal(x, I)
mconprior = Distributions.MvNormal(rand(mcon), I)
mconprogram = mconprior
mconloglikelihood(x) = Distributions.logpdf(mconprogram,x)
mconmodel = iMCMCModel(mconinv,mconkernel,mconloglikelihood,mconprior)

# discrete multivariate iMCMCModel with dimension mdis
mdis = 5
mdisinv(x,v) = (v,x)
mdiskernel(x) = Distributions.Multinomial(1, [i==1 ? 0.5 : 1 for i in x]/(mdis-0.5))
mdisprior = Distributions.Multinomial(1, fill(1,mdis)/mdis)
mdisprogram = mdisprior
mdisloglikelihood(x) = Distributions.logpdf(mdisprogram,x)
mdismodel = iMCMCModel(mdisinv,mdiskernel,mdisloglikelihood,mdisprior)
