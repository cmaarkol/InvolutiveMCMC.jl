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

# swap involution for MH
uni_mh = Involution(s->s[2],s->s[1])
multi_mh = Involution(s->s[Int(end/2)+1:end],s->s[1:Int(end/2)])

# single-site update (only works for multivariate distributions)
gibb1 = Involution(
    s->vcat(s[2][1],s[1][2:end]),
    s->vcat(s[1][1],s[2][2:end])
)

# continuous iMCMCModel
conkernel(x) = Distributions.Normal(x,1)
conprior = Distributions.Normal()
conprogram = conprior
conloglikelihood(x) = Distributions.logpdf(conprogram,x)
conmodel = iMCMCModel(uni_mh,conkernel,conloglikelihood,conprior)
println("Sample from continuous model")
sample(rng,conmodel,sampler,1000)

# discrete iMCMCModel
diskernel(x) = Distributions.Bernoulli(0.4)
disprior = Distributions.Bernoulli(0.2)
disprogram = disprior
disloglikelihood(x) = Distributions.logpdf(disprogram,x)
dismodel = iMCMCModel(uni_mh,diskernel,disloglikelihood,disprior)
println("Sample from discrete model")
sample(rng,conmodel,sampler,1000)

# continuous multivariate iMCMCModel with dimension mcon
mcon = 5
mconkernel(x) = Distributions.MvNormal(x, I)
mconprior = Distributions.MvNormal(rand(mcon), I)
mconprogram = mconprior
mconloglikelihood(x) = Distributions.logpdf(mconprogram,x)
mconmodel = iMCMCModel(multi_mh,mconkernel,mconloglikelihood,mconprior)
println("Sample from multivariate continuous model")
sample(rng,mconmodel,sampler,1000)

# discrete multivariate iMCMCModel with dimension mdis
mdis = 5
mdiskernel(x) = Distributions.Multinomial(1, [i==1 ? 0.5 : 1 for i in x]/(mdis-0.5))
mdisprior = Distributions.Multinomial(1, fill(1,mdis)/mdis)
mdisprogram = mdisprior
mdisloglikelihood(x) = Distributions.logpdf(mdisprogram,x)
mdismodel = iMCMCModel(multi_mh,mdiskernel,mdisloglikelihood,mdisprior)
println("Sample from multivariate discrete model")
sample(rng,mdismodel,sampler,1000)
