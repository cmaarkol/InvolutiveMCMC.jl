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
mh = Involution(s->s[2],s->s[1])
mmh = Involution(s->s[Int(end/2)+1:end],s->s[1:Int(end/2)])
# Φ(x,a,y,v) = (v,a,y,x)
# newx(x,a,y,v) = v
# newv(x,a,y,v) = (a,y,x)
cmh = n -> Involution(
    s->s[end],
    s->vcat(s[Int(end/(n+1))+1:n*Int(end/(n+1))],s[1:Int(end/(n+1))])
)
cmmh = n -> Involution(
    s->s[n*Int(end/(n+1))+1:end],
    s->vcat(s[Int(end/(n+1))+1:n*Int(end/(n+1))],s[1:Int(end/(n+1))])
)
# single-site update (only works for multivariate distributions)
gibb1 = Involution(
    s->vcat(s[2][1],s[1][2:end]),
    s->vcat(s[1][1],s[2][2:end])
)
# number of samples
N = 10

# continuous (c)
ckernel = AuxKernel(x->Normal(x,1))
cprior = Normal()
cprogram = cprior
cloglikelihood(x) = logpdf(cprogram,x)
cmodel = iMCMCModel(mh,ckernel,cloglikelihood,cprior)
println("Sample from continuous model")
sample(rng,cmodel,sampler,N)

# discrete (d)
dkernel = AuxKernel(x->Bernoulli(0.4))
dprior = Bernoulli(0.2)
dprogram = dprior
dloglikelihood(x) = logpdf(dprogram,x)
dmodel = iMCMCModel(mh,dkernel,dloglikelihood,dprior)
println("Sample from discrete model")
sample(rng,dmodel,sampler,N)

# multivariate continuous (mc)
mcdim = 5
mckernel = AuxKernel(x->MvNormal(x, I))
mcprior = MvNormal(rand(mcdim), I)
mcprogram = mcprior
mcloglikelihood(x) = logpdf(mcprogram,x)
mcmodel = iMCMCModel(mmh,mckernel,mcloglikelihood,mcprior)
println("Sample from multivariate continuous model")
sample(rng,mcmodel,sampler,N)

# multivariate discrete (md)
mddim = 5
mdkernel = AuxKernel(x->Multinomial(1, [i==1 ? 0.5 : 1 for i in x]/(mddim-0.5)))
mdprior = Multinomial(1, fill(1,mddim)/mddim)
mdprogram = mdprior
mdloglikelihood(x) = logpdf(mdprogram,x)
mdmodel = iMCMCModel(mmh,mdkernel,mdloglikelihood,mdprior)
println("Sample from multivariate discrete model")
sample(rng,mdmodel,sampler,N)

# continuous composite kernel (cc)
cckernel = CompositeAuxKernel([x->Normal(x,1), y->Normal(y,1)])
ccmodel = iMCMCModel(cmh(2),cckernel,cloglikelihood,cprior)
println("Sample from continuous model with composite kernel")
sample(rng,ccmodel,sampler,N)

# muiltivate continuous composite kernel (mcc)
mcckernel = CompositeAuxKernel([x -> MvNormal(x,I), a -> Dirichlet(abs.(a)), y -> MvNormal(y,I)])
mccmodel = iMCMCModel(cmmh(3),mcckernel,mcloglikelihood,mcprior)
println("Sample from multivariate continuous model with composite kernel")
sample(rng,mccmodel,sampler,N)

# product continuous (pc)
pckernel = ProductAuxKernel([v1->Normal(v1,1), v2->Normal(v2+2,1), v3->Normal(v3-2,1)], ones(Int,3))
pcprior = MvNormal([1,2,3],I)
pcprogram = pcprior
pcloglikelihood(x) = logpdf(pcprogram,x)
pcmodel = iMCMCModel(mmh,pckernel,pcloglikelihood,pcprior)
println("Sample from product continuous model")
sample(rng,pcmodel,sampler,N)

# product multivariate continuous (pmc)

