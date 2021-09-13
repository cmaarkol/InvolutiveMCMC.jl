# involutive MCMC
struct iMCMC <: AbstractMCMC.AbstractSampler end

# state of the involutive MCMC
struct iMCMCState{S,L}
    "Sample of the involutive MCMC."
    sample::S
    "Log-likelihood of the sample."
    loglikelihood::L
end

# first step of the involutive MCMC
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    ::iMCMC;
    kwargs...
)
    xsample, logp = initial_sample(rng, model)
    return xsample, iMCMCState(xsample, logp)
end

# subsequent steps of the involutive MCMC
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    ::iMCMC,
    state::iMCMCState;
    kwargs...,
)
    inv = getinvolution(model)
    kernel = getkernel(model)

    # println("----STEP START----")
    # previous sample and its log likelihood
    xsample = state.sample
    xlogp = state.loglikelihood
    # println("xsample = ", xsample)
    # println("xlogp = ", xlogp)

    # sample from the auxiliary kernel and compute its log likelihood
    vsample = Random.randn(rng, xsample, kernel)
    vlogp = Distributions.loglikelihood(kernel, xsample, vsample)
    # println("vsample = ", vsample)
    # println("vlogp = ", vlogp)

    # compute the new sample and auxiliary using involution
    newxsample, newvsample = proposal(inv, xsample, vsample)
    # println("newxsample = ", newxsample)
    # println("newvsample = ", newvsample)

    # compute the log likelihood of the newxsample and newvsample
    _, _, newxlogp = logpdf(model, newxsample)
    # println("newxlogp = ", newxlogp)
    newvlogp = Distributions.loglikelihood(kernel, newxsample, newvsample)
    # println("newvlogp = ", newvlogp)

    # compute the log Hastings acceptance ratio
    invlogabsdetjac = logabsdetjacinv(inv, xsample, vsample)
    # println("invlogabsdetjac = ", invlogabsdetjac)
    logα = newxlogp + newvlogp - xlogp - vlogp + invlogabsdetjac
    # println("log acceptance ratio = ", logα)

    nextsample, nextstate = if -Random.randexp(rng) < logα
        newxsample, iMCMCState(newxsample, newxlogp)
    else
        xsample, iMCMCState(xsample, xlogp)
    end
    # println(nextsample)
    return nextsample, nextstate
end
