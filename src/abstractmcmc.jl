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
    # initial sample from the model
    f = initial_sample(rng, model)

    # compute log-likelihood of the initial sample
    loglikelihood = Distributions.loglikelihood(model, f)

    return f, iMCMCState(f, loglikelihood)
end

# subsequent steps of the involutive MCMC
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    ::iMCMC,
    state::iMCMCState;
    kwargs...,
)
    # println("----STEP START----")
    # previous sample and its log likelihood
    xsample = state.sample
    xloglikelihood = state.loglikelihood
    # println("xsample = ", xsample)
    # println("xloglikelihood = ", xloglikelihood)

    # sample from the auxiliary kernel and compute its log likelihood
    vsample = aux_kernel_sampler(rng, model, xsample)
    vloglikelihood = Distributions.loglikelihood(auxiliary_kernel(model), xsample, vsample)
    # println("vsample = ", vsample)
    # println("vloglikelihood = ", vloglikelihood)

    # compute the new sample and auxiliary using involution
    newxsample, newvsample = proposal(model, xsample, vsample)
    # println("newxsample = ", newxsample)
    # println("newvsample = ", newvsample)

    # compute the log likelihood of the newxsample and newvsample
    newxloglikelihood = Distributions.loglikelihood(model, newxsample)
    # println("newxloglikelihood = ", newxloglikelihood)
    newvloglikelihood = Distributions.loglikelihood(auxiliary_kernel(model), newxsample, newvsample)
    # println("newvloglikelihood = ", newvloglikelihood)

    # compute the log Hastings acceptance ratio
    involutionlogabsdetjac = logabsdetjac(model.involution, xsample, vsample)
    # println("involutionlogabsdetjac = ", involutionlogabsdetjac)
    logα = newxloglikelihood + newvloglikelihood - xloglikelihood - vloglikelihood + involutionlogabsdetjac
    # println("log acceptance ratio = ", logα)

    nextsample, nextstate = if -Random.randexp(rng) < logα
        newxsample, iMCMCState(newxsample, newxloglikelihood)
    else
        xsample, iMCMCState(xsample, xloglikelihood)
    end
    # println(nextsample)
    return nextsample, nextstate
end
