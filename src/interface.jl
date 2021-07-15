# private interface

"""
    initial_sample(rng, model)

Return the initial sample for the `model` using the random number generator `rng`.
"""
function initial_sample(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel)
    return Random.rand(rng)
end

"""
    aux_kernel_sampler(rng,model)

Return a random sample from the model's auxiliary_kernel

"""
function aux_kernel_sampler(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel)
    return Random.rand(rng, model.auxiliary_kernel)
end

"""
    aux_kernel_loglikelihood(model, sample)

Return the log likelihood of sample from the model's auxiliary_kernel

"""
function aux_kernel_loglikelihood(model::AbstractMCMC.AbstractModel, sample)
    return Distributions.loglikelihood(model.auxiliary_kernel, sample)
end

"""
    proposal(involution, x, v)

Compute the proposal for the next sample using involution, the current sample x and the auxiliary sample v.

See also: [`proposal!`](@ref)
"""
function proposal(model::AbstractMCMC.AbstractModel, x, v)
    newx, newv = model.involution(x, v)
    return newx, newv
end
