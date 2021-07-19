# private interface

"""
    initial_sample(rng, model)

Return the initial sample for the `model` using the random number generator `rng`.
"""
function initial_sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel
)
    init_sample = Random.rand(rng, model.auxiliary_kernel(0))
    while isinf(model.loglikelihood(init_sample))
        init_sample = Random.rand(rng, model.auxiliary_kernel(0))
    end
    return init_sample
end

"""
    aux_kernel_sampler(rng, model, x)

Return a random sample from the `model`'s `auxiliary_kernel` conditioned on `x`

"""
function aux_kernel_sampler(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, x)
    return Random.rand(rng, model.auxiliary_kernel(x))
end

"""
    aux_kernel_loglikelihood(model, x, v)

Return the log likelihood of `v` from the `model`'s `auxiliary_kernel` conditioned on `x`

"""
function aux_kernel_loglikelihood(model::AbstractMCMC.AbstractModel, x, v)
    return Distributions.loglikelihood(model.auxiliary_kernel(x), v)
end

"""
    proposal(involution, x, v)

Compute the proposal for the next sample using `involution`, the current sample `x` and the auxiliary sample `v`.

See also: [`proposal!`](@ref)
"""
function proposal(model::AbstractMCMC.AbstractModel, x, v)
    newx, newv = model.involution(x, v)
    return newx, newv
end
