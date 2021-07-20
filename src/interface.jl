# private interface

"""
    initial_sample(rng, model)

Return the initial sample for the `model` using the random number generator `rng`.
"""
function initial_sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel
)
    return Random.rand(rng, prior(model))
end

"""
    aux_kernel_sampler(rng, model, x)

Return a random sample from the `model`'s `auxiliary_kernel` conditioned on `x`

"""
function aux_kernel_sampler(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, x)
    return Random.rand(rng, auxiliary_kernel(model, x))
end

"""
    aux_kernel_loglikelihood(model, x, v)

Return the log likelihood of `v` from the `model`'s `auxiliary_kernel` conditioned on `x`

"""
function aux_kernel_loglikelihood(model::AbstractMCMC.AbstractModel, x, v)
    return Distributions.loglikelihood(auxiliary_kernel(model, x), v)
end

"""
    proposal(model, x, v)

Compute the proposal for the next sample using the `model`'s involution, the current sample `x` and the auxiliary sample `v`.
"""
function proposal(model::AbstractMCMC.AbstractModel, x, v)
    newx, newv = involution(model, x, v)
    return newx, newv
end

"""
    split_in_half(s)

Split the array `s` in equal halves
"""
function split_in_half(s::AbstractArray)
    dim = Int(length(s)/2)
    if dim == 1
        return s[1], s[2]
    else
        return s[1:dim], s[dim+1:end]
    end
end
