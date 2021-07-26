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
aux_kernel_sampler(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, x) = randkernel(rng, auxiliary_kernel(model), x)

function aux_kernel_sampler_old(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, x)
    kernels = auxiliary_kernel(model)
    if typeof(kernels) <: AbstractArray
        v = typeof(x)[]
        for kernel in kernels
            x = Random.rand(rng, kernel(x))
            append!(v, [x])
        end
    else
        v = Random.rand(rng, kernels(x))
    end
    return v
end

"""
    aux_kernel_loglikelihood(model, x, v)

Return the log likelihood of `v` from the `model`'s `auxiliary_kernel` conditioned on `x`

"""
function aux_kernel_loglikelihood_old(model::AbstractMCMC.AbstractModel, x, v)
    kernels = auxiliary_kernel(model)
    if typeof(kernels) <: AbstractArray
        p = 0; conditioned_sample = x
        for (kernel,vsample) in zip(kernels,v)
            # println(kernel(conditioned_sample))
            # println(vsample)
            p += Distributions.loglikelihood(kernel(conditioned_sample), vsample)
            conditioned_sample = vsample
        end
    else
        p = Distributions.loglikelihood(kernels(x), v)
    end
    # println("p = ",p)
    return p
end

"""
    proposal(model, x, v)

Compute the proposal for the next sample using the `model`'s involution, the current sample `x` and the auxiliary sample `v`.
"""
function proposal(model::AbstractMCMC.AbstractModel, x, v)
    if typeof(v) <: AbstractVector
        # flatten v and then reshape newv
        flatv = collect(Iterators.flatten(v))
        flatstate = vcat(x,flatv)
        # println("flatstate = ", flatstate)
        newx, newflatv = involution(model, flatstate)
        # println("newx = ", newx)
        # println("newflatv = ", newflatv)
        newv = reshape(newflatv, [length(elem) for elem in v])
    else
        newx, newv = involution(model, vcat(x,v))
    end
    return newx, newv
end

"""
    reshape(s,shape)

Reshape the vector `s` into a vector describes by `shape`
"""
function reshape(s::AbstractVector, shape)
    length(s) == sum(shape) || error("Cannot reshape a vector with this shape")
    if length(s) == 0
        return s
    elseif all(shape .== 1)
        return s
    else
        current_index = 1
        news = typeof(s)[]
        for eshape in shape
            append!(news, [s[current_index:current_index+eshape-1]])
            current_index += eshape
        end
        return news
    end
end
