# private interface

"""
    initial_sample(rng, model)

Return the initial sample for the `model` using the random number generator `rng`.
"""
function initial_sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel
)
    if typeof(prior(model)) <: Distributions.Sampleable
        return Random.rand(rng, prior(model))
    else
        return prior(model)
    end
end

"""
    aux_kernel_sampler(rng, model, x)

Return a random sample from the `model`'s `auxiliary_kernel` conditioned on `x`

"""
aux_kernel_sampler(rng::Random.AbstractRNG, model::AbstractMCMC.AbstractModel, x) = Random.rand(rng, x, auxiliary_kernel(model))

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
        newflatx, newflatv = involution(model, flatstate)
        # println("newflatx = ", newflatx)
        # println("newflatv = ", newflatv)
        # newv = reshape_sample(newflatv, [length(elem) for elem in v])
        newx = model.involution.shapex(newflatx)
        newv = model.involution.shapev(newflatv)
    else
        newx, newv = involution(model, vcat(x,v))
    end
    return newx, newv
end

"""
    reshape(s,shape)

Reshape the vector `s` into a vector describes by `shape`
"""
function reshape_sample(s::AbstractVector, shape)
    if length(s) == 0
        return s
    elseif all(shape .== 1)
        return s
    else
        length(s) == sum(shape) || error("Cannot reshape a vector with this shape")
        current_index = 1
        news = typeof(s)[]
        for eshape in shape
            append!(news, [s[current_index:current_index+eshape-1]])
            current_index += eshape
        end
        return news
    end
end

"""
    deep_empty(meta::Metadata)

Similar to `empty!(::Metadata)` in VarInfo but with the type of `meta.dists` also removed.
"""

function deep_empty!(meta::DynamicPPL.Metadata)
    empty!(meta.idcs)
    empty!(meta.vns)
    empty!(meta.ranges)
    empty!(meta.vals)
    empty!(meta.dists)
    # need to find a way to empty the type of meta.dists
    # something like this would be brilliant
    # convert!(Vector{Distribution}, meta.dists)
    empty!(meta.gids)
    empty!(meta.orders)
    for k in keys(meta.flags)
        empty!(meta.flags[k])
    end
    return meta
end


"""
    trans_dim_gen_logπ(vi, spl::Sampler, model; empty_vns=[])

Modified from `Turing.Inference.gen_logπ` to accommodate for samples of different length.

Generate a function that takes `θ` and returns logpdf at `θ` for the model specified by
`(vi, spl, model)`.
"""
function trans_dim_gen_logπ(vi, spl, model)
    function logπ(x)::Float64
        x = vcat(x)
        # number of variables in vi
        n = length(vi.metadata)
        # set the values for variables one by one
        for i in 1:n
            # empty data stored in vi for variables i+1:n
            for j in i+1:n
                deep_empty!(vi.metadata[j])
            end
            # set variable at i according to x
            vi[spl] = x
            # run model with new value for variable i, which initialises the dimension for variable i+1
            model(vi, spl)
        end
        lj = Turing.Inference.getlogp(vi)
        return lj
    end
    return logπ
end


"""
    insert_list(x1::AbstractVector,i::Int64,x2::AbstractVector)

insert `x2` into `x1` at position `i`
"""

function insert_list(x1::AbstractVector,i::Int64,x2::AbstractVector)
    res = Float64.(x1)
    inserted = 1
    while inserted <= length(x2)
        res = insert!(res,i+inserted-1,x2[inserted])
        inserted += 1
    end
    return res
end

"""
    remove_list(x::AbstractVector,i::Int64,n::Int64)

remove `x[i:i+n-1]` from `x`
"""

function remove_list(x::AbstractVector,i::Int64,n::Int64)
    res = vcat(x[1:i-1],x[i+n:end])
    return res
end
