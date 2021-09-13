# private interface

"""
    initial_sample(rng, model)

Return the initial sample for the `model` using the random number generator `rng`.
"""
function initial_sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel
)
    # sample from prior (high-level)
    x, logp = sample_model(rng, getloglikelihood(model))
    return x, logp
end

"""
    proposal(model, x, v)

Compute the proposal for the next sample using the `model`'s involution, the current sample `x` and the auxiliary sample `v`.
"""
function proposal(inv::AbstractInvolution, x, v)
    newx, newv = involution(inv, x, v)
    return newx, newv
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
    np_gen_logπ(spl::Sampler, model)

Modified from `Turing.Inference.gen_logπ` to accommodate for nonparametric models.

Generate a function that takes `x` and returns a triple `(k, ds, logp)` where
- `x[1:k]` is the vector of values being used as samples;
- `ds` is the vector of the next distributions;
- `logp` is the log density at `x` for the model specified by `model`.
"""
function np_gen_logπ(spl, turing_model)
    function logπ(x)
        x = vcat(x)
        k = 0
        # variable info for model
        vi = VarInfo(turing_model, spl)
        n = length(vi.metadata)
        # invariant: x[1:k] is used as samples for variables 1:i
        for i in 1:n
            # empty data stored in vi for variables i+1:n
            for j in i+1:n
                deep_empty!(vi.metadata[j])
            end
            # try to set the value for the i-th variable according to x
            try
                vi[spl] = x
                k += length(vi.metadata[i].vals)
            catch err
                if isa(err, BoundsError)
                    return k, vi.metadata[i].dists, -Inf
                end
            end
            # run model with new value, which initialises the dimension for the (i+1)-th variable
            turing_model(vi, spl)
        end
        logp = Turing.Inference.getlogp(vi)
        return k, missing, logp
    end
    return logπ
end

"""
    gen_logπ(spl::Sampler, model)

Modified from `Turing.Inference.gen_logπ` to accommodate for fixed parameter space models.

Generate a function that takes `x` and returns the log density at `x` for the model specified by `model`.
"""

function gen_logπ(spl, turing_model)
    function logπ(x)
        x = vcat(x)
        # variable info for model
        vi = VarInfo(turing_model, spl)
        # set the value according to x
        vi[spl] = x
        # run model with new value
        turing_model(vi, spl)
        logp = Turing.Inference.getlogp(vi)
        return logp
    end
    return logπ
end

"""
    sample_model(rng, loglikelihood)

Sample from the prior of `loglikelihood`
"""
function sample_model(rng::Random.AbstractRNG, loglikelihood)
    x = []
    k, ds, logp = loglikelihood(x)
    while logp == -Inf
        x = x[1:k]
        for dist in ds
            e = Random.rand(rng, dist)
            append!(x, typeof(dist) <: UnivariateDistribution ? [e] : e)
        end
        k, ds, logp = loglikelihood(x)
    end
    return x[1:k], logp
end
