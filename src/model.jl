"""
    `iMCMCModel` is the internal model structure.
"""
struct iMCMCModel{I,A} <: AbstractMCMC.AbstractModel
    "Involution."
    involution::I
    "Auxiliary kernel."
    auxiliary_kernel::A
    "Log likelihood function."
    loglikelihood

    iMCMCModel(
        involution,
        auxiliary_kernel,
        turing_model;
        sampler=DynamicPPL.SampleFromPrior()
    ) =
    new{typeof(involution),typeof(auxiliary_kernel)}(involution, auxiliary_kernel, np_gen_logπ(sampler,turing_model))
end

# evaluate the loglikelihood of a sample
logpdf(model::iMCMCModel, x) = model.loglikelihood(x)

# get functions
getinvolution(model::iMCMCModel) = model.involution
getkernel(model::iMCMCModel) = model.auxiliary_kernel
getloglikelihood(model::iMCMCModel) = model.loglikelihood

"""
    `AbstractInvolution` is the abstract type of all involutions.

- `involution` run the involution and return a newx-newv tuple.
- `logabsdetjac` returns the log absolute value of the Jacobian determinant of the involution with respect to `x` and `v`.
"""
abstract type AbstractInvolution{AD} <: Bijectors.ADBijector{AD, 0} end

"""
    `Involution` constructs an involution where logabsdetjac is defined by the user.
"""
struct Involution{T, L, AD} <: AbstractInvolution{AD}
    "Involution." # function that maps current state (x,v) to the new state
    involution::T
    "Logabsdetjac" # function that maps current state (x,v) to its log absolute value Jacobian determinant.
    logabsdetjac::L

    Involution(involution, logabsdetjac) = new{typeof(involution), typeof(logabsdetjac), ADBackend()}(involution, logabsdetjac)
end

involution(i::Involution, x, v) = i.involution(x, v)
logabsdetjacinv(i::Involution, x, v) = i.logabsdetjac(x, v)

"""
    `ADInvolution` constructs an involution where logabsdetjac is defined using Bijectors.
"""
struct ADInvolution{T, S, AD} <: AbstractInvolution{AD}
    "Involution." # function that maps current state (x,v) to the new state
    involution::T
    "State function." # function that maps s to (x,v)
    state::S

    ADInvolution(involution, state) = new{typeof(involution), typeof(state), ADBackend()}(involution, state)
end

involution(i::ADInvolution, x, v) = i.involution(x, v)

function (b::ADInvolution)(s)
    x, v = b.state(s)
    newx, newv = involution(b, x, v)
    return vcat(newx, newv)
end
function (ib::Bijectors.Inverse{<:ADInvolution})(s)
    b = ib.orig
    x, v = b.state(s)
    newx, newv = involution(b, x, v)
    return vcat(newx, newv)
end

logabsdetjacinv(i::ADInvolution, x, v) = Bijectors.logabsdetjac(i, convert(Vector{Float64}, vcat(x, v)))

"""
    `CompositeInvolution` constructs an involution where logabsdetjac is defined by the user.
"""
struct CompositeInvolution{I, L, AD} <: AbstractInvolution{AD}
    "InvolutionMap." # map n to an involution that map current state (x,v) to the new state
    involutionMap::I
    "Logabsdetjac." # map n to the log absolute value Jacobian determinant of the corresponding involution
    logabsdetjacMap::L

    # The algorithm is run as follows.
    # n = init
    # while constraint(n, sample)
    #     nextsample = step(sample, involutionMap(n))
    #     n = next(n)
    # end
    "Initialisation."
    init
    "Constaint."
    constraint
    "Next."
    next

    CompositeInvolution(
        involutionMap,
        logabsdetjacMap;
        init = (rng, x) -> 1,
        constraint = (n, x) -> n ≤ length(x),
        next = n -> n+1) =
    new{typeof(involutionMap), typeof(logabsdetjacMap), ADBackend()}(involutionMap, logabsdetjacMap, init, constraint, next)
end

involution(i::CompositeInvolution, x, v, n::Int) = i.involutionMap(n)(x,v)

logabsdetjacinv(i::CompositeInvolution, x, v, n::Int) = i.logabsdetjacMap(n)(x,v)

"""
    `Bijection` constructs an involution where logabsdetjac is defined by the user.
"""
struct Bijection{T, I, L, AD} <: AbstractInvolution{AD}
    "Bijection." # function that maps current state (x,v) to the new state
    bijection::T
    "Inverse of bijection" # inverse of bijection
    invbijection::I
    "Logabsdetjac" # function that maps current state (x,v) to its log absolute value Jacobian determinant.
    logabsdetjac::L

    Bijection(bijection, invbijection, logabsdetjac) = new{typeof(bijection), typeof(invbijection), typeof(logabsdetjac), ADBackend()}(bijection, invbijection, logabsdetjac)
end

randdir(rng::Random.AbstractRNG, i::Bijection) = Random.rand(rng, Bernoulli(0.5))
involution(i::Bijection, x, v, d::Bool) = d ? i.bijection(x, v) : i.invbijection(x, v)
function logabsdetjacinv(i::Bijection, x, v, d::Bool)
    if d
        return i.logabsdetjac(x, v)
    else
        newx, newv = i.invbijection(x, v)
        return -i.logabsdetjac(newx, newv)
    end
end

"""
    `AbstractAuxKernel` is the abstract type of all auxiliary kernels.

- `Random.rand` returns the `n`-th element of the random sample from the kernel conditioned on the vector `x::Vector{Float64}`.
- `Random.randn` returns a random sample from the kernel conditioned on the vector `x::Vector{Float64}`.
- `Distributions.loglikelihood` returns the log likelihood of the kernel conditioned on `x` and `v`.
"""
abstract type AbstractAuxKernel end

"""
    `AuxKernel` constructs an auxiliary kernel of the type Vector{Float64} -> Vector{Distributions.UnivariateDistribution}
"""
struct AuxKernel{T} <: AbstractAuxKernel
    "Auxiliary Kernel." # auxiliary kernel function
    auxkernel::T

    AuxKernel(auxkernel) = new{typeof(auxkernel)}(auxkernel)
end

Random.rand(rng, x, k::AuxKernel, n::Int) = Random.randn(rng, x, k)[n]

function Random.randn(rng::Random.AbstractRNG, x, k::AuxKernel)
    dists = k.auxkernel(x)
    v = []
    for dist in dists
        if typeof(dist) <: Distributions.UnivariateDistribution
            append!(v, Random.rand(rng, dist))
        else
            error("Auxiliary kernel conditioned on x is not univariate.")
        end
    end
    return v
end

function Distributions.loglikelihood(k::AuxKernel, x, v)
    dists = k.auxkernel(x)
    n = length(dists)
    logp = 0
    for i in 1:n
        if typeof(dists[i]) <: Distributions.UnivariateDistribution
            logp += Distributions.loglikelihood(dists[i], v[i])
        else
            error("Auxiliary kernel conditioned on x is not univariate.")
        end
    end
    return logp
end

"""
    `ModelAuxKernel` constructs an auxiliary kernel via the `@model` macro.
"""
struct ModelAuxKernel{T,S} <: AbstractAuxKernel
    "Kernel Model."
    kmodel::T
    "Kernel Sampler"
    ksampler::S

    ModelAuxKernel(kmodel; ksampler = DynamicPPL.SampleFromPrior()) = new{typeof(kmodel), typeof(ksampler)}(kmodel, ksampler)
end

Random.rand(rng::Random.AbstractRNG, x, k::ModelAuxKernel, n::Int) = Random.randn(rng, x, k)[n]

function Random.randn(rng::Random.AbstractRNG, x, k::ModelAuxKernel)
    model = k.kmodel(x)
    vi = VarInfo(model, k.ksampler)
    v = vi[k.ksampler]
    return v
end

function Distributions.loglikelihood(k::ModelAuxKernel, x, v)
    logp = gen_logπ(k.ksampler, k.kmodel(x))(v)
    return logp
end

"""
    `PointwiseAuxKernel` constructs an auxiliary kernel for each dimension

- `kernel` should be of type (Int, Vector{Float64}) -> Distributions.Sampleable
"""
struct PointwiseAuxKernel{T} <: AbstractAuxKernel
    "Kernel."
    kernel::T

    PointwiseAuxKernel(kernel) = new{typeof(kernel)}(kernel)
end

Random.rand(rng::Random.AbstractRNG, x, k::PointwiseAuxKernel, n::Int) = Random.rand(rng, k.kernel(n, x))

Random.randn(rng::Random.AbstractRNG, x, k::PointwiseAuxKernel) = map(i->Random.rand(rng, x, k, i), 1:length(x))

Distributions.loglikelihood(k::PointwiseAuxKernel, x, v) =
sum(i->Distributions.loglikelihood(k.kernel(i, x), v[i]), 1:length(x), init=0.0)

"""
    `CompositeAuxKernel` constructs a composition of kernels

- `kernels` should be of type Vector{AbtractAuxKernel}
"""
struct CompositeAuxKernel{T} <: AbstractAuxKernel
    "Kernels."
    kernels::T

    CompositeAuxKernel(kernels) = new{typeof(kernels)}(kernels)
end

function Random.rand(rng::Random.AbstractRNG, x, k::CompositeAuxKernel, n::Int)
    v = []
    conditioned = x
    for kernel in k.kernels
        newvelem = Random.rand(rng, conditioned, kernel, n)
        append!(v,[newvelem])
        conditioned = vcat(conditioned[1:n-1], [newvelem], conditioned[n+1:end])
    end
    return v
end

function Random.randn(rng::Random.AbstractRNG, x, k::CompositeAuxKernel)
    v = []
    conditioned = x
    for kernel in k.kernels
        newv = Random.randn(rng, conditioned, kernel)
        append!(v,[newv])
        conditioned = newv
    end
    return v
end

function Distributions.loglikelihood(k::CompositeAuxKernel, x, v)
    logp = 0
    conditioned = x
    for i in 1:length(k.kernels)
        logp += Distributions.loglikelihood(k.kernels[i], conditioned, v[i])
        conditioned = v[i]
    end
    return logp
end
