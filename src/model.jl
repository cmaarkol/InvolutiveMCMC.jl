# internal model structure consisting of involution, auxiliary kernel, log-likelihood function and the prior.
struct iMCMCModel{I,A,L,P} <: AbstractMCMC.AbstractModel
    "Involution."
    involution::I
    "Auxiliary kernel."
    auxiliary_kernel::A
    "Log likelihood function."
    loglikelihood::L
    "Prior."
    prior::P
end

function iMCMCModel(involution, auxiliary_kernel, loglikelihood,prior)
    return iMCMCModel{typeof(involution),typeof(auxiliary_kernel),typeof(loglikelihood),typeof(prior)}(involution, auxiliary_kernel, loglikelihood, prior)
end

# Involution Φ(x,v) = (newx,newv)
struct Involution{T, A, AD} <: Bijectors.ADBijector{AD, 0}
    "New sample." # function that maps (x,v) to newx, i.e. π1∘Φ
    newx::T
    "New auxiliary." # function that maps (x,v) to newv, i.e. π2∘Φ
    newv::A

    "Zero_logabsdetjac." # flag of whether logabsdetjac is 0 (default to be false)
    z_logabsdetjac::Bool
    "Shape of sample" # function that maps a flattened x to x
    shapex
    "Shape of auxiliary" # function that maps a flattened v to v
    shapev
    # TODO: implement a way to check whether the function is involutive, i.e. newx(newx(x,v), newv(x,v)), newv((newx(x,v), newv(x,v))) == x, v
    # Φ == Array ∘ (s->map(f->f(s), [newx, newv]))
end

# ADBackend() returns ForwardDiffAD, which means we use ForwardDiff.jl for AD by default
function Involution(newx, newv; z_logabsdetjac=false, shapex=identity, shapev=identity)
    return Involution{typeof(newx), typeof(newv), ADBackend()}(newx,newv,z_logabsdetjac,shapex,shapev)
end

(b::Involution)(s) = vcat(b.newx(s),b.newv(s))
(ib::Bijectors.Inverse{<:Involution})(s) = vcat(b.newx(s),b.newv(s))

# run involution
involution(model::iMCMCModel, s) = model.involution.newx(s), model.involution.newv(s)

# compute the log Jacobian determinant of the involution of the state `(x,v)`
function Bijectors.logabsdetjac(model::iMCMCModel, x, v)
    if model.involution.z_logabsdetjac
        return 0.0
    else
        flatv = collect(Iterators.flatten(v))
        return Bijectors.logabsdetjac(model.involution, vcat(x, flatv))
    end
end

# Auxiliary kernel
abstract type AbstractAuxKernel end

struct AuxKernel{T} <: AbstractAuxKernel
    "Auxiliary Kernel." # only one auxiliary kernel function
    auxkernel::T
end

struct CompositeAuxKernel{T} <: AbstractAuxKernel
    "Composition of Auxiliary Kernels." # vector of auxiliary kernel functions
    comp_auxkernel::T
end

struct ProductAuxKernel{T} <: AbstractAuxKernel
    "Product of Auxiliary Kernels."
    prod_auxkernel::T
    "Shape"
    shape
    # if cross_ref, prod_auxkernel is of type {samples} -> Vector{Distributions}
    # if not cross_ref, prod_auxkernel is of type Vector{samples -> Distributions}
    "Cross Reference"
    cross_ref::Bool
end

AuxKernel(auxkernel) = AuxKernel{typeof(auxkernel)}(auxkernel)

CompositeAuxKernel(comp_auxkernel) = CompositeAuxKernel{typeof(comp_auxkernel)}(comp_auxkernel)

ProductAuxKernel(prod_auxkernel, shape; cross_ref=false) = ProductAuxKernel{typeof(prod_auxkernel)}(prod_auxkernel,shape,cross_ref)

function Random.rand(rng::Random.AbstractRNG, x, k::AuxKernel)
    x = typeof(x) <: AbstractVector && length(x) == 1 ? x[1] : x
    dist = k.auxkernel(x)
    if typeof(dist) <: Distributions.Sampleable
        return Random.rand(rng,dist)
    else
        error("Auxiliary kernel conditioned on x is not sampleable.")
    end
end

function Random.rand(rng::Random.AbstractRNG, x, k::CompositeAuxKernel)
    v = []
    for kernel in k.comp_auxkernel
        x = Random.rand(rng, x, kernel)
        v = vcat(v,[x])
    end
    return v
end

function Random.rand(rng::Random.AbstractRNG, x, k::ProductAuxKernel)
    if k.cross_ref
        vsample = map(
            dist->Random.rand(rng,dist),
            k.prod_auxkernel(x)
        )
    else
        vsample = map(
            (kernel,xsample)->Random.rand(rng, xsample, kernel),
            k.prod_auxkernel,
            reshape_sample(x,k.shape)
        )
    end
    flatvsample = collect(Iterators.flatten(vsample))
    return flatvsample
end

function Distributions.loglikelihood(k::AuxKernel, x, v)
    x = typeof(x) <: AbstractVector && length(x) == 1 ? x[1] : x
    dist = k.auxkernel(x)
    if typeof(dist) <: Distributions.Sampleable
        return Distributions.loglikelihood(dist, v)
    else
        error("Auxiliary kernel conditioned on x is not sampleable.")
    end
end

Distributions.loglikelihood(k::CompositeAuxKernel, x, v) = sum(map(
    (kernel, xsample, vsample) -> Distributions.loglikelihood(kernel, xsample, vsample),
    k.comp_auxkernel,
    vcat([x],v[1:end-1]),
    v
))

function Distributions.loglikelihood(k::ProductAuxKernel, x, v)
    if k.cross_ref
        return sum(map(
            (dist,vsample) -> Distributions.loglikelihood(dist,vsample),
            k.prod_auxkernel(x),
            reshape_sample(v,k.shape)
        ))
    else
        return sum(map(
            (kernel, xsample, vsample) -> Distributions.loglikelihood(kernel, xsample, vsample),
            k.prod_auxkernel,
            reshape_sample(x,k.shape),
            reshape_sample(v,k.shape)
        ))
    end
end

# obtain auxiliary_kernel conditioned on x
auxiliary_kernel(model::iMCMCModel) = model.auxiliary_kernel

# evaluate the loglikelihood of a sample
Distributions.loglikelihood(model::iMCMCModel, x) = model.loglikelihood(
    typeof(x) <: AbstractVector ? collect(Iterators.flatten(x)) : x
)

# obtain `model`'s prior
prior(model::iMCMCModel) = model.prior
