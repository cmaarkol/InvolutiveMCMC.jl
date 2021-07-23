# internal model structure consisting of prior and log-likelihood function

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

# Implement involution Φ(x,v) = (newx,newv)
struct Involution{T, A, AD} <: Bijectors.ADBijector{AD, 0}
    "New sample." # function that maps (x,v) to newx, i.e. π1∘Φ
    newx::T
    "New auxiliary." # function that maps (x,v) to newv, i.e. π2∘Φ
    newv::A

    # TODO: implement a way to check whether the function is involutive, i.e. newx(newx(x,v), newv(x,v)), newv((newx(x,v), newv(x,v))) == x, v
    # Φ == Array ∘ (s->map(f->f(s), [newx, newv]))
end

# ADBackend() returns ForwardDiffAD, which means we use ForwardDiff.jl for AD by default
function Involution(newx, newv)
    return Involution{typeof(newx), typeof(newv), ADBackend()}(newx,newv)
end

(b::Involution)(s) = vcat(b.newx(s),b.newv(s))
(ib::Bijectors.Inverse{<:Involution})(s) = vcat(b.newx(s),b.newv(s))

# run involution
involution(model::iMCMCModel, s) = model.involution.newx(s), model.involution.newv(s)

# compute the log Jacobian determinant of the involution of the state `(x,v)`
function Bijectors.logabsdetjac(model::iMCMCModel, x, v)
    flatv = collect(Iterators.flatten(v))
    Bijectors.logabsdetjac(model.involution, vcat(x, flatv))
end

# obtain auxiliary_kernel conditioned on x
auxiliary_kernel(model::iMCMCModel) = model.auxiliary_kernel

# evaluate the loglikelihood of a sample
Distributions.loglikelihood(model::iMCMCModel, x) = model.loglikelihood(x)

# obtain `model`'s prior
prior(model::iMCMCModel) = model.prior
