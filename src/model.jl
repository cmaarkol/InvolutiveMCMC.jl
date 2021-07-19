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

# obtain involution
# TODO: use Bijectors.jl for involutions
involution(model::iMCMCModel) = model.involution

# compute the log Jacobian determinant of the involution of a sample-auxiliary pair
logabsdetjac(model::iMCMCModel, x, v) = 0

# obtain auxiliary_kernel conditioned on x
auxiliary_kernel(model::iMCMCModel, x) = model.auxiliary_kernel(x)

# evaluate the loglikelihood of a sample
Distributions.loglikelihood(model::iMCMCModel, x) = model.loglikelihood(x)

# obtain auxiliary_kernel conditioned on x
prior(model::iMCMCModel, x) = model.prior
