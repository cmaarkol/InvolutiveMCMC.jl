# internal model structure consisting of prior and log-likelihood function

struct iMCMCModel{I,A,L} <: AbstractMCMC.AbstractModel
    "Involution."
    involution::I
    "Auxiliary kernel."
    auxiliary_kernel::A
    "Log likelihood function."
    loglikelihood::L
end

function iMCMCModel(involution, auxiliary_kernel, loglikelihood)
    return iMCMCModel{typeof(involution),typeof(auxiliary_kernel),typeof(loglikelihood)}(involution, auxiliary_kernel, loglikelihood)
end

# obtain involution
# TODO: use Bijectors.jl for involutions
involution(model::iMCMCModel) = model.involution

# compute the log Jacobian determinant of the involution of a sample-auxiliary pair
logabsdetjac(model::iMCMCModel, x, v) = 0

# obtain auxiliary_kernel
auxiliary_kernel(model::iMCMCModel) = model.auxiliary_kernel

# evaluate the loglikelihood of a sample
Distributions.loglikelihood(model::iMCMCModel, x) = model.loglikelihood(x)
