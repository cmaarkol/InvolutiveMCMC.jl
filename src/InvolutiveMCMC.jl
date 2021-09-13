module InvolutiveMCMC

using AbstractMCMC: AbstractMCMC
using Distributions: Distributions
using Bijectors: Bijectors, ADBackend
using Turing, DynamicPPL

using InfiniteArrays: InfiniteArrays
using Random: Random

export iMCMCModel, iMCMC, Involution, ADInvolution, CompositeInvolution, Bijection, AuxKernel, ModelAuxKernel, PointwiseAuxKernel, CompositeAuxKernel

# reexports
using AbstractMCMC: sample
export sample

include("abstractmcmc.jl")
include("model.jl")
include("interface.jl")

end # module
