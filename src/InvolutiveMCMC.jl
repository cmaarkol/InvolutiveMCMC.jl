module InvolutiveMCMC

using AbstractMCMC: AbstractMCMC
using Distributions: Distributions
using Bijectors: Bijectors, ADBackend
using Turing

using Random: Random

export iMCMCModel, iMCMC, Involution, AuxKernel, CompositeAuxKernel, ProductAuxKernel, trans_dim_gen_logπ

# reexports
using AbstractMCMC: sample
export sample

include("abstractmcmc.jl")
include("model.jl")
include("interface.jl")

end # module
