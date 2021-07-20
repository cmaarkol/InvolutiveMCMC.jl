module InvolutiveMCMC

using AbstractMCMC: AbstractMCMC
using Distributions: Distributions
using Bijectors: Bijectors, ADBackend

using Random: Random

export iMCMCModel, iMCMC, Involution

# reexports
using AbstractMCMC: sample
export sample

include("abstractmcmc.jl")
include("model.jl")
include("interface.jl")

end # module
