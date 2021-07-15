module InvolutiveMCMC

using AbstractMCMC: AbstractMCMC
using Distributions: Distributions
using Bijectors: Bijectors

using Random: Random

export iMCMCModel, iMCMC

# reexports
using AbstractMCMC: sample
export sample

include("abstractmcmc.jl")
include("model.jl")
include("interface.jl")

end # module
