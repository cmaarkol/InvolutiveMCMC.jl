using InvolutiveMCMC
using Turing, Distributions, Random, LinearAlgebra, MCMCChains, StatsPlots

filename = "gdemo"
imagepath = "test/images/"

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
  s² ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s²))
  x ~ Normal(m, sqrt(s²))
  y ~ Normal(m, sqrt(s²))
end
gdemomodel = gdemo(1.5, 2)

#  Run sampler, collect results
inf = "hmc"
rng1 = MersenneTwister(1)
hmcchn = sample(rng1, gdemomodel, HMC(0.1, 5), 1000)
hmcp = plot(hmcchn)
savefig(join([imagepath, filename, "-", inf, ".png"]))

# define the iMCMC model
inf = "imcmc"
rng2 = MersenneTwister(2)
ADMH = ADInvolution(
    (x, v)->(v, x),
    s->(s[1:Int(end/2)], s[Int(end/2)+1:end])
)
kernel = PointwiseAuxKernel((n, t)-> n == 1 ? Truncated(Normal(t[1],1), 0, 10) : Normal(t[n], 1.0))
model = iMCMCModel(ADMH, kernel, gdemomodel)
imcmcsamples = sample(rng2, model, iMCMC(), 1000)
imcmcchn = Chains(convert(Matrix{Float64}, reduce(hcat, imcmcsamples)'), [:s², :m])
imcmcp = plot(imcmcchn)
savefig(join([imagepath, filename, "-", inf, ".png"]))
