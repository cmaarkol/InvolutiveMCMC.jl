using InvolutiveMCMC
using Turing, Random, Distributions, Bijectors

rng = MersenneTwister(1)
iterations = 100

"""Involutions"""

# Metropolis-Hastings
ADMH = ADInvolution(
    (x, v)->(v, x),
    s->(s[1:Int(end/2)], s[Int(end/2)+1:end])
)
MH = Involution((x, v)->(v, x), (x, v)->0.0)

"""Kernels"""

@model function norminf(x)
    n = length(x)
    y = tzeros(Float64, n)
    for i in 1:n
        y[i] ~ Normal(x[i], 1)
    end
end
norminfk = ModelAuxKernel(norminf)

"""Models with fixed dimensions"""

# model with one sample
@model function fixed1(x)
    y ~ Normal(x,1)
    # var"y" ~ Normal(x,1)
end

f1 = iMCMCModel(MH, norminfk, fixed1(2))
f1chain = sample(rng, f1, iMCMC(), iterations)

adf1 = iMCMCModel(ADMH, norminfk, fixed1(2))
adf1chain = sample(rng, adf1, iMCMC(), iterations)

# model with samples of different dimensions
@model function fixed2(x)
    y ~ Normal(x,1)
    z ~ MvNormal([9,3],[1.0 0.0; 0.0 1.0])
end

f2 = iMCMCModel(MH, norminfk, fixed2(2))
f2chain = sample(rng, f2, iMCMC(), iterations)

adf2 = iMCMCModel(ADMH, norminfk, fixed2(2))
adf2chain = sample(rng, adf2, iMCMC(), iterations)

# model with samples of different types
@model function fixed3(x)
    n = 3
    y = tzeros(Float64, n)
    for i in 1:n
        y[i] ~ Normal(x,1)
    end
    z ~ Bernoulli(0.3)
end

# auxiliary kernel for fixed3
@model function norm3bern1(x)
    n = 3
    y = tzeros(Float64, n)
    for i in 1:n
        y[i] ~ Normal(x[i], 1)
    end
    z ~ Bernoulli(Bool(x[n+1]) ? 0.2 : 0.8)
end
norm3bern1k = ModelAuxKernel(norm3bern1)

f3 = iMCMCModel(MH, norm3bern1k, fixed3(2))
f3chain = sample(rng, f3, iMCMC(), iterations)

adf3 = iMCMCModel(ADMH, norm3bern1k, fixed3(2))
adf3chain = sample(rng, adf3, iMCMC(), iterations)
