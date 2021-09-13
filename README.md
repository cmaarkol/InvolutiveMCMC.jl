# InvolutiveMCMC.jl

Julia implementation of involutive MCMC.

## Overview

This package implements involutive MCMC in the Julia language, as described in
[Neklyudov et al., 2020](https://arxiv.org/pdf/2006.16653.pdf).

Involutive MCMC "provides a unified view of many known MCMC algorithms, which facilitates the derivation of powerful extensions." (Neklyudov et al., 2020)

### Metropolis-Hastings (MH)

We illustrate how to sample from simple model given in [Turing's getting started webpage](https://turing.ml/dev/docs/using-turing/get-started) using involutive MCMC.

We first set up the environment.
```julia
using InvolutiveMCMC
using Turing, Distributions, Random, LinearAlgebra, MCMCChains, StatsPlots
```

Then, we define the Turing model `gdemomodel` which define a simple Normal model with unknown mean and variance.
```julia
@model function gdemo(x, y)
  s² ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s²))
  x ~ Normal(m, sqrt(s²))
  y ~ Normal(m, sqrt(s²))
end
gdemomodel = gdemo(1.5, 2)
```

Now, we define `model`, a `iMCMCModel` for Metropolis-Hastings with arguments:
- The swap involution `ADMH`.
- The auxiliary kernel `kernel` that returns the truncated normal distribution `Truncated(Normal(s²,1), 0, 10)` for the first component and `Normal(m, 1.0)` for the second given variance `s²` and mean `m`.
- The Turing model `gdemomodel`.
```julia
ADMH = ADInvolution(
    (x, v)->(v, x),
    s->(s[1:Int(end/2)], s[Int(end/2)+1:end])
)
kernel = PointwiseAuxKernel((n, t)-> n == 1 ? Truncated(Normal(t[1],1), 0, 10) : Normal(t[n], 1.0))
model = iMCMCModel(ADMH, kernel, gdemomodel)
```

We sample the posterior of `gdemomodel` by running `model` and plot it.
```julia
imcmcsamples = sample(rng2, model, iMCMC(), 1000)
imcmcchn = Chains(convert(Matrix{Float64}, reduce(hcat, imcmcsamples)'), [:s², :m])
imcmcp = plot(imcmcchn)
```
![gdemo-imcmc](test/images/gdemo-imcmc.png)

The complete file can be found in [gdemo.jl](test/gdemo.jl).

## Type Hierarchy

### iMCMCModel

The type `iMCMCModel` is a subtype of `AbstractMCMC.AbstractModel`.

```julia
iMCMCModel(
    involution,
    auxiliary_kernel,
    turing_model;
    sampler = DynamicPPL.SampleFromPrior()
)
```

### AbstractInvolution

The abstract type `AbstractInvolution` subsumes any types of involutions and is a subtype of `Bijectors.ADBijector`, which allows us to use the `logabsdetjac` function.

The following methods need to be implemented for each `AbstractInvolution` type:
- `involution(i::AbstractInvolution, x, v)`
    + Runs the involution `i` on the state `(x, v)` and returns the result.
- `logabsdetjac(i::AbstractInvolution, x, v)::Float64`
    + Return the log absolute value of the Jacobian determinant of the involution `i` at `(x,v)`.

#### Involution

The type `Involution` constructs an involution with an explicit logabsdetjac.

```julia
Involution(involution, logabsdetjac)

# Metropolis-Hasting
mh = Involution((x, v)->(v, x), (x, v)->0.0)
```

#### ADInvolution

The type `ADInvolution` constructs an involution without explicit logabsdetjac but with a `state` function that maps the state `s` to the tuple `(x,v)`.

```julia
ADInvolution(involution, state)

# Metropolis-Hasting
admh = ADInvolution(
    (x, v)->(v, x),
    s->(s[1:Int(end/2)], s[Int(end/2)+1:end])
)
```

#### CompositeInvolution

The type `CompositeInvolution` constructs an involution that is a composition of functions.

```julia
CompositeInvolution(
    involutionMap,
    logabsdetjacMap;
    init = (rng, x) -> 1,
    constraint = (n, x) -> n ≤ length(x),
    next = n -> n+1)
)
```

A `iMCMCModel` with a `CompositeInvolution` proposes the next sample as follows.
- initialise `n` as `init(rng, sample)`
- while `constraint(n, sample)`
  - performs an NPiMCMC step with involution `involutionMap(n)` and `logabsdetjacMap(n)` and obtain the `nextsample`
  - update `sample` as `nextsample`
  - update `n` as `next(n)`

The Gibbs sampler and LMH can be expressed as `CompositeInvolution`s.
```julia
# Gibbs
gibbs = CompositeInvolution(
    n ->
        ((x, v) ->
            (vcat(x[1:n-1], v[n], x[n+1:end]),
                vcat(v[1:n-1], x[n], v[n+1:end]))),
    n -> (x, v) -> 0.0),
    init = (rng, x) -> 1,
    constraint = (n, x) -> n ≤ length(x),
    next = n -> n+1
)

# Lightweight Metropolis-Hastings
lmh = CompositeInvolution(
    n ->
        ((x, v) ->
            (vcat(x[1:n-1], v[n], x[n+1:end]),
                vcat(v[1:n-1], x[n], v[n+1:end]))),
    n -> (x, v) -> 0.0,
    init = (rng, x) ->
        rand(rng, DiscreteUniform(1, length(x))),
    constraint = (n, x) -> n < Inf,
    next = n -> Inf
)
```

#### Bijection

The type `Bijection` constructs an involution with a bijection which is not neccessarily involutive.

```julia
Bijection(bijection, invbijection, logabsdetjac)

# shift bijection
shift = Bijection((x, v)->(v.+1, x), (x, v)->(v, x.-1), (x, v)->0.0)
```

To see why this correctly defines an involution, read the AD trick.

### AbstractAuxKernel

The abstract type `AbstractAuxKernel` subsumes any types of auxiliary kernels.

The following methods need to be implemented for each `AbstractAuxKernel` type:
- `Random.rand(rng::Random.AbstractRNG, x, k::AbstractAuxKernel, n::Int)`
    + Sample the distribution specified by `k(x)` and return the `n`-th component.
- `Random.randn(rng::Random.AbstractRNG, x, k::AbstractAuxKernel)`
    + Sample the distribution specified by `k(x)` and return the result.
- `Distributions.loglikelihood(k::AbstractAuxKernel, x, v)`
    + Return the log likelihood of `v` in the distribution specified by `k(x)`.

#### AuxKernel

The type `AuxKernel` constructs an auxiliary kernel with a `auxkernel` function of type `Vector{Float64} -> Vector{Distributions.UnivariateDistribution}`.

```julia
AuxKernel(auxkernel)

# Normal
kernel = AuxKernel(p -> [Normal(p[1],1)])
```

#### ModelAuxKernel

The type `ModelAuxKernel` constructs an auxiliary kernel using a Turing model.

```julia
ModelAuxKernel(kmodel; ksampler = DynamicPPL.SampleFromPrior())

@model function dis1norminf(x)
    n = length(x)-1
    k ~ Poisson(n-1)
    y = tzeros(Float64, n)
    for i in 1:n
        y[i] ~ Normal(x[i+1], 0.1)
    end
end
dis1norminfk = ModelAuxKernel(dis1norminf)
```

#### PointwiseAuxKernel

The type `PointwiseAuxKernel` constructs an auxiliary kernel of type `(Int, Vector{Float64} -> Distributions.Sampleable` where each component is specified.

```julia
PointwiseAuxKernel(kernel)

pointwisek = PointwiseAuxKernel(
    (n, t) ->
        n == 1 ?
        DiscreteUniform(max(0,t[n]-1),t[n]+1) :
        Normal(t[n], 1.0)
)
```

#### CompositeAuxKernel

The type `CompositeAuxKernel` constructs an auxiliary kernel that is a composition of kernels.

```julia
CompositeAuxKernel(kernels)

compositek = CompositeAuxKernel([dis1norminfk,pointwisek])
```

## Bibliography

Neklyudov, K., Welling, M., Egorov. E., Vetrov D. Involutive MCMC: a unifying frmaework. Proceedings of the 37th International Conference on Machine Learning, PMLR 119, 2020.
