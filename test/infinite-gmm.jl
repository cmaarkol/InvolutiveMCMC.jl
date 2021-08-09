using Turing, Random, Plots

# Generate some test data.
Random.seed!(1)
data = vcat(randn(10), randn(10) .- 5, randn(10) .+ 10)
data .-= mean(data)
data /= std(data);

@model function infinite_gmm(x)
    # number of mixtures
    k ~ Poisson(3.0)
    K = Int(k)+1
    # means of each mixture
    μ = tzeros(Float64, K)
    for i in 1:K
        μ[i] ~ Normal(0.0,1.0)
    end

    # Draw observation
    z = tzeros(Int, length(x))
    for j in 1:length(x)
        z[j] ~ Categorical(ones(K)/K)
        x[j] ~ Normal(μ[Int(z[j])], 1.0)
    end
end

# MCMC sampling
Random.seed!(2)
iterations = 1000
model_fun = infinite_gmm(data);
chain = sample(model_fun, SMC(), iterations);

# # Extract the number of mixture for each sample of the Markov chain.
# ks = Array(chain)[:,1] .+ 1

# histogram(ks, xlabel = "Number of clusters", legend = false)
# savefig("test/images/infinite-gmm-histogram-smc.png")

# Using iMCMC as the sampler
using DynamicPPL, LinearAlgebra, InvolutiveMCMC, InfiniteArrays

function trans_dim_gen_logπ(vi, spl, model)
    function logπ(x)::Float64
        # x_old, lj_old = vi[spl], getlogp(vi)
        empty!(vi) # empty vi (added line)
        vi[spl] = x
        model(vi, spl)
        lj = getlogp(vi)
        # empty!(vi) # empty vi (added line)
        # vi[spl] = x_old
        # setlogp!(vi, lj_old)
        return lj
    end
    return logπ
end

# generate the log likelihood of model
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(model_fun, spl) # If you want the log joint
model_loglikelihood = trans_dim_gen_logπ(log_joint, spl, model_fun)
first_sample = log_joint[spl]

# define a RJMCMC iMCMC model
kernel = CompositeAuxKernel([
    ProductAuxKernel(
        xs->vcat(
            # change mode from k=xs[1] to j
            [DiscreteUniform(max(0,xs[1]-1),xs[1]+1)],
            # a copy of xs
            map(i->Dirac(xs[i]),1:1+Int(xs[1])+1+length(data))),
        Ones(∞),
        cross_ref = true
    ),
    ProductAuxKernel(
        xs->vcat(
            # copy of mode j=xs[1]
            [Dirac(xs[1])],
            # sample means of already defined normal distributions
            map(i->Normal(xs[i],1.0),3:3+ Int(min(xs[1],xs[2]))),
            # sample means of new normal distributions
            fill(Normal(0.0,1.0),max(0,Int(xs[1]-xs[2]))),
            # Categorical distribution for the data
            fill(Categorical(ones(Int(xs[1])+1)/(Int(xs[1])+1)), length(data))),
        Ones(∞),
        cross_ref = true
    )
])
# involution
# Φ(x,y,v) = (v,y,x)
# newx(x,y,v) = v
# newv(x,y,v) = (y,x)
rjmcmc_inv = Involution(
    s->s[1+2*(1+Int(s[1])+1+length(data))+1:end],
    s->vcat(s[1+Int(s[1])+1+length(data)+1:1+2*(1+Int(s[1])+1+length(data))],s[1:1+Int(s[1])+1+length(data)]),
    z_logabsdetjac = true,
    shapev = flatv -> [flatv[1:1+1+Int(flatv[2])+1+length(data)],flatv[1+1+Int(flatv[2])+1+length(data)+1:end]]
)
model = iMCMCModel(rjmcmc_inv,kernel,model_loglikelihood,first_sample)

# generate chain
rng = MersenneTwister(1)
imcmc_chain = Chains(sample(rng,model,iMCMC(),100;discard_initial=10))

imcmc_ks = Array(imcmc_chain)[:,1] .+ 1

histogram(imcmc_ks, xlabel = "Number of clusters", legend = false)
savefig("test/images/infinite-gmm-histogram-imcmc.png")