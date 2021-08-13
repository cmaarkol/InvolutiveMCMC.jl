using Turing, Random, Plots
using DynamicPPL, LinearAlgebra, InvolutiveMCMC, InfiniteArrays

"""
    The birth-death (split-merge) model

This is introduced by Richardson & Green. 1997 specifically for an infinite Gaussian mixture model. We implement the version that is given in Cusumano-Towner et al. Automating Involutive MCMC using Probabilistic and
Differentiable Programming. 2020.
"""

# Generate some test data.
Random.seed!(1)
data = vcat(randn(10), randn(10) .- 5, randn(10) .+ 10)
data .-= mean(data)
data /= std(data);

@model function infinite_gmm(ds)
    # number of mixtures
    k ~ Poisson(3.0)
    K = Int(k)+1

    # mean and variance of each mixture
    μ = tzeros(Float64, K)
    v = tzeros(Float64, K)
    for i in 1:K
        μ[i] ~ Normal(0.0,10.0)
        v[i] ~ InverseGamma(1.0, 10.0)
    end
    # weight of each mixture
    w ~ Dirichlet(ones(K)/K)

    # Draw observation
    z = tzeros(length(ds))
    for j in 1:length(ds)
        z[j] ~ Categorical(w)
        ds[j] ~ Normal(μ[Int(z[j])], v[Int(z[j])])
    end
end

# MCMC sampling
Random.seed!(2)
iterations = 1000
model_gmm = infinite_gmm(data);

"""
    Sample using SMC
"""

chain = sample(model_gmm, SMC(), iterations)
ks = Array(chain)[:,1] .+ 1
histogram(ks, xlabel = "Number of clusters", legend = false)
savefig("test/images/split-merge-gmm-histogram-smc.png")

"""
    Sample using iMCMC with the simple kernel that works very well in `infinite-gmm.jl`
"""

# generate the log likelihood of model
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(model_gmm, spl)
model_lll = trans_dim_gen_logπ(log_joint, spl, model_gmm)
first_sample = log_joint[spl]


# kernel
@model function simple_kernel(xs)
    # unpack xs
    k = xs[1]
    K = Int(k)+1
    μs = xs[2:K+1]
    vs = xs[K+2:2*K+1]
    ws = xs[2*K+2:3*K+1]
    zs = xs[3*K+2:end]

    newk ~ DiscreteUniform(max(0,k-1),k+1)
    newK = Int(newk)+1
    newμ = tzeros(Float64, newK)
    newv = tzeros(Float64, newK)
    for i in 1:newK
        if i <= length(μs)
            # sample means and variances of existing normals
            newμ[i] ~ Normal(μs[i],1.0)
            newv[i] ~ InverseGamma(1.0, 10.0)
        else
            # sample means and variances of new normals
            newμ[i] ~ Normal(0.0,10.0)
            newv[i] ~ InverseGamma(1.0, 10.0)
        end
    end
    neww ~ Dirichlet(ones(newK)/newK)

    newz = tzeros(length(zs))
    for j in 1:length(zs)
        newz[j] ~ Categorical(neww)
    end

    return vcat(newk, newμ, newv, neww, newz)
end
skernel = ModelAuxKernel(simple_kernel)

# involution
# split_points(s, n) return the splitting point of s where n is the length of the data
function split_point(s, n)
    K = Int(s[1])+1
    return 1+K+K+K+n
end
smh = Involution(
    s->s[split_point(s, length(data))+1:end],
    s->s[1:split_point(s, length(data))],
    z_logabsdetjac = true
)

# model
smodel = iMCMCModel(smh,skernel,model_lll,first_sample)

# generate chain
rng = MersenneTwister(1)
s_imcmc_chain = sample(rng,smodel,iMCMC(),1000;discard_initial=10)
s_imcmc_ks = [state[1]+1 for state in s_imcmc_chain]
histogram(s_imcmc_ks, xlabel = "Number of clusters", legend = false)
savefig("test/images/split-merge-gmm-histogram-imcmc-s.png")
