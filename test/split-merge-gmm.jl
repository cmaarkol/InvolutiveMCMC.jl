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
        μ[i] ~ Normal(0.0,1.0)
        v[i] ~ InverseGamma(1.0, 100.0)
    end
    # weight of each mixture
    w ~ Dirichlet(ones(K)/K)

    # Draw observation
    z = tzeros(length(ds))
    for j in 1:length(ds)
        z[j] ~ Categorical(w)
        ds[j] ~ Normal(μ[Int(z[j])], sqrt(v[Int(z[j])]))
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
            newμ[i] ~ Normal(μs[i], 1.0)
            newv[i] ~ InverseGamma(1.0, 100.0)
        else
            # sample means and variances of new normals
            newμ[i] ~ Normal(0.0, 10.0)
            newv[i] ~ InverseGamma(1.0, 100.0)
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

"""
    Sample using iMCMC with the split-merge model
"""

# kernel
@model function split_merge(xs)
    # unpack xs
    k = xs[1]
    K = Int(k)+1
    μs = xs[2:K+1]
    vs = xs[K+2:2*K+1]
    ws = xs[2*K+2:3*K+1]
    zs = xs[3*K+2:end]

    split ~ Bernoulli(3/(K+2))
    cluster_to_split = 0
    u1, u2, u3 = zeros(3)
    cluster_to_merge = 0
    reloc = zeros(length(data))
    if split
        # split
        cluster_to_split ~ DiscreteUniform(1,K)
        u1 ~ Beta(2,2)
        u2 ~ Beta(2,2)
        u3 ~ Beta(1,1)
        cluster_to_merge ~ DiscreteUniform(0,0)
        # relocation of z
        for i in 1:length(data)
            reloc[i] ~ Bernoulli(1-u1)
        end
    else
        # merge
        cluster_to_split ~ DiscreteUniform(0,0)
        u1 ~ Beta(1,1)
        u2 ~ Beta(1,1)
        u3 ~ Beta(1,1)
        cluster_to_merge ~ DiscreteUniform(1,K-1)
        # relocation of z
        for i in 1:length(data)
            reloc[i] ~ Bernoulli(0.5)
        end
    end
end
smkernel = ModelAuxKernel(split_merge)

# involution

# unpack state
function unpackstate(state)
    k = state[1]
    K = Int(k)+1
    μs = state[2:K+1]
    vs = state[K+2:2*K+1]
    ws = state[2*K+2:3*K+1]
    zs = state[3*K+2:3*K+1+length(data)]
    split, cluster_to_split, u1, u2, u3, cluster_to_merge = state[3*K+1+length(data)+1:end-length(data)]
    reloc = state[end-length(data)+1:end]
    return k, K, μs, vs, ws, zs, Bool(split), Int(cluster_to_split), u1, u2, u3, Int(cluster_to_merge), reloc
end

function newx(state)
    # unpack state
    k, K, μs, vs, ws, zs, split, cluster_to_split, u1, u2, u3, cluster_to_merge, reloc = unpackstate(state)

    if split
        # compute new weight, mean and variance
        splitμ = μs[cluster_to_split]
        splitv = vs[cluster_to_split]
        splitw = ws[cluster_to_split]
        neww1 = splitw*u1
        neww2 = splitw*(1-u1)
        newμ1 = splitμ - u2*sqrt(splitv)*sqrt(neww2/neww1)
        newμ2 = splitμ + u2*sqrt(splitv)*sqrt(neww1/neww2)
        newv1 = u3*(1-u2^2)*splitv*(splitw/neww1)
        newv2 = (1-u3)*(1-u2^2)*splitv*(splitw/neww2)
        # put new mixture in newx
        k += 1
        μs[cluster_to_split] = newμ1
        vs[cluster_to_split] = newv1
        ws[cluster_to_split] = neww1
        push!(μs, newμ2)
        push!(vs, newv2)
        push!(ws, neww2)
        for i in 1:length(data)
            if zs[i] == cluster_to_split
                if Bool(reloc[i])
                    zs[i] = k+1
                end
            end
        end
    else
        # merge cluster_to_merge with the last one
        w1 = ws[cluster_to_merge]
        w2 = ws[K]
        μ1 = μs[cluster_to_merge]
        μ2 = μs[K]
        v1 = vs[cluster_to_merge]
        v2 = vs[K]
        neww = w1+w2
        newμ = (w1*μ1 + w2*μ2)/neww
        newv = (w1*(μ1^2+v1)+w2*(μ2^2+v2))/neww-newμ^2
        # put new mixture in place of cluster_to_merge
        k -= 1
        ws[cluster_to_merge] = neww
        μs[cluster_to_merge] = newμ
        vs[cluster_to_merge] = newv
        ws = ws[1:K-1]
        μs = μs[1:K-1]
        vs = vs[1:K-1]
        # relocate z
        for i in 1:length(data)
            if zs[i] == K
                zs[i] = cluster_to_merge
            end
        end
    end
    return vcat(k, μs, vs, ws, zs)
end

function newv(state)
    # unpack state
    k, K, μs, vs, ws, zs, split, cluster_to_split, u1, u2, u3, cluster_to_merge, reloc = unpackstate(state)

    if split
        # split -> merge should gives us the orginal x
        split = false
        cluster_to_merge = cluster_to_split
        cluster_to_split = 0
        # u1, u2, u3 = fill(0.5, 3)
    else
        # merge -> split should gives us the orginal x

        # making sure μs[loweri] < μs[upperi]
        loweri = cluster_to_merge
        upperi = K
        if μs[loweri] > μs[upperi]
            loweri = K
            upperi = cluster_to_merge
        end

        μ1 = μs[loweri]
        μ2 = μs[upperi]
        w1 = ws[loweri]
        w2 = ws[upperi]
        v1 = vs[loweri]
        v2 = vs[upperi]
        neww = w1+w2
        newμ = (w1*μ1 + w2*μ2)/neww
        newv = (w1*(μ1^2+v1)+w2*(μ2^2+v2))/neww-newμ^2

        split = true
        cluster_to_split = cluster_to_merge
        u1 = w1/(neww)
        u2 = (newμ-μ1)/(sqrt(newv*w2/w1))
        u3 = (v1*w1)/((1-u2^2)*newv*neww)
        cluster_to_merge = 0
        for i in 1:length(data)
            if zs[i] == K
                reloc[i] = true
            else
                reloc[i] = false
            end
        end
    end
    return vcat(split, cluster_to_split, u1, u2, u3, cluster_to_merge, reloc)
end

sm = Involution(
    newx,
    newv,
    z_logabsdetjac = true
)

# generate the log likelihood of model
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(model_gmm, spl)
model_lll = trans_dim_gen_logπ(log_joint, spl, model_gmm)
first_sample = log_joint[spl]

# split-merge model
smmodel = iMCMCModel(sm,smkernel,model_lll,first_sample)

# generate chain
rng = MersenneTwister(12)
sm_imcmc_chain = sample(rng,smmodel,iMCMC(),1000)
sm_imcmc_ks = [state[1]+1 for state in sm_imcmc_chain]
histogram(sm_imcmc_ks, xlabel = "Number of clusters", legend = false)
savefig("test/images/split-merge-gmm-histogram-imcmc-sm.png")
