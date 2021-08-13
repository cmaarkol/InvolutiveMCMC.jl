using Turing, Random, Plots

# Generate some test data.
Random.seed!(1)
data = vcat(randn(10), randn(10) .- 5, randn(10) .+ 10)
data .-= mean(data)
data /= std(data);

@model function infinite_gmm(x)
    # number of mixtures
    k ~ Poisson(3.0)
    # println("k = ",k)
    K = Int(k)+1
    # means of each mixture
    μ = tzeros(Float64, K)
    for i in 1:K
        μ[i] ~ Normal(0.0,1.0)
        # println("μ[",i,"] = ", μ[i])
    end

    # Draw observation
    z = tzeros(length(x))
    for j in 1:length(x)
        z[j] ~ Categorical(ones(K)/K)
        # println("z[",j,"] = ", z[j])
        x[j] ~ Normal(μ[Int(z[j])], 1.0)
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
savefig("test/images/infinite-gmm-histogram-smc.png")

"""
    Sample using iMCMC
"""

# import packages
using DynamicPPL, LinearAlgebra, InvolutiveMCMC, InfiniteArrays

# generate the log likelihood of model
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(model_gmm, spl)
model_lll = trans_dim_gen_logπ(log_joint, spl, model_gmm)
first_sample = log_joint[spl]

"""
    First attempt
"""

# kernel using CompositeAuxKernel and ProductAuxKernel
mode_kernel = ProductAuxKernel(
    xs -> [DiscreteUniform(max(0,xs[1]-1),xs[1]+1)],
    [1],
    cross_ref = true,
    copy_info = (f=identity,n=2)
)
auxvar_kernel = ProductAuxKernel(
    xs->vcat(
        # sample means of already defined normal distributions
        map(i->Normal(xs[i],1.0),3:3+ Int(min(xs[1],xs[2]))),
        # sample means of new normal distributions
        fill(Normal(0.0,1.0),max(0,Int(xs[1]-xs[2]))),
        # Categorical distribution for the data
        fill(Categorical(ones(Int(xs[1])+1)/(Int(xs[1])+1)), 30)),
    Ones(∞),
    cross_ref = true,
    copy_info = (f=xs->[xs[1]],n=1)
)
kernel = CompositeAuxKernel([mode_kernel,auxvar_kernel])

# involution
length_of_m_mixture_sample = m->1+m+length(data)
rjmcmc_inv = Involution(
    # ((k,xs),(j,k,xs),(j,vs)) -> (j,vs)
    s->s[1+2*(1+length_of_m_mixture_sample(Int(s[1])))+1:end],
    # ((k,xs),(j,k,xs),(j,vs)) -> ((k,j,vs),(k,xs))
    s->vcat(
        s[1+length_of_m_mixture_sample(Int(s[1]))+2],
        s[1+length_of_m_mixture_sample(Int(s[1]))+1],
        s[1+length_of_m_mixture_sample(Int(s[1]))+3:1+2*(1+length_of_m_mixture_sample(Int(s[1])))],
        s[1:1+length_of_m_mixture_sample(Int(s[1]))]),
    z_logabsdetjac = true,
    shapev = flatv -> [flatv[1:1+1+Int(flatv[2])+1+length(data)],flatv[1+1+Int(flatv[2])+1+length(data)+1:end]]
)
# model
model = iMCMCModel(rjmcmc_inv,kernel,model_lll,first_sample)

# generate samples and plot histogram
rng = MersenneTwister(1)
imcmc_chain = sample(rng,model,iMCMC(),1000;discard_initial=10)
imcmc_ks = [state[1]+1 for state in imcmc_chain]
histogram(imcmc_ks, xlabel = "Number of clusters", legend = false)
savefig("test/images/infinite-gmm-histogram-imcmc.png")


"""
    A very simple model which works quite well
"""

# kernel
@model function simple_kernel(xs)
    # unpack xs
    k = xs[1]
    K = Int(k)+1
    μs = xs[2:K+1]
    zs = xs[K+2:end]

    newk ~ DiscreteUniform(max(0,k-1),k+1)
    newK = Int(newk)+1
    newμ = tzeros(Float64, newK)
    for i in 1:newK
        if i <= length(μs)
            # sample means of existing normals
            newμ[i] ~ Normal(μs[i],1.0)
        else
            # sample means of new normals
            newμ[i] ~ Normal(0.0,1.0)
        end
    end

    newz = tzeros(length(zs))
    for j in 1:length(zs)
        newz[j] ~ Categorical(ones(newK)/newK)
    end

    return vcat(newk, newμ, newz)
end
skernel = ModelAuxKernel(simple_kernel)

# involution
# split_points(s, n) return the splitting point of s where n is the length of the data
function split_point(s, n)
    k1 = s[1]
    return 1+Int(k1)+1+n
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
savefig("test/images/infinite-gmm-histogram-imcmc-s.png")
