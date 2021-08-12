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
model_fun = infinite_gmm(data);
chain = sample(model_fun, SMC(), iterations);

# Extract the number of mixture for each sample of the Markov chain.
ks = Array(chain)[:,1] .+ 1

histogram(ks, xlabel = "Number of clusters", legend = false)
savefig("test/images/infinite-gmm-histogram-smc.png")

# Using iMCMC as the sampler
using DynamicPPL, LinearAlgebra, InvolutiveMCMC, InfiniteArrays

# generate the log likelihood of model
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(model_fun, spl)
model_loglikelihood = trans_dim_gen_logπ(log_joint, spl, model_fun, empty_vns=[log_joint.metadata.μ,log_joint.metadata.z])
first_sample = log_joint[spl]

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

# define a RJMCMC iMCMC model
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
model = iMCMCModel(rjmcmc_inv,kernel,model_loglikelihood,first_sample)

# generate chain
rng = MersenneTwister(1)
imcmc_chain = sample(rng,model,iMCMC(),1000;discard_initial=10)

imcmc_ks = [state[1]+1 for state in imcmc_chain]
histogram(imcmc_ks, xlabel = "Number of clusters", legend = false)
savefig("test/images/infinite-gmm-histogram-imcmc.png")