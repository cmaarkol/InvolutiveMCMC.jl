using Turing, DynamicPPL, InfiniteArrays

# Figure out how to generate the log likelihood of a posterior given by a Turing @model function.

function my_gen_logπ(vi, spl, model)
    function logπ(x)::Float64
        # append x with infinite 1s in order to allow x with different lengths
        x = vcat(x,InfiniteArrays.Ones(∞))
        # for some reasons, it takes two runs of vi to get the correct log likelihood
        vi[spl] = x
        model(vi, spl)
        lj = Turing.Inference.getlogp(vi)
        vi[spl] = x
        model(vi, spl)
        lj = Turing.Inference.getlogp(vi)
        return lj
    end
    return logπ
end

# I am still struggling with gen_logπ

# # workflow 1
# empty!(vi) # empty everything
# vi[spl] = x # does not seem to do anything
# model(vi, spl) # just another run of the model, independent of x
# vi[spl] = x # gives an error due to dimension change, but also set k according to x, but not y nor z
# DynamicPPL._empty!(vns)
# vi[spl] = x # does not set y nor z
# model(vi, spl) # a run of the model with k set but not y nor z
# vi[spl] = x # finally setting all according to x
# model(vi, spl) # a run of the model according to x

# # workflow 2 (works for np_simple)
# DynamicPPL._empty!(vns) # empty all y
# vi[spl] = x # set k but not y nor z
# model(vi, spl) # a run of the model with k set but not y nor z, I think to initialise y and z
# vi[spl] = x # actually setting k, y and z
# model(vi, spl) # a run of the model with x

# # workflow 3
# vi[spl] = x # set k and partly μ and z (but in the wrong dimension)
# model(vi, spl) # out of bound error

# # workflow 4 (works for infinite_gmm)
# DynamicPPL._empty!(vns) # empty all μ and z
# vi[spl] = x # set k only
# model(vi, spl) # run with k only
# vi[spl] = x # set k and μ and z
# model(vi, spl) # run with x


function empty_gen_logπ(vi, spl, model, empty_vns)
    function logπ(x)::Float64
        # empty the metadata for empty_vns
        # This is required to ensure the length of unbounded variables are not fixed
        for vns in empty_vns
            DynamicPPL._empty!(vns)
        end
        # initialise the model
        vi[spl] = x # set k but not y nor z
        model(vi, spl) # a run of the model with k set but not y nor z, I think to initialise y and z
        vi[spl] = x # actually setting k, y and z
        model(vi, spl) # a run of the model with x
        lj = Turing.Inference.getlogp(vi)
        return lj
    end
    return logπ
end

function np_gen_logπ(vi, spl, model)
    function logπ(x)::Float64
        # number of variables in vi
        n = length(vi.metadata)
        # set the values for variables one by one
        for i in 1:n
            # empty data stored in vi for variables i+1:n
            for j in i+1:n
                DynamicPPL._empty!(vi.metadata[j])
            end
            # set variable at i according to x
            vi[spl] = x
            # run model with new value for variable i, which initialises the dimension for variable i+1
            model(vi, spl)
        end
        lj = Turing.Inference.getlogp(vi)
        return lj
    end
    return logπ
end

# simple model

@model function simple(x)
    y ~ Normal(x,1)
end

@model function simple(x)
    y ~ Normal(x,1)
    z ~ Normal(9,1)
    return y
end

spl = DynamicPPL.SampleFromPrior()
s_model = simple(2)
s_lj = VarInfo(s_model, spl)
s_lll = my_gen_logπ(s_lj, spl, s_model)
true_s_lll(x) = Distributions.loglikelihood(Normal(2,1),x)
test_s(x) = s_lll(x) ≈ true_s_lll(x)
# test_s always return true means that my_gen_logπ does return the log likelihood of s_model

# model with unbounded number of dimension

@model function np_simple(x)
    k ~ Poisson(3.0)
    println("k = ", k)
    K = Int(k)+1
    y = tzeros(Float64, K)
    for i in 1:K
        y[i] ~ Normal(x,1)
        println("y[",i,"] = ", y[i])
    end
    z ~ Categorical(ones(K)/K)
    println("z = ", z)
    return K
end

spl = DynamicPPL.SampleFromPrior()
np_s_model = np_simple(2)
np_s_lj = VarInfo(np_s_model, spl)
np_s_lll = empty_gen_logπ(np_s_lj, spl, np_s_model, [np_s_lj.metadata.y])
true_np_s_lll(xs) =
    Distributions.loglikelihood(Poisson(3.0),xs[1]) +
    sum(map(x->Distributions.loglikelihood(Normal(2,1),x),xs[2:Int(xs[1])+2])) +
    Distributions.loglikelihood(Categorical(ones(Int(xs[1])+1)/(Int(xs[1])+1)),xs[Int(xs[1])+3])
test_np(x) = np_s_lll(x) ≈ true_np_s_lll(x)
# test_np always return true means that my_gen_logπ does return the log likelihood of np_s_model

# infinite Gaussian mixture model

@model function infinite_gmm(x)
    # number of mixtures
    k ~ Poisson(3.0)
    println("k = ",k)
    K = Int(k)+1
    # means of each mixture
    μ = tzeros(Float64, K)
    for i in 1:K
        μ[i] ~ Normal(0.0,1.0)
        println("μ[",i,"] = ", μ[i])
    end

    # Draw observation
    z = tzeros(length(x))
    for j in 1:length(x)
        z[j] ~ Categorical(ones(K)/K)
        println("z[",j,"] = ", z[j])
        x[j] ~ Normal(μ[Int(z[j])], 1.0)
    end
end

spl = DynamicPPL.SampleFromPrior()
data = [1,2,3]
gmm_model = infinite_gmm(data)
gmm_lj = VarInfo(gmm_model, spl)
gmm_lll = empty_gen_logπ(gmm_lj, spl, gmm_model, [gmm_lj.metadata.μ, gmm_lj.metadata.z])
function true_gmm_lll(xs,ds)
    k = Int(xs[1])
    K = k+1
    μs = xs[2:K+1]
    catdist = Categorical(ones(K)/K)
    zs = xs[K+2:end]

    pk = Distributions.loglikelihood(Poisson(3.0),k)
    pμ = sum(map(μ->Distributions.loglikelihood(Normal(0,1),μ),μs))
    pz = sum(map(z->Distributions.loglikelihood(catdist,z), zs))
    pd = sum(map((d,z)->Distributions.loglikelihood(
        Normal(μs[z],1),d
        ),ds,zs))
    return pk + pμ + pz + pd
end
test_gmm(x) = gmm_lll(x) ≈ true_gmm_lll(x,data)
# test_np always return true means that my_gen_logπ does return the log likelihood of np_s_model

# model where a variable is defined twice
@model function twotimes()
    x ~ Normal(0,1)
    x ~ Normal(10,1)
end

spl = DynamicPPL.SampleFromPrior()
tt_model = twotimes()
tt_lj = VarInfo(tt_model, spl)
tt_lll = np_gen_logπ(tt_lj, spl, tt_model)
# gives Distributions.loglikelihood(Normal(0,1),x) + Distributions.loglikelihood(Normal(10,1),x)

# model with different distributions

@model function stochastic_b(K)
    split ~ Bernoulli(0.5)
    cluster_to_split = 0
    u1, u2, u3 = zeros(3)
    cluster_to_merge = 0
    if K == 1 || split
        # we split if there is only one mixture or we sample true
        cluster_to_split ~ DiscreteUniform(1,K)
        u1 ~ Beta(2,2)
        u2 ~ Beta(2,2)
        u3 ~ Beta(1,1)
        cluster_to_merge ~ DiscreteUniform(0,0)
    else
        cluster_to_split ~ DiscreteUniform(0,0)
        u1 ~ Beta(1,1)
        u2 ~ Beta(1,1)
        u3 ~ Beta(1,1)
        cluster_to_merge ~ DiscreteUniform(1,K-1)
    end
end

spl = DynamicPPL.SampleFromPrior()
sb_model = stochastic_b(2)
sb_lj = VarInfo(sb_model, spl)
sb_lll = np_gen_logπ(sb_lj, spl, sb_model)

# models not allowed

# variable y cannot be of type Vector{Normal} and Vector{InverseGamma} at the same time
# possible solution: empty the type of meta.dists
@model function diffdist(n)
    x = tzeros(Float64, n)
    y = tzeros(Float64, n)
    for i in 1:n
        x[i] ~ Bernoulli(0.5)
        if x[i]
            y[i] ~ Normal(0,1)
        else
            y[i] ~ InverseGamma(1,1)
        end
    end
end

# cluster_to_merge is not defined if split is not true
@model function stochastic_b(K)
    split ~ Bernoulli(0.5)
    cluster_to_split = 0
    u1, u2, u3 = zeros(3)
    cluster_to_merge = 0
    if K == 1 || split
        # we split if there is only one mixture or we sample true
        cluster_to_split ~ DiscreteUniform(1,K)
        u1 ~ Beta(2,2)
        u2 ~ Beta(2,2)
        u3 ~ Beta(1,1)
    else
        cluster_to_merge ~ DiscreteUniform(1,K-1)
    end
end