# Load libraries.
using InvolutiveMCMC
using Turing, StatsPlots, Random

filename = "04-hidden-markov-model"
imagepath = "test/images/"

# Turn off progress monitor.
Turing.setprogress!(false);

# Set a random seed and use the forward_diff AD mode.
Random.seed!(12345678);

# Define the emission parameter.
y = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
N = length(y);  K = 3;

# Plot the data we just made.
plot(y, xlim = (0,30), ylim = (-1,5), size = (500, 250))

# Turing model definition.
@model BayesHmm(y, K) = begin
    # Get observation length.
    N = length(y)

    # State sequence.
    s = tzeros(Int, N)

    # Emission matrix.
    m = Vector(undef, K)

    # Transition matrix.
    T = Vector{Vector}(undef, K)

    # Assign distributions to each element
    # of the transition matrix and the
    # emission matrix.
    for i = 1:K
        T[i] ~ Dirichlet(ones(K)/K)
        m[i] ~ Normal(i, 0.5)
    end

    # Observe each point of the input.
    s[1] ~ Categorical(K)
    y[1] ~ Normal(m[s[1]], 0.1)

    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1]]))
        y[i] ~ Normal(m[s[i]], 0.1)
    end
end;

hmm_model = BayesHmm(y, 3)

inf = "hmc&gibbs"
g = Gibbs(HMC(0.01, 50, :m, :T), PG(120, :s))
chn = sample(hmm_model, g, 1000);

# Extract our m and s parameters from the chain.
m_set = [MCMCChains.group(chn, :m).value[i,:] for i in 1:1000]
s_set = [Int.(MCMCChains.group(chn, :s).value[i,:]) for i in 1:1000]

# Iterate through the MCMC samples.
Ns = 1:length(chn)

# Make an animation.
function make_animation(m_set, s_set, Ns)
    return @animate for i in Ns
        m = m_set[i]
        s = s_set[i]
        emissions = m[s]

        p = plot(y, chn = :red,
            size = (500, 250),
            xlabel = "Time",
            ylabel = "State",
            legend = :topright, label = "True data",
            xlim = (0,30),
            ylim = (-1,5));
        plot!(emissions, color = :blue, label = "Sample $i")
    end every 3
end
gif(make_animation(m_set, s_set, Ns), join([imagepath, filename, "-", inf, ".gif"]), fps = 15)

# Using iMCMC as the sampler
inf = "imcmc"
ADMH = ADInvolution(
    (x, v)->(v, x),
    s->(s[1:Int(end/2)], s[Int(end/2)+1:end])
)
@model function turingkernel(t)
    s = tzeros(Int, N)
    m = Vector(undef, 3)
    T = Vector{Vector}(undef, 3)

    for i = 1:3
        T[i] ~ Dirichlet(ones(3)/3)
        m[i] ~ Normal(t[i+9], 1)
    end

    s[1] ~ Categorical(3)
    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1]]))
    end
end
kernel = ModelAuxKernel(turingkernel)
model = iMCMCModel(ADMH, kernel, hmm_model)

# generate chain
rng2 = MersenneTwister(2)
imcmcsamples = sample(rng2, model, iMCMC(), 1000)

imcmcmat = convert(Matrix{Float64}, reduce(hcat, imcmcsamples)')
imcmc_m_set = [imcmcmat[i,:][10:12] for i in 1:size(imcmcmat,1)]
imcmc_s_set = [Int.(imcmcmat[i,:][13:end]) for i in 1:size(imcmcmat,1)]

gif(make_animation(imcmc_m_set, imcmc_s_set, 1:size(imcmcmat,1)), join([imagepath, filename, "-", inf, ".gif"]), fps = 15)
