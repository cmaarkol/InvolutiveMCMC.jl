# Load libraries.
using Turing, StatsPlots, Random

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
gif(make_animation(m_set, s_set, Ns), "test/images/04-hidden-markov-model-hmc.gif", fps = 15)

# Using iMCMC as the sampler
using DynamicPPL, LinearAlgebra, InvolutiveMCMC

# generate the log likelihood of model
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(hmm_model, spl) # If you want the log joint
model_loglikelihood = Turing.Inference.gen_logπ(log_joint, spl, hmm_model)
first_sample = log_joint[spl]

# define the iMCMC model
mh = Involution(s->s[Int(end/2)+1:end],s->s[1:Int(end/2)])
# proposal distribution for parameters
kernel = ProductAuxKernel(
    vcat(
        fill(AuxKernel(T_i->Dirichlet(ones(3)/3)), 3),
        fill(AuxKernel(m_i->Normal(m_i,1)),3),
        fill(AuxKernel(s_i->Categorical(ones(3)/3)),30)
    ),
    vcat([3,3,3],ones(Int,33))
)
model = iMCMCModel(mh,kernel,model_loglikelihood,first_sample)

# generate chain
rng = MersenneTwister(1)
imcmc_chn = Array(Chains(sample(rng,model,iMCMC(),1000;discard_initial=10)))

imcmc_m_set = [imcmc_chn[i,:][10:12] for i in 1:size(imcmc_chn,1)]
imcmc_s_set = [Int.(imcmc_chn[i,:][13:end]) for i in 1:size(imcmc_chn,1)]

gif(make_animation(imcmc_m_set, imcmc_s_set, 1:size(imcmc_chn,1)), "test/images/04-hidden-markov-model-imcmc.gif", fps = 15)
