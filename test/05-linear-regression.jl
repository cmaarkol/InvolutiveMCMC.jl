# Import Turing and Distributions.
using Turing, Distributions

# Import RDatasets.
using RDatasets

# Import MCMCChains, Plots, and StatPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# Functionality for splitting and normalizing the data.
using MLDataUtils: shuffleobs, splitobs, rescale!

# Functionality for evaluating the model predictions.
using Distances

# Set a seed for reproducibility.
using Random
Random.seed!(0)

# Hide the progress prompt while sampling.
Turing.setprogress!(false);

# Import the "Default" dataset.
data = RDatasets.dataset("datasets", "mtcars");

# Show the first six rows of the dataset.
first(data, 6)

size(data)

# Remove the model column.
select!(data, Not(:Model))

# Split our dataset 70%/30% into training/test sets.
trainset, testset = splitobs(shuffleobs(data), 0.7)

# Turing requires data in matrix form.
target = :MPG
train = Matrix(select(trainset, Not(target)))
test = Matrix(select(testset, Not(target)))
train_target = trainset[:, target]
test_target = testset[:, target]

# Standardize the features.
μ, σ = rescale!(train; obsdim = 1)
rescale!(test, μ, σ; obsdim = 1)

# Standardize the targets.
μtarget, σtarget = rescale!(train_target; obsdim = 1)
rescale!(test_target, μtarget, σtarget; obsdim = 1);

# Bayesian linear regression.
@model function linear_regression(x, y)
    # Set variance prior.
    σ₂ ~ truncated(Normal(0, 100), 0, Inf)
    
    # Set intercept prior.
    intercept ~ Normal(0, sqrt(3))
    
    # Set the priors on our coefficients.
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(nfeatures, sqrt(10))
    
    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    y ~ MvNormal(mu, sqrt(σ₂))
end

lr_model = linear_regression(train, train_target)

chain = sample(lr_model, NUTS(0.65), 3_000);
plot(chain)
savefig("test/images/05-linear-regression-nuts.png")

# Using iMCMC as the sampler
using DynamicPPL, LinearAlgebra, InvolutiveMCMC

# generate the log likelihood of model
spl = DynamicPPL.SampleFromPrior()
log_joint = VarInfo(lr_model, spl) # If you want the log joint
model_loglikelihood = Turing.Inference.gen_logπ(log_joint, spl, lr_model)
first_sample = log_joint[spl]

# define the iMCMC model
mh = Involution(s->s[Int(end/2)+1:end],s->s[1:Int(end/2)])
# proposal distribution for parameters
kernel = ProductAuxKernel(
    vcat(
        [σ₂ -> truncated(Normal(σ₂,1),0,Inf),
        intercept -> Normal(intercept, 1)],
        fill(m_i->Normal(m_i,1),size(train,2))
    ),
    ones(Int,12)
)
model = iMCMCModel(mh,kernel,model_loglikelihood,first_sample)

# generate chain
rng = MersenneTwister(1)
imcmc_chn = Chains(sample(rng,model,iMCMC(),1000;discard_initial=10))
plot(imcmc_chn)
savefig("test/images/05-linear-regression-imcmc.png")
