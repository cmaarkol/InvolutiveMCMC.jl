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

using InvolutiveMCMC

filename = "05-linear-regression"
imagepath = "test/images/"

# Hide the progress prompt while sampling.
Turing.setprogress!(false);

# Import the "Default" dataset.
data = RDatasets.dataset("datasets", "mtcars");

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

inf = "nuts"
rng1 = MersenneTwister(1)
chain = sample(rng1, lr_model, NUTS(0.65), 3_000);
plot(chain)
savefig(join([imagepath, filename, "-", inf, ".png"]))

# Using iMCMC as the sampler
inf = "imcmc"
ADMH = ADInvolution(
    (x, v)->(v, x),
    s->(s[1:Int(end/2)], s[Int(end/2)+1:end])
)
kernel = PointwiseAuxKernel(
    (n, t) -> n == 1 ? truncated(Normal(t[n],1),0,Inf) : Normal(t[n], 1))
model = iMCMCModel(ADMH, kernel, lr_model)

# generate chain
rng2 = MersenneTwister(2)
imcmcsamples = sample(rng2, model, iMCMC(), 3000)
imcmcchn = Chains(convert(Matrix{Float64}, reduce(hcat, imcmcsamples)'), vcat([:σ₂, :intercept], map(i -> join(["coefficients[", i, "]"]), 1:10)))
plot(imcmcchn)
savefig(join([imagepath, filename, "-", inf, ".png"]))

# Make a prediction given an input vector.
function prediction(chain, x)
    p = get_params(chain[200:end, :, :])
    targets = p.intercept' .+ x * reduce(hcat, p.coefficients)'
    return vec(mean(targets; dims = 2))
end

# Calculate the predictions for the training and testing sets
# and unstandardize them.
p = prediction(chain, train)
train_prediction_bayes = μtarget .+ σtarget .* p
p = prediction(chain, test)
test_prediction_bayes_nuts = μtarget .+ σtarget .* p
p = prediction(imcmcchn, test)
test_prediction_bayes_imcmc = μtarget .+ σtarget .* p

# Show the predictions on the test data set.
DataFrame(
    MPG = testset[!, target],
    NUTSBayes = test_prediction_bayes_nuts,
    iMCMCBayes = test_prediction_bayes_imcmc,
)

println(
    "Training set:",
    "\n\tNUTS Bayes loss: ",
    msd(train_prediction_bayes, trainset[!, target]),
    "\n\tiMCMC Bayes loss: ",
    msd(train_prediction_bayes, trainset[!, target]),
)

println(
    "Test set:",
    "\n\tNUTS Bayes loss: ",
    msd(test_prediction_bayes_nuts, testset[!, target]),
    "\n\tiMCMC Bayes loss: ",
    msd(test_prediction_bayes_imcmc, testset[!, target]),
)
