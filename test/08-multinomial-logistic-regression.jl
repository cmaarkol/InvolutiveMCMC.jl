using InvolutiveMCMC
# Load Turing.
using Turing

# Load RDatasets.
using RDatasets

# Load StatsPlots for visualizations and diagnostics.
using StatsPlots

# Functionality for splitting and normalizing the data.
using MLDataUtils: shuffleobs, splitobs, rescale!

# We need a softmax function which is provided by NNlib.
using NNlib: softmax

# Set a seed for reproducibility.
using Random
Random.seed!(0)

# Hide the progress prompt while sampling.
Turing.setprogress!(false);

filename = "08-multinomial-logistic-regression"
imagepath = "test/images/"

# Import the "iris" dataset.
data = RDatasets.dataset("datasets", "iris");

# Show twenty random rows.
data[rand(1:size(data, 1), 20), :]

# Recode the `Species` column.
species = ["setosa", "versicolor", "virginica"]
data[!, :Species_index] = indexin(data[!, :Species], species)

# Show twenty random rows of the new species columns
data[rand(1:size(data, 1), 20), [:Species, :Species_index]]

# Split our dataset 50%/50% into training/test sets.
trainset, testset = splitobs(shuffleobs(data), 0.5)

# Define features and target.
features = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]
target = :Species_index

# Turing requires data in matrix and vector form.
train_features = Matrix(trainset[!, features])
test_features = Matrix(testset[!, features])
train_target = trainset[!, target]
test_target = testset[!, target]

# Standardize the features.
μ, σ = rescale!(train_features; obsdim = 1)
rescale!(test_features, μ, σ; obsdim = 1);

# Bayesian multinomial logistic regression
@model function logistic_regression(x, y, σ)
    n = size(x, 1)
    length(y) == n || throw(DimensionMismatch("number of observations in `x` and `y` is not equal"))

    # Priors of intercepts and coefficients.
    intercept_versicolor ~ Normal(0, σ)
    intercept_virginica ~ Normal(0, σ)
    coefficients_versicolor ~ MvNormal(4, σ)
    coefficients_virginica ~ MvNormal(4, σ)

    # Compute the likelihood of the observations.
    values_versicolor = intercept_versicolor .+ x * coefficients_versicolor
    values_virginica = intercept_virginica .+ x * coefficients_virginica
    for i in 1:n
        # the 0 corresponds to the base category `setosa`
        v = softmax([0, values_versicolor[i], values_virginica[i]])
        y[i] ~ Categorical(v)
    end
end;

model_mlr = logistic_regression(train_features, train_target, 1)

inf = "hmc"
rng1 = MersenneTwister(1)
chain = sample(rng1, model_mlr, HMC(0.05, 10), MCMCThreads(), 1500, 3)
plot(chain)
savefig(join([imagepath, filename, "-", inf, ".png"]))

function prediction(x::Matrix, chain)
    # Pull the means from each parameter's sampled values in the chain.
    intercept_versicolor = mean(chain, :intercept_versicolor)
    intercept_virginica = mean(chain, :intercept_virginica)
    coefficients_versicolor = [
        mean(chain, k) for k in
        MCMCChains.namesingroup(chain, :coefficients_versicolor)
    ]
    coefficients_virginica = [
        mean(chain, k) for k in
        MCMCChains.namesingroup(chain, :coefficients_virginica)
    ]

    # Compute the index of the species with the highest probability for each observation.
    values_versicolor = intercept_versicolor .+ x * coefficients_versicolor
    values_virginica = intercept_virginica .+ x * coefficients_virginica
    species_indices = [argmax((0, x, y)) for (x, y) in zip(values_versicolor, values_virginica)]
    
    return species_indices
end;

# Make the predictions.
predictions = prediction(test_features, chain)

# Calculate accuracy for our test set.
mean(predictions .== testset[!, :Species_index])

for s in 1:3
    rows = testset[!, :Species_index] .== s
    println("Number of `", species[s], "`: ", count(rows))
    println("Percentage of `", species[s], "` predicted correctly: ",
        mean(predictions[rows] .== testset[rows, :Species_index]))
end

# Using iMCMC as the sampler

inf = "imcmc"
ADMH = ADInvolution(
    (x, v)->(v, x),
    s->(s[1:Int(end/2)], s[Int(end/2)+1:end])
)
kernel = PointwiseAuxKernel((n, t) -> Normal(t[n], 1))
model = iMCMCModel(ADMH, kernel, model_mlr)

# generate chain
rng2 = MersenneTwister(2)
imcmcsamples = sample(rng2, model, iMCMC(), 3000)
imcmcchn = Chains(convert(Matrix{Float64}, reduce(hcat, imcmcsamples)'),
    vcat([:intercept_versicolor, :intercept_virginica],
    map(i -> join(["coefficients_versicolor[", i, "]"]), 1:4),
    map(i -> join(["coefficients_virginica[", i, "]"]), 1:4)))
plot(imcmcchn)
savefig(join([imagepath, filename, "-", inf, ".png"]))

# Make the predictions.
imcmc_predictions = prediction(test_features, imcmcchn)

# Calculate accuracy for our test set.
mean(imcmc_predictions .== testset[!, :Species_index])

println("imcmc predictions")
for s in 1:3
    rows = testset[!, :Species_index] .== s
    println("Number of `", species[s], "`: ", count(rows))
    println("Percentage of `", species[s], "` predicted correctly: ",
        mean(imcmc_predictions[rows] .== testset[rows, :Species_index]))
end
