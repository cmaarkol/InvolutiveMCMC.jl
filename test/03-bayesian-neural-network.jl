# Import libraries.
using InvolutiveMCMC
using Turing, Flux, Plots, Random, ReverseDiff, DynamicPPL

filename = "03-bayesian-neural-network"
imagepath = "test/images/"

# Hide sampling progress.
Turing.setprogress!(false);

# Use reverse_diff due to the number of parameters in neural networks.
Turing.setadbackend(:reversediff)

# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

# Generate artificial data.
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M]))

x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

# Store all the data for later.
xs = [xt1s; xt0s]
ts = [ones(2*M); zeros(2*M)]

# Plot data points.
function plot_data()
    x1 = map(e -> e[1], xt1s)
    y1 = map(e -> e[2], xt1s)
    x2 = map(e -> e[1], xt0s)
    y2 = map(e -> e[2], xt0s)

    Plots.scatter(x1,y1, color="red", clim = (0,1))
    Plots.scatter!(x2, y2, color="blue", clim = (0,1))
end

plot_data()

# Turn a vector into a set of weights and biases.
function unpack(nn_params::AbstractVector)
    W₁ = reshape(nn_params[1:6], 3, 2);   
    b₁ = nn_params[7:9]
    
    W₂ = reshape(nn_params[10:15], 2, 3); 
    b₂ = nn_params[16:17]
    
    Wₒ = reshape(nn_params[18:19], 1, 2); 
    bₒ = nn_params[20:20]
    return W₁, b₁, W₂, b₂, Wₒ, bₒ
end

# Construct a neural network using Flux and return a predicted value.
function nn_forward(xs, nn_params::AbstractVector)
    W₁, b₁, W₂, b₂, Wₒ, bₒ = unpack(nn_params)
    nn = Chain(Dense(W₁, b₁, tanh),
               Dense(W₂, b₂, tanh),
               Dense(Wₒ, bₒ, σ))
    return nn(xs)
end;

# Create a regularization term and a Gaussain prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)

# Specify the probabilistic model.
@model function bayes_nn(xs, ts)
    # Create the weight and bias vector.
    nn_params ~ MvNormal(zeros(20), sig .* ones(20))
    
    # Calculate predictions for the inputs given the weights
    # and biases in theta.
    preds = nn_forward(xs, nn_params)
    
    # Observe each prediction.
    for i = 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end;

# model
bnn_model = bayes_nn(hcat(xs...), ts)

# Perform inference.
inf = "hmc"
N = 5000
rng1 = MersenneTwister(1)
ch = sample(rng1, bnn_model, HMC(0.05, 4), N);

# Extract all weight and bias parameters.
theta = MCMCChains.group(ch, :nn_params).value;

# Plot the data we have.
plot_data()

# Find the index that provided the highest log posterior in the chain.
_, i = findmax(ch[:lp])

# Extract the max row value from i.
i = i.I[1]

# Plot the posterior distribution with a contour plot.
x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))
Z = [nn_forward([x, y], theta[i, :])[1] for x=x_range, y=y_range]
contour!(x_range, y_range, Z)
savefig(join([imagepath, filename, "-", inf, ".png"]))

# define the iMCMC model
inf = "imcmc"
rng2 = MersenneTwister(2)
ADMH = ADInvolution(
    (x, v)->(v, x),
    s->(s[1:Int(end/2)], s[Int(end/2)+1:end])
)
kernel = AuxKernel(t -> map(x -> Normal(x,1), t))
model = iMCMCModel(ADMH, kernel, bnn_model)

# generate chain
imcmcsamples = sample(rng2, model, iMCMC(), 5000)
imcmcchn = convert(Matrix{Float64}, reduce(hcat, imcmcsamples)')

# Find the index that provided the highest log posterior in the chain.
logp = InvolutiveMCMC.np_gen_logπ(DynamicPPL.SampleFromPrior(), bnn_model)
_, imcmcmaxindex = findmax(map(t -> logp(t)[3], imcmcsamples))

# Plot the posterior distribution with a contour plot.
imcmcZ = [nn_forward([x, y], imcmcchn[imcmcmaxindex, :])[1] for x=x_range, y=y_range]
contour!(x_range, y_range, imcmcZ)
savefig(join([imagepath, filename, "-", inf, ".png"]))
