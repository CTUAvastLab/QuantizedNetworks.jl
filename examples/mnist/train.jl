using Pkg
Pkg.activate(@__DIR__)

using Revise
using QuantizedNetworks
using Plots
using Random

include(joinpath(@__DIR__, "utilities.jl"))

# data
Random.seed!(1234)
dataset = MNIST
train, test = createloader(dataset; batchsize = 256)

# model
input_size = size(first(train)[1], 1)
nclasses = size(first(train)[2], 1)
nhidden = 256

model = Chain(
    Dense(input_size => nhidden, relu),
    Dense(nhidden => nclasses),
)

σ = hardtanh
kwargs = (;
    init = (dims...) -> ClippedArray(dims...; lo = -1, hi = 1),
    output_quantizer = identity,
    batchnorm = true,
)

model_bin = Chain(
    QuantDense(input_size => nhidden, σ; kwargs...),
    QuantDense(nhidden =>nclasses; kwargs...),
)

# training
epochs = 15
loss = logitcrossentropy

history = train_model(model, AdaBelief(), train, test; epochs, loss)
history_bin = train_model(model_bin, AdaBelief(), train, test; epochs, loss)

# plots
plt1 = plot(history.train_acc; label = "normal model", title = "Train $(dataset)");
plot!(plt1, history_bin.train_acc; label = "binary model");

plt2 = plot(history.test_acc; label = "normal model", title = "Test $(dataset)");
plot!(plt2, history_bin.test_acc; label = "binary model");

plt = plot(
    plt1,
    plt2;
    layout = (2, 1),
    size = (600, 800),
    ylims = (0, 1),
    xlabel = "epoch",
    ylabel = "accuracy (%)",
    legend = :bottomright
)

savefig(plt, "$(dataset).png")
