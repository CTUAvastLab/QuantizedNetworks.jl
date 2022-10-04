using Pkg
Pkg.activate(@__DIR__)

using Revise
using BinNN
using Plots

include(joinpath(@__DIR__, "utilities.jl"))

# data
dataset = "Flower"
train, test = createloader(900, 900; batchsize = 100)

# model
input_size = size(first(train)[1], 1)
nclasses = size(first(train)[2], 1)

model = Chain(
    Dense(input_size, 20, relu),
    Dense(20, 20, relu),
    Dense(20, nclasses),
)

weight_lims = (-1, 1)
bias_lims = nothing
weight_quantizer = Sign()

model_bin = Chain(
    QuantDense(model[1]; σ = identity, weight_quantizer, weight_lims, bias_lims),
    BatchNorm(20, relu),
    QuantDense(model[2]; σ = identity, weight_quantizer, weight_lims, bias_lims),
    BatchNorm(20, relu),
    QuantDense(model[3]; weight_quantizer, weight_lims, bias_lims),
)

# training
epochs = 50

history = train_model(model, AdaBelief(), train, test; epochs)
history_bin = train_model(model_bin, AdaBelief(), train, test; epochs)

# plots
plt1 = plot(history.train_acc; label = "normal model", title = "Train $(dataset)")
plot!(plt1, history_bin.train_acc; label = "binary model")

plt2 = plot(history.test_acc; label = "normal model", title = "Test $(dataset)")
plot!(plt2, history_bin.test_acc; label = "binary model")

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
