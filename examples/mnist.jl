using Revise
using BinNN
using Plots

include(joinpath(@__DIR__, "utilities.jl"))

# data
dataset = MNIST
train, test = createloader(dataset; batchsize = 256)

# model
nclasses = 10
imgsize = (28, 28, 1)

model = Chain(
    Dense(prod(imgsize), 32, relu),
    Dense(32, nclasses),
)

weight_lims = (-1, 1)
input_quantizer = identity
weight_quantizer = Sign()

model_bin = Chain(
    QuantDense(prod(imgsize) => 32, relu; input_quantizer, weight_quantizer, weight_lims),
    BatchNorm(32),
    QuantDense(32 => nclasses; input_quantizer, weight_quantizer, weight_lims),
)

# training
η = 3f-4
epochs = 30

history = train_model(model, ADAM(η), train, test; epochs)
history_bin = train_model(model_bin, ADAM(η), train, test; epochs)

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
