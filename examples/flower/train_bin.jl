using Pkg
Pkg.activate(@__DIR__)

using Revise
using QuantizedNetworks
using Plots
using Random

include(joinpath(@__DIR__, "utilities.jl"))

# data
Random.seed!(1234)
dataset = "Flower"
train, test = createloader(900, 900; batchsize = 100)

# model
input_size = size(first(train)[1], 1)
nclasses = size(first(train)[2], 1)

model = Chain(
    Dense(input_size => 20, relu),
    Dense(20 => 20, relu),
    Dense(20 => nclasses),
)

k = 25
σ = hardtanh
output_quantizer = Sign(STE(1))
kwargs = (;
    init = (dims...) -> ClippedArray(dims...; lo = -1, hi = 1),
    output_quantizer = output_quantizer,
    batchnorm = true,
)

model_bin = Chain(
    FeatureQuantizer(input_size, k; output_quantizer),
    QuantDense(k*input_size => 20, σ; kwargs...),
    QuantDense(20 => 20, σ; kwargs...),
    QuantDense(20 =>nclasses; kwargs...),
)

# training
epochs = 100
loss = logitcrossentropy

history = train_model(model, AdaBelief(0.01), train, test; epochs, loss)
history_bin = train_model(model_bin, AdaBelief(0.01), train, test; epochs, loss)

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

# plot
plt_flower = plot(
    plot_decision(model, test.data; title = "Normal model (test)"),
    plot_decision(model_bin, test.data; title = "Binary model (test)");
    layout = (2, 1),
    size = (600, 800),
    legend = false,
)

savefig(plt_flower, "$(dataset)_decision.png")
