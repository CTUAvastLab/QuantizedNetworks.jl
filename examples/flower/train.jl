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
output_quantizer = identity

model_bin = Chain(
    QuantDense(model[1]; output_quantizer, weight_lims),
    QuantDense(model[2]; output_quantizer, weight_lims),
    QuantDense(model[3]; output_quantizer, weight_lims),
)

# training
epochs = 100

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

# plot
plt_flower = plot(
    plot_decision(model, test.data; title = "Normal model (test)"),
    plot_decision(model_bin, test.data; title = "Binary model (test)");
    layout = (2, 1),
    size = (600, 800),
    legend = false,
)

savefig(plt_flower, "$(dataset)_decision.png")
