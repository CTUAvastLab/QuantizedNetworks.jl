######
#   A tutorial how to use the explainer
######

# Blossoming NuLog - A practical demonstration on Flower problem
# 
# NuLog is a package designs to investigate the duality of neural networks with 
# ternary weights {-1, 0, +1 } and binary outputs (either {0, +1} or {-1,+1}), 
# and logic. We can first convert the neural network to a set of logical rules,
# or ask for explanation in terms of logical rules.
# The system is in its napkins at the moment, therefore expect some rough edges.

# In below tutorial, we use a QuantizedNetwork.jl package to train the neural 
# networks with the above restrictions to weights and nonlinearities. Then, the
# networs are converted to logic to see, how many rules. Finally, we use the explainer
# to convert the neural network to a small compact set of rules.


# We start by importing few images.
using Pkg
Pkg.activate(@__DIR__)

using QuantizedNetworks
using NuLog
using GLMakie
using Random

include(joinpath(@__DIR__, "utilities.jl"))

# Instantiate the loader for the flower dataset. The flower problem is a toy dataset 
# with symmetrically spaced non-linearly transformed normal distributions. It was 
# introduced in *Sum-Product-Transform Networks: Exploiting Symmetries using 
# Invertible Transformations, 2020* for density estimation. Here, we modify it to 
# classification problem, where each leaf is one class.

Random.seed!(1234)
dataset = "Flower"
train, test = createloader(900, 900; batchsize = 100)
input_size = size(first(train)[1], 1)
nclasses = size(first(train)[2], 1)


# We first train a classical neural network without restriction on weights for comparison
model = Chain(
    Dense(input_size => 20, relu),
    Dense(20 => 20, relu),
    Dense(20 => nclasses),
)

# training
epochs = 100
loss = logitcrossentropy
history = train_model(model, AdaBelief(), train, test; epochs, loss)

# We now train a neural network with quantized weights. The QuantizedNetworks.jl
# is very flexible. We set the output quantizer to Sign with a straight-through
# estimator of the gradient, BatchNormalization, and hardtanh nonlinearity before 
# the batch normalization. These are some tricks to ease the training.
# Notice the first layer is feature quantizer, `FeatureQuantizer`, which
# quantizes real-valued features to binary values. Subsequent layers, 
# are QuantDense. See QuantizedNetworks for details.
# input and produces binary output.
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

history_bin = train_model(model_bin, AdaBelief(), train, test; epochs, loss)

# A difference in decision boundary can be seen below
let 
    fig = Figure(resolution = (1000, 700))
    ga = fig[1, 1] = GridLayout()
    ax_trn = Axis(ga[1, 1], title = "accuracy on the training set")
    lines!(ax_trn, history.train_acc, label = "normal model")
    lines!(ax_trn, history_bin.train_acc, label = "binary model")
    ax_tst = Axis(ga[1, 2], title = "accuracy on the testing set")
    lines!(ax_tst, history.test_acc, label = "normal model")
    lines!(ax_tst, history_bin.test_acc, label = "binary model")
    leg = Legend(ga[1, 3], ax_trn)
    fig

    x = test.data[1]
    ax = Axis(ga[2,1])
    xr = minimum(x[1,:])-0.1:0.1:maximum(x[1,:])+0.1
    yr = minimum(x[2,:])-0.1:0.1:maximum(x[2,:])+0.1
    z = [argmax(model([x, y])) for x in xr, y in yr]
    heatmap!(ax, xr, yr, z)
    scatter!(ax, x[1,:], x[2,:])

    ax = Axis(ga[2,2])
    z = [argmax(vec(model_bin([x, y]))) for x in xr, y in yr]
    heatmap!(ax, xr, yr, z)
    scatter!(ax, x[1,:], x[2,:])
    fig
end



# Now, we can take a trained quantized neural network and convert it to
# a set of rules preserving its structure at the moment. We werify that
# the model composed by a set of rules has the same output as the model
# original model.

rule_model = NuLog.nn2logic(model_bin)
x = first(train)[1]
rule_model(x) ≈ (model_bin(x) .> 0)


# Units of the rule_model are composed by a rules. The first layer contains
# rules converting real values to the true / false, as it was created from
# `FeatureQuantizer` layer.

rule_model[1].rule_sets


# Notice that some rules on the training set "acts" as constants, as they
# are always equal to true / false. Subsequent layers are created by a
# **conjuctions* of m-of-n rules

rule_model[2].rule_sets


logic_model = NuLog.nn2logic(model, simulate = true)



###
logic_model = NuLog.nn2logic(model_bin, simulate = true)
method = BackwardExplanation()
e = explain(method, logic_model, x[:,1])

# Explainer can be used to extract rules for all samples in the 
# training set, which effectively makes it a classifier
rules = Dict{Int,Any}()
for i in 1:size(x, 2)
    v = x[:,i]
    any(r(v) for r in values(rules)) && continue
    r = explain(method, logic_model, v)
    cᵢ = argmax(logic_model(v)[:])
    rules[cᵢ] = ruleset(r, get(rules, cᵢ, Fact(false)))
end

# Let's now show, how extracted rules cover the space. We see that 
# there are "holes", which are not covered by any rule

let 
    fig = Figure(resolution = (800, 600))
    ga = fig[1, 1] = GridLayout()
    x = test.data[1]
    ax = Axis(ga[1,1])
    xr = minimum(x[1,:])-0.1:0.1:maximum(x[1,:])+0.1
    yr = minimum(x[2,:])-0.1:0.1:maximum(x[2,:])+0.1
    z = map(Iterators.product(xr, yr)) do (x,y)
        for (i, r) in rules
            r([x,y]) && return(i)
        end 
        return(0)
    end

    heatmap!(ax, xr, yr, z)
    scatter!(ax, x[1,:], x[2,:])
    ax = Axis(ga[1,2])
    z = [argmax(vec(model_bin([x, y]))) for x in xr, y in yr]
    heatmap!(ax, xr, yr, z)
    scatter!(ax, x[1,:], x[2,:])
    fig
end

heatmap(xr, yr, z)