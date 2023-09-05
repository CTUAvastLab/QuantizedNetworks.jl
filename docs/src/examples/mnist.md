# MNIST Example

The MNIST dataset is a widely recognized collection of handwritten digits used for training and benchmarking machine learning models. 
Following example demonstrates a use case of QuantizedNeteorks package and discrete neural networks on a problem of classifying handwritten digits.

## Create a project

Create a Julia project and add following dependecies
- MLDatasets 
- QuantizedNetworks 
- Plots 
- NuLog 
- ProgressMeter 
- StatsBase 

## Utilities

As it is needed to process and prepare data to be used in traing a network, it is best to create a file for all the helper functions called utilities.jl

### Packages
After creating the file include the following packages and their respective functions to the file.

```julia
using QuantizedNetworks
using QuantizedNetworks.Flux
using MLDatasets
using ProgressMeter

using QuantizedNetworks.Flux.Data: DataLoader
using QuantizedNetworks.Flux: onehotbatch, onecold
using QuantizedNetworks.Flux.Losses: logitcrossentropy, mse
using QuantizedNetworks.Flux.Optimise: update!
```
### Streamlining dependecies
In this case it is best to set the environment variable `DATADEPS_ALWAYS_ACCEPT` to `true` in order to streamline and automate the management of data dependencies during the execution of a Julia script or program. By doing this, it bypasses any prompts or user interactions that might otherwise occur when data dependencies need to be fetched or updated.

```julia
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
```

!!! note 
    This practice is useful in automated scripts or when you want to ensure that your Julia program runs without interruption. However, it is important to exercise caution and only use this approach when you are confident about the data dependencies your code relies on, as it bypasses potential prompts that could help prevent unintended changes to your data.

### Data loading

Create a function to work with `MNIST` dataset and a desired batchsize, in order to preprocess the data and prepare it for training.

```julia
function createloader(dataset = MLDatasets.MNIST; batchsize::Int = 256)
    xtrain, ytrain = dataset(:train)[:]
    train_loader = DataLoader(
        (Flux.flatten(xtrain), onehotbatch(ytrain, 0:9));
        batchsize,
        shuffle=true,
    )

    xtest, ytest = dataset(:test)[:]
    test_loader = DataLoader(
        (Flux.flatten(xtest), onehotbatch(ytest, 0:9));
        batchsize,
    )
    return train_loader, test_loader
end
```

```
createloader (generic function with 2 methods)
```

### Accuracy measure

Add a function to calculate the accuracy of model predictions versus the data.

```julia
function accuracy(data_loader, model)
    acc = 0
    num = 0
    for (x, y) in data_loader
        acc += sum(onecold(model(x)) .== onecold(y))
        num += size(x)[end]
    end
    return acc / num
end
```
```
accuracy (generic function with 1 method)
```

### Training

Add a function to simplify the training of a model.

```julia
function train_model(model, opt, train, test; loss = logitcrossentropy, epochs::Int = 30)
    p = Progress(epochs, 1)
    ps = Flux.params(model)
    history = (
        train_acc = [accuracy(train, model)],
        test_acc = [accuracy(test, model)],
    )

    for _ in 1:epochs
        for (x, y) in train
            gs = gradient(() -> loss(model(x), y), ps)
            update!(opt, ps, gs)
        end

        # compute accuracy
        push!(history.train_acc, accuracy(train, model))
        push!(history.test_acc, accuracy(test, model))

        # print progress
        showvalues = [
            (:acc_train_0, round(100 * history.train_acc[1]; digits = 2)),
            (:acc_train, round(100 * history.train_acc[end]; digits = 2)),
            (:acc_test_0, round(100 * history.test_acc[1]; digits = 2)),
            (:acc_test, round(100 * history.test_acc[end]; digits = 2)),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history
end
```
```
train_model (generic function with 1 method)
```

## Create a model

### Dependecies

Now create a julia script that will perform the training of neural networks.

Start by adding the required packages, activating the current project and including the `utilities.jl` file.

```julia
using Pkg
Pkg.activate(@__DIR__)

using Revise
using QuantizedNetworks
using Plots
using Random

include(joinpath(@__DIR__, "utilities.jl"))
```

!!! note
    It is necessary to activate the project yo set the active project environment to be the one located in the directory where the script or notebook is executed, not the global environment.

### Load data

Set the seed for generating random numers (in order to be able to replicate the results) and call the createloader function to load the training and testing datasets.
It is also usefull to prepare the dimensions of the architecture of a neural network with 3 layers (input, hidden and output).

```julia
Random.seed!(1234)
dataset = MNIST
train, test = createloader(dataset; batchsize = 256)

input_size = size(first(train)[1], 1)
nclasses = size(first(train)[2], 1)
nhidden = 256
```

### Standard model

Create a standard `Flux.jl` model to have as a reference to compare the discrete version performance to.

```julia
model = Chain(
    Dense(input_size => nhidden, relu),
    Dense(nhidden => nclasses),
)
```

```
Chain(
  Dense(784 => 256, relu),              # 200_960 parameters
  Dense(256 => 10),                     # 2_570 parameters
)                   # Total: 4 arrays, 203_530 parameters, 795.289 KiB.
```

### Binary model

Use the hard hyperbolic tangent function as the activation function.
As there are a lot of keyword arguments it is best to create a separate `NamedTuple` to make it easier to read and understand.

#### Hyperparameters

```julia
σ = hardtanh
kwargs = (;
    init = (dims...) -> ClippedArray(dims...; lo = -1, hi = 1),
    output_quantizer = Sign(),
    batchnorm = true,
)
```

```
(init = var"#15#16"(), output_quantizer = Sign(STE(2)), batchnorm = true)
```


Explanation:
- init: It takes a function to initialise the weight matrix, in this case it is an anonymus function that takes `n` dimensions and creates an n-dimensional ClippedArray which is clamped in the range `[-1, 1]`
- For other arguments look up [`QuantDense`](@ref)

#### Model

Now create a binary model.
It will have two `QuantDense` layers, but it will be preceded by a layer that is defined as a anonymus function that will only transform the input values to binary values ${-1, 1}$.
In other words it will qunatize the input features of data.

```julia
model_bin = Chain(
    x -> Float32.(ifelse.(x .> 0, 1, -1)),
    QuantDense(input_size => nhidden, σ; kwargs...),
    QuantDense(nhidden => nclasses; kwargs...),
)
```

```
Chain(
  var"#17#18"(),
  QuantDense(
    Float32[-0.028019346 -0.047897033 … -0.06890707 0.05361874; -0.0464801 -0.02306688 … -0.02808203 -0.0005326818; … ; -0.06333841 0.0009912428 … -0.0015381856 -0.06842575; -0.036003914 -0.03693979 … -0.03655994 0.057226446],  # 200_704 parameters
    false,
    identity,
    Ternary(0.05, STE(2)),
    identity,
    Sign(STE(2)),
    BatchNorm(256, hardtanh),           # 512 parameters, plus 512
  ),
  QuantDense(
    Float32[0.13371286 -0.13981822 … -0.13812053 -0.11148539; 0.12859496 -0.030631518 … -0.045713615 0.1391876; … ; -0.06140963 0.1481502 … -0.001152252 -0.046711177; -0.128069 -0.11804912 … 0.08063337 -0.14401688],  # 2_560 parameters
    false,
    identity,
    Ternary(0.05, STE(2)),
    identity,
    Sign(STE(2)),
    BatchNorm(10),                      # 20 parameters, plus 20
  ),
)         # Total: 6 trainable arrays, 203_796 parameters,
          # plus 4 non-trainable, 532 parameters, summarysize 798.969 KiB.
```

## Train a model

Define a desired number of epochs, the loss function and run the training of two models, by calling the `train_model`. function from `utilities.jl`
It will take a few minutes to complete.

```julia
epochs = 15
loss = logitcrossentropy

history = train_model(model, AdaBelief(), train, test; epochs, loss)
history_bin = train_model(model_bin, AdaBelief(), train, test; epochs, loss)
```

```
(train_acc = [0.10143333333333333, 0.6454, 0.8493, 0.91595, 0.92225, 0.9266166666666666, 0.9364166666666667, 0.9343, 0.9495, 0.94765, 0.9584666666666667, 0.9565333333333333, 0.959, 0.9600666666666666, 0.95585, 0.9612833333333334], test_acc = [0.1026, 0.6464, 0.8468, 0.9075, 0.9137, 0.9159, 0.9288, 0.9228, 0.9336, 0.9336, 0.944, 0.9423, 0.9405, 0.9376, 0.9358, 0.9456])
```

### Plot the results

Plot the results to compare the two models and make further adjustments to the hyperparametars.
```julia
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
```
