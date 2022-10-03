using Revise

using BinNN
using BinNN.Flux
using MLDatasets

using BinNN.Flux.Data: DataLoader
using BinNN.Flux: onehotbatch, onecold
using BinNN.Flux.Losses: logitcrossentropy
using BinNN.Flux.Optimise: update!

# initialization
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
batchsize = 256
imgsize = (28, 28, 1)
nclasses = 10
epochs = 30

function loss_and_accuracy(data_loader, model)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(ŷ) .== onecold(y)) ## Decode the output of the model
        num += size(x)[end]
    end
    return ls / num, acc / num
end

# train data
xtrain, ytrain = MLDatasets.MNIST(:train)[:]
train = (Flux.flatten(xtrain), onehotbatch(ytrain, 0:9))
train_loader = DataLoader(train; batchsize, shuffle=true)

# test data
xtest, ytest = MLDatasets.MNIST(:test)[:]
test = (Flux.flatten(xtest), onehotbatch(ytest, 0:9))
test_loader = DataLoader(test; batchsize)


weight_lims = (-1, 1)
input_quantizer = identity
weight_quantizer = SwishSign()

model = Chain(
    QuantDense(prod(imgsize) => 32; input_quantizer, weight_quantizer, weight_lims),
    QuantDense(32 => nclasses; input_quantizer, weight_quantizer, weight_lims),
)

ps = Flux.params(model)
opt = ADAM(3f-4)

## Training
train_loss, train_acc = loss_and_accuracy(train_loader, model)
test_loss, test_acc = loss_and_accuracy(test_loader, model)
println("Initialization:")
println("  train_loss = $train_loss, train_accuracy = $train_acc")
println("  test_loss = $test_loss, test_accuracy = $test_acc")

for epoch in 1:epochs
    for (x, y) in train_loader
        gs = gradient(() -> logitcrossentropy(model(x), y), ps)
        update!(opt, ps, gs)
    end

    ## Report on train and test
    if epoch % 5 == 0 || epoch == 1
        train_loss, train_acc = loss_and_accuracy(train_loader, model)
        test_loss, test_acc = loss_and_accuracy(test_loader, model)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
    end
end
