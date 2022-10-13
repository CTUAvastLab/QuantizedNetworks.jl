using BinNN
using BinNN.Flux
using MLDatasets
using ProgressMeter

using BinNN.Flux.Data: DataLoader
using BinNN.Flux: onehotbatch, onecold
using BinNN.Flux.Losses: logitcrossentropy
using BinNN.Flux.Optimise: update!

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# data loading
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

# performance metrics
function accuracy(data_loader, model)
    acc = 0
    num = 0
    for (x, y) in data_loader
        acc += sum(onecold(model(x)) .== onecold(y))
        num += size(x)[end]
    end
    return acc / num
end

# train loop
function train_model(model, opt, train, test; epochs::Int = 30)
    p = Progress(epochs, 1)
    ps = Flux.params(model)
    history = (
        train_acc = [accuracy(train, model)],
        test_acc = [accuracy(test, model)],
    )

    for _ in 1:epochs
        for (x, y) in train
            gs = gradient(() -> logitcrossentropy(model(x), y), ps)
            update!(opt, ps, gs)
        end

        # compute accuracy
        push!(history.train_acc, accuracy(train, model))
        push!(history.test_acc, accuracy(test, model))

        # print progress
        showvalues = [
            (:acc_train_0, round(100 * history.train_acc[1]; digits = 2)),
            (:acc_train, round(100 * history.train_acc[end]; digits = 2)),
            (:acc_train_0, round(100 * history.test_acc[1]; digits = 2)),
            (:acc_train, round(100 * history.test_acc[end]; digits = 2)),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history
end