using BinNN
using BinNN.Flux
using ProgressMeter

using BinNN.Flux.Data: DataLoader
using BinNN.Flux: onehotbatch, onecold
using BinNN.Flux.Losses: logitcrossentropy
using BinNN.Flux.Optimise: update!

# data creation
# The flower problem is a toy dataset with symmetrically spaced
# non-linearly transformed normal distributions. It was introduced in
# *Sum-Product-Transform Networks: Exploiting Symmetries using
# Invertible Transformations, 2020* for density estimation. Here, we
# modify it to classification problem, where each leaf is one class.
function generate_flower(n, npetals = 8)
	n = div(n, npetals)
	x = mapreduce(hcat, (1:npetals) .* (2π/npetals)) do θ
		x0 = tanh.(randn(1, n) .- 1) .+ 4.0 .+ 0.05.* randn(1, n)
		y0 = randn(1, n) .* 0.3

		return vcat(
            x0 * cos(θ) .- y0 * sin(θ),
            x0 * sin(θ) .+ y0 * cos(θ),
        )
	end
	y = mapreduce(i -> fill(i, n), vcat, 1:npetals)
	return x, y
end

function createloader(n::Int, n_test::Int = n; batchsize::Int = 100, npetals = 8)
    xtrain, ytrain = generate_flower(n, npetals)
    train_loader = DataLoader(
        (Flux.flatten(xtrain), onehotbatch(ytrain, 1:npetals));
        batchsize,
        shuffle=true,
    )

    xtest, ytest = generate_flower(n_test, npetals)
    test_loader = DataLoader(
        (Flux.flatten(xtest), onehotbatch(ytest, 1:npetals));
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
