struct FeatureQuantizer{M<:AbstractMatrix, B, Q}
    weight::M
    bias::B
    output_quantizer::Q

    function FeatureQuantizer(
        weight::W,
        bias::B;
        output_quantizer::Q = Sign(),
    ) where {W<:AbstractMatrix, B<:AbstractMatrix, Q}

        return new{W, B, Q}(weight, bias, output_quantizer)
    end
end

Flux.@functor FeatureQuantizer

function FeatureQuantizer(
    dim::Int,
    k::Int;
    init_weight = glorot_uniform,
    init_bias = (d...) -> randn(Float32, d...),
    kwargs...
)
    return FeatureQuantizer(init_weight(dim, k), init_bias(dim, k); kwargs...)
end

function Base.show(io::IO, q::FeatureQuantizer)
    print(io, "FeatureQuantizer(", size(q.weight, 1), " => ", prod(size(q.weight)), ")")
end

function (q::FeatureQuantizer)(x)
    w, b = q.weight, q.bias
    w1, b1, x1 = size(w, 1), size(b, 1), size(x, 1)
    if !(w1 == b1 == x1)
        msg = "first dimension of weight ($w1), bias ($b1) and x ($x1) must match"
        throw(DimensionMismatch(msg))
    end

    y = similar(x, length(w), size(x, 2))
    for col in axes(x, 2)
        for j in axes(w,2), i in axes(x,1)
            idx = (i-1)*size(w,2) + j
            y[idx,col] = x[i,col] * w[i, j] + b[i, j]
        end
    end
    return q.output_quantizer(y)
end

function ChainRulesCore.rrule(q::FeatureQuantizer, x)

    function FeatureQuantizer_pullback(Δy)
        w, b = q.weight, q.bias
        Δw, Δb, Δx = zero.((w, b, x))

        for col in axes(x, 2)
            for j in axes(w,2), i in axes(x,1)
                if !ismissing(x[i, col])
                    Δw[i, j] += x[i,col] * Δy[i, col]
                    Δb[i, j] += Δy[i, col]
                    Δx[i,col] +=  w[i, j] * Δy[i, col]
                end
            end
        end
        if isa(q.output_quantizer, AbstractQuantizer)
            Δw .*= pullback(q.output_quantizer, w)
            Δb .*= pullback(q.output_quantizer, b)
            Δx .*= pullback(q.output_quantizer, x)
        end
        return (; w = Δw, b = Δb), Δx
    end
    return q(x), FeatureQuantizer_pullback
end
