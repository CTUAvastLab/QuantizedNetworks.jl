abstract type AbstractFQuantizer end

struct FQuantizer{M<:AbstractMatrix, B} <: AbstractFQuantizer
    weight::M
    bias::B
    output_missing::Bool

    function FQuantizer(
        weight::AbstractMatrix,
        bias::AbstractMatrix;
        weight_lims = nothing,
        bias_lims = nothing,
        output_missing::Bool = false,
    )

        weight = isnothing(weight_lims) ? weight : ClippedArray(weight, weight_lims...)
        bias = isnothing(bias_lims) ? bias : ClippedArray(bias, bias_lims...)
        return new{typeof(weight), typeof(bias)}(weight, bias, output_missing)
    end
end

Flux.@functor FQuantizer

function FQuantizer(
    dims::NTuple{2, <:Integer};
    init_weight = glorot_uniform,
    init_bias = (d...) -> randn(Float32, d...),
    kwargs...
)
    return FQuantizer(init_weight(dims...), init_bias(dims...); kwargs...)
end

function (q::FQuantizer)(x)
    return forward_pass(q.weight, q.bias, x; output_missing = q.output_missing)
end

function Base.show(io::IO, q::AbstractFQuantizer)
    n_out = prod(size(q.weight)) + q.output_missing * size(q.weight, 1)
    print(io, "FQuantizer(", size(q.weight, 1), " => ", n_out, ")")
end

ispositive(x::T) where {T <:Real} = ifelse(x > 0, one(T), -one(T))
ispositive(x::Missing) = -1

function forward_pass(w, b, x; output_missing::Bool = false)
    w1, b1, x1 = size(w, 1), size(b, 1), size(x, 1)
    if w1 != x1
        msg = "first dimension of w ($w1) must match first dimension of x ($x1)"
        throw(DimensionMismatch(msg))
    end
    if b1 != x1
        msg = "first dimension of b ($b1) must match first dimension of x ($x1)"
        throw(DimensionMismatch(msg))
    end

    if output_missing
        y = similar(w, length(w) + size(x, 1), size(x, 2))
        y[end-size(x,1)+1:end, :] .= ifelse.(ismissing.(x), -1, 1)
    else
        y = similar(w, length(w), size(x, 2))
    end

    for col in 1:size(x, 2)
        for j in 1:size(w,2), i in 1:size(x,1)
            idx = (i-1)*size(w,2) + j
            y[idx,col] = ispositive(x[i,col] * w[i, j] + b[i, j])
        end
    end
    return y
end

function ChainRulesCore.rrule(::typeof(forward_pass), w, b, x; output_missing::Bool = false)
    y = forward_pass(w, b, x; output_missing)

    function fquantizer_forward_pass_pullback(Δy)
        Δw, Δb, Δx = zero.((w, b, x))

        for col in 1:size(x, 2)
            for j in 1:size(w,2), i in 1:size(x,1)
                idx = (i-1)*size(w,2) + j
                if ismissing(x[i, col])
                    tmp = Δy[i, col] * y[idx, col]*(1 - y[idx, col])
                    Δw[i, j] += x[i,col] * tmp
                    Δb[i, j] += tmp
                    Δx[i,col] +=  tmp * w[i, j]
                end
            end
        end
        return NoTangent(), Δw, Δb, Δx
    end
    return y, fquantizer_forward_pass_pullback
end
