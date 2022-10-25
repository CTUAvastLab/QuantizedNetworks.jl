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

function forward_pass(w, b, x; output_missing::Bool)
    if output_missing
        y = similar(w, length(w) + size(x, 1), size(x, 2))
        y[end-size(x,1)+1:end, :] .= ismissing.(x)
    else
        y = similar(w, length(w), size(x, 2))
    end

    for col in 1:size(x, 2)
        for j in 1:size(w,2), i in 1:size(x,1)
            idx = (i-1)*size(w,2) + j
            y[idx,col] = !ismissing(x[i,col]) * ifelse((x[i,col] * w[i, j] + b[i, j]) > 0, 1, -1)
        end
    end
    return y
end

function ChainRulesCore.rrule(::typeof(forward_pass), w, b, x; output_missing::Bool)
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


# pokus
struct FQuantizer2{M<:AbstractMatrix, B} <: AbstractFQuantizer
    weight::M
    bias::B
    output_missing::Bool

    function FQuantizer2(
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

Flux.@functor FQuantizer2

function FQuantizer2(
    dims::NTuple{2, <:Integer};
    init_weight = glorot_uniform,
    init_bias = (d...) -> randn(Float32, d...),
    kwargs...
)
    return FQuantizer2(init_weight(dims...), init_bias(dims...); kwargs...)
end

ispositive(x::T) where {T <:Real} = T(x > 0)
ispositive(x::Missing) = false

function (q::FQuantizer2)(x)
    w, b = q.weight, q.bias
    xr = reshape(x, size(x, 1), 1, :)
    return Flux.flatten(ispositive.(xr .* w .+ b))
end
