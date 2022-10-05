"""
    QuantDense(in => out, σ=identity; bias=true, init=glorot_uniform)
    QuantDense(W::AbstractMatrix, [bias, σ]; kwargs...)
"""
struct QuantDense{F, M<:AbstractMatrix, B, Q1, Q2, N}
    weight::M
    bias::B
    σ::F

    weight_quantizer::Q2
    output_quantizer::Q1
    batchnorm::N
end

function QuantDense(
    weight::AbstractArray,
    bias,
    σ = identity;
    weight_quantizer = Ternary(),
    output_quantizer = Sign(),
    weight_lims = nothing,
    bias_lims = nothing,
    batchnorm::Bool = true,
 )
    bias = _create_bias(weight, bias, size(weight,1))
    if isa(bias, AbstractArray) && !isnothing(bias_lims)
        bias = ClippedArray(bias, bias_lims...)
    end

    return QuantDense(
        isnothing(weight_lims) ? weight : ClippedArray(weight, weight_lims...),
        bias,
        batchnorm ? identity : σ,
        weight_quantizer,
        output_quantizer,
        batchnorm ? BatchNorm(size(weight, 1), σ) : identity,
    )
end

function QuantDense(
    (in, out)::Pair{<:Integer, <:Integer},
    σ = identity;
    init = glorot_uniform,
    bias = false,
    kwargs...
)
    return QuantDense(init(out, in), bias, σ; kwargs...)
end

function QuantDense(
    l::Dense;
    weight = copy(l.weight),
    bias = copy(l.bias),
    σ = l.σ,
    kwargs...
)

    return QuantDense(weight, bias, σ; kwargs...)
end

@functor QuantDense

function (l::QuantDense)(x::AbstractVecOrMat)
    σ = NNlib.fast_act(l.σ, x)  # replaces tanh => tanh_fast, etc
    wq = l.weight_quantizer(l.weight)

    return l.output_quantizer(l.batchnorm(σ.(wq * x .+ l.bias)))
end

function (l::QuantDense)(x::AbstractArray)
    return reshape(l(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
end

# TODO improve printing
function Base.show(io::IO, l::QuantDense)
    print(io, "QuantDense(", size(l.weight, 2), " => ", size(l.weight, 1))

    l.σ == identity || print(io, ", ", l.σ)
    l.batchnorm == identity || print(io, ", ", l.batchnorm.λ)

    kwargs = String[]
    if isa(l.weight, ClippedArray)
        push!(kwargs, "weight_lims=$((l.weight.lo, l.weight.hi))")
    end
    if l.bias == false
        push!(kwargs, "bias=false")
    else
        if isa(l.bias, ClippedArray)
            push!(kwargs, "bias_lims=$((l.bias.lo, l.bias.hi))")
        end
    end
    push!(kwargs, "$(l.weight_quantizer)")
    push!(kwargs, "$(l.output_quantizer)")
    l.batchnorm == identity || push!(kwargs, "batchnorm=false")
    print(io, "; ", join(kwargs, ", "), ")")
end
