"""
    QuantDense(in => out, σ=identity; bias=true, init=glorot_uniform)
    QuantDense(W::AbstractMatrix, [bias, σ]; kwargs...)

Create quantized fully connected layer, whose forward pass is given by:

y = σ.(weight_quantizer(W) * input_quantizer(x) .+ bias)

The input `x` should be a vector of length `in`, or batch of vectors represented as an
`in × N` matrix, or any array with `size(x,1) == in`. The out `y` will be a vector  of
 length `out`, or a batch with `size(y) == (out, size(x)[2:end]...)`

The weight matrix and/or the bias vector (of length `out`) may be provided explicitly or generated randomly using `init` keyword argument.

# Keyword arguments
- `bias = false` will switch off trainable bias for the layer
- `init = glorot_uniform` specifies the function for initialization of
the weight matrix `W = init(out, in)`.
- `input_quantizer = identity` is a quantization function that is applied to the input
- `weight_quantizer = Sign()` is a quantization function that is applied to the weight matrix
- `weight_lims = nothing`
- `bias_lims = nothing`

"""
struct QuantDense{F, M<:AbstractMatrix, B, Q1, Q2}
    weight::M
    bias::B
    σ::F

    input_quantizer::Q1
    weight_quantizer::Q2
end

function QuantDense(
    weight::AbstractArray,
    bias,
    σ = identity;
    input_quantizer = identity,
    weight_quantizer = Sign(),
    weight_lims = nothing,
    bias_lims = nothing,
 )
    bias = _create_bias(weight, bias, size(weight,1))

    # use ClippedArray if there are limits
    if !isnothing(weight_lims)
        weight = ClippedArray(weight, weight_lims...)
    end
    if isa(bias, AbstractArray) && !isnothing(bias_lims)
        bias = ClippedArray(bias, bias_lims...)
    end
    return QuantDense(weight, bias, σ, input_quantizer, weight_quantizer)
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

QuantDense(l::Dense; kwargs...) = QuantDense(copy(l.weight), copy(l.bias), l.σ; kwargs...)

@functor QuantDense

function (l::QuantDense)(x::AbstractVecOrMat)
    σ = NNlib.fast_act(l.σ, x)  # replaces tanh => tanh_fast, etc
    xbin = l.input_quantizer(x)
    wbin = l.weight_quantizer(l.weight)

    return σ.(wbin * xbin .+ l.bias)
end

function (l::QuantDense)(x::AbstractArray)
    return reshape(l(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
end

# TODO improve printing
function Base.show(io::IO, l::QuantDense)
    print(io, "QuantDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
end
