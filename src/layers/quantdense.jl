"""
The `QuantDense` module defines a custom dense layer for neural networks that combines quantization and sparsity techniques for efficient inference. 
This module includes various constructors and methods to create and utilize quantized dense layers.

## Constructor

- weight (learnable weights used for the dense layer)
- bias (learnable biases used for the dense layer)
- σ (activation function applied to the layer's output)
- weight_quantizer (weight quantization function)
- weight_sparsifier (weight sparsification function)
- output_quantizer (output quantization function)
- batchnorm (optional batch normalization layer)

## Functor

`QuantDense` serves as a functor and it applies the  layer to the input data x.
It performs quantization of weights, sparsification, batch normalization (if enabled), and output quantization.
If necessary the function resahpes the input.

```julia
using Random; Random.seed!(3);
x = Float32.([1 2]);
kwargs = (;
    init = (dims...) -> ClippedArray(dims...; lo = -1, hi = 1),
    output_quantizer = Ternary(1),
    weight_quantizer = Sign(),
    weight_sparsifier = identity,
    batchnorm = true,
)

qd = QuantDense(1 => 2, identity; kwargs...)

qd(x)
```
"""
struct QuantDense{F, M, B, Q1, S, Q2, N}
    weight::M
    bias::B
    σ::F

    weight_quantizer::Q1
    weight_sparsifier::S
    output_quantizer::Q2
    batchnorm::N
end

function QuantDense(
    weight::AbstractArray,
    bias,
    σ = identity;
    weight_quantizer = Ternary(),
    weight_sparsifier = identity,
    output_quantizer = Sign(),
    batchnorm::Bool = true,
 )

    return QuantDense(
        weight,
        create_bias(weight, bias, size(weight,1)),
        batchnorm ? identity : σ,
        weight_quantizer,
        weight_sparsifier,
        output_quantizer,
        batchnorm ? BatchNorm(size(weight, 1), σ) : identity,
    )
end

function QuantDense(
    (in, out)::Pair{<:Integer, <:Integer},
    σ = identity;
    init = ClippedArray,
    bias = false,
    kwargs...
)
    return QuantDense(init(out, in), bias, σ; kwargs...)
end

function QuantDense(in::Integer, out::Integer, σ = identity; kwargs...)
    return QuantDense(in => out, σ; kwargs...)
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
    wqs = l.weight_sparsifier(wq)

    return l.output_quantizer(l.batchnorm(σ.(wqs * x .+ l.bias)))
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
    l.batchnorm == identity && push!(kwargs, "batchnorm=false")
    print(io, "; ", join(kwargs, ", "), ")")
end

"""
This function converts a QuantDense layer to a logic-compatible layer by applying the weight quantization and sparsification techniques.
It returns a Dense layer suitable for efficient inference.
"""
function nn2logic(layer::QuantDense)
    bn = layer.batchnorm
    W = layer.weight_quantizer(layer.weight)
    W = layer.weight_sparsifier(W)
    b = bn.β .* sqrt.(bn.σ² .+ bn.ϵ) ./ bn.γ .- bn.μ
    b = floor.(b) .+ 0.5
    Dense(W, b, layer.output_quantizer)
end
