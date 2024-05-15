"""
The `DenseBlock` module defines a custom block structure for neural networks.
It consists of a chain of layers, making it suitable for creating dense blocks in deep neural networks.
Encapsulates functionality for quantized dense layers, batch normalization, and output quantization.
It is defined to be a functor object.

## Constructor
Constructor creates a DenseBlock object containing a chain of layers.
It takes several optional arguments:
- (in, out) specifies the input and output dimensions.
- σ is an activation function (default is the identity function).
- weight_quantizer sets the quantizer for weights (default is a ternary quantizer).
- output_quantizer sets the quantizer for the layer's output (default is a sign quantizer).
- batchnorm determines whether batch normalization is applied (default is true).
It constructs a chain of layers including a quantized dense layer, optional batch normalization, and an output quantizer.
"""
struct DenseBlock <: AbstractBlock
    layers
end

function DenseBlock(
    (in, out)::Pair{<:Integer,<:Integer},
    σ=identity;
    weight_quantizer=Ternary(),
    output_quantizer=Sign(),
    batchnorm::Bool=true,
    kwargs...
)

    return DenseBlock(Chain(
        QuantizedDense(
            in => out,
            batchnorm ? identity : σ;
            quantizer=weight_quantizer,
            kwargs...
        ),
        batchnorm ? BatchNorm(out, σ) : identity,
        output_quantizer,
    ))
end

@functor DenseBlock

# Conversion to standard Dense layer
extract_dense(l::DenseBlock) = l.layers[1]
extract_batchnorm(l::DenseBlock) = l.layers[2]
extract_quantizer(l::DenseBlock) = l.layers[3]


"""
This function is overwritten from the `Flux` package converts a `DenseBlock` into a standard dense layer `Flux.Dense`
with quantized weights, adjusted biases, and the specified output quantization.

Extractors serve the purpose of extracting specific components from a `DenseBlock` like:
- `extract_dense` for quantized dense layer.
- `extract_batchnorm` for the optional batch normalization layer.
- `extract_quantizer` for the output quantization function.
"""
function Flux.Dense(l::DenseBlock)
    d = extract_dense(l)
    bn = extract_batchnorm(l)

    # dense params
    weight = l.quantizer(copy(l.weight))
    bias = copy(l.bias)
    if bn != identity
        bias = (sqrt.(bn.σ² .+ bn.ϵ) ./ bn.γ) .* bn.β .- bn.μ .+ bias
        # inds = findall(iszero, bn.γ)
        # bias[inds] .= bn.β[inds]
    end
    return Dense(weight, bias, extract_quantizer(l))
end