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
