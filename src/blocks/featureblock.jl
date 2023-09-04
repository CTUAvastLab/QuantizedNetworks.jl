"""
A custom struct representing a feature block. 
Holds a collection of layers which include feature quantization layers 
and an optional quantizer and where each layer performs a specific operation on the input data.
The struct is also a functor, so it can be used as a function.

## Constructor
You specify the dimensionality of the input features `dim` and the number of quantization levels `k`. 
Additionally, you can choose to include an extra quantizer layer for handling missing data by setting output_missing to true. 
The quantizer argument lets you specify the quantization function to be used (with a default of `Sign()`), 
and any additional keyword arguments are passed to the FeatureQuantizer constructor. 
"""
struct FeatureBlock <: AbstractBlock
    layers
end

function FeatureBlock(
    dim::Int,
    k::Int;
    output_missing::Bool = false,
    quantizer = Sign(),
    kwargs...
)

    layers = Any[FeatureQuantizer(dim, k; kwargs...)]
    if output_missing
        push!(layers, quantizer)
    end
    return FeatureBlock(Parallel(vcat, layers...))
end

@functor FeatureBlock
