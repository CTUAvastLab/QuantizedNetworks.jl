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
