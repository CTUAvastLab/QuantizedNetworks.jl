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

    q = FeatureQuantizer(dim, k; kwargs...)
    return if output_missing
        Parallel(vcat, q, MissingQuantizer(; quantizer))
    else
        q
    end
end

@functor FeatureBlock
