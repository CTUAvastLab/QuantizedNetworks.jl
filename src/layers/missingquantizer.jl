struct MissingQuantizer{Q}
    output_quantizer::Q

    MissingQuantizer(; output_quantizer::Q = Sign()) where {Q} = new{Q}(output_quantizer)
end

Flux.@functor MissingQuantizer

function (q::MissingQuantizer)(x)
    T = truetype(x)
    return q.output_quantizer(ifelse.(ismissing.(x), -one(T), one(T)))
end
