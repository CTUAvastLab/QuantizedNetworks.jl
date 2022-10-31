struct MissingQuantizer{Q}
    output_quantizer::Q

    MissingQuantizer(; output_quantizer::Q = Sign()) where {Q} = new{Q}(output_quantizer)
end

Flux.@functor MissingQuantizer

function (q::MissingQuantizer)(x)
    y = zero(x)
    y .= ifelse.(ismissing.(x), -1, 1)
    return q.output_quantizer(y)
end
